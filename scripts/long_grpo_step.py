"""One long-answer GRPO update with sequential matched-shape answer forwards.

Generate every prescribed answer before computing group advantages. All
probability paths use batch one and native prefill/one-token cache forwards.
Each answer contributes loss/number_of_answers; gradients accumulate before
one clipping operation and one AdamW/scheduler update. No history is detached.
"""

import time

import torch

from cached_block_score import score_completion_logps_block_checkpointed
from cached_training_rollout import sample_completions_cached_training
from check_grpo_continuation import prompt_token_ids
from check_grpo_training import likelihood_gate
from grpo_math import completion_token_mask, group_advantages, grpo_loss
from gsm8k_data import configured_answer_rule, score_answer


def one_step(model, reference, tokenizer, questions, optimizer, scheduler,
             rollout_generator, config, report, label, deadline, *,
             device="cuda", frozen_dtype=torch.bfloat16):
    """Keep the complete sampled group, then update once from all its answers."""
    started = time.perf_counter()
    prompt_ids = prompt_token_ids(tokenizer, questions, config)
    total_answers = len(prompt_ids)
    group_size = config["answers_per_question"]
    rule = configured_answer_rule(config)
    block_size = config.get("checkpoint_block_tokens", 16)
    answers = []
    rollouts = []
    sampling_progress = dict(label=label, completed_answers=answers,
                             expected_answers=total_answers,
                             incomplete_answer_valid_tokens_lower_bound=0,
                             partial_answer_token_count_exact=False)
    report.data["current_stage"] = label + "_sequential_rollout"
    report.data["current_partial_rollout"] = sampling_progress
    report.write()
    print(label, "questions:", [row["id"] for row in questions], flush=True)

    def check_deadline(stage):
        if time.monotonic() >= deadline:
            report.data["deadline_stage"] = stage
            report.write()
            raise TimeoutError("Long-answer deadline before optimizer update at " + stage)

    sampling_started = time.perf_counter()
    for index, tokens in enumerate(prompt_ids):
        check_deadline("sample_answer_" + str(index + 1))
        sampling_progress.update(current_answer_index=index,
                                 incomplete_answer_valid_tokens_lower_bound=0,
                                 partial_answer_token_count_exact=False)

        def observe_sampling(progress):
            sampling_progress["incomplete_answer_valid_tokens_lower_bound"] = progress["generated_steps"]
            sampling_progress["partial_answer_token_count_exact"] = progress["active_rows"] == 0
            report.write()
            print(label, "ROLLOUT:", dict(answer=index + 1, **progress), flush=True)

        rollout = sample_completions_cached_training(
            model, [tokens], tokenizer.pad_token_id, [tokenizer.eos_token_id],
            config["max_new_tokens"], device, frozen_dtype,
            temperature=config["temperature"], generator=rollout_generator,
            progress_callback=observe_sampling, deadline=deadline)
        row = questions[index // group_size]
        ids = rollout["completion_ids"][0][rollout["completion_mask"][0]].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        grading = score_answer(text, row["target"], rule)
        ended = bool(rollout["ended_with_eos"][0].item())
        reward = grading["reward"] if ended else config["truncated_answer_reward"]
        answers.append(dict(question_id=row["id"], question=row["question"], completion=text,
                            token_ids=ids, token_count=len(ids), ended_with_eos=ended,
                            truncated=not ended, **grading, training_reward=reward))
        rollouts.append(rollout)
        sampling_progress.update(completed_answers=answers, completed_answer_count=len(answers),
                                 completed_valid_tokens=sum(answer["token_count"] for answer in answers),
                                 incomplete_answer_valid_tokens_lower_bound=0,
                                 partial_answer_token_count_exact=True)
        report.write()
        check_deadline("completed_sample_answer_" + str(index + 1))
    rollout_seconds = time.perf_counter() - sampling_started
    rewards = [answer["training_reward"] for answer in answers]
    report.data["physical_rollouts"] += 1
    report.data["current_completed_rollout"] = dict(
        label=label, question_ids=[row["id"] for row in questions], answers=answers,
        rewards=rewards, rollout_seconds=rollout_seconds)
    report.data.pop("current_partial_rollout", None)
    report.write()
    grouped_rewards = torch.tensor(rewards, device=device).reshape(len(questions), group_size)
    advantages = group_advantages(grouped_rewards, config["advantage_epsilon"]).flatten()
    # Only optimizer state changes at the final completed-group boundary.
    optimizer.zero_grad(set_to_none=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss_totals = dict(loss=0.0, policy_loss=0.0, kl=0.0)
    clip_token_total = 0.0
    completed_tokens = 0
    scoring_seconds = 0.0
    backward_seconds = 0.0
    old_rows = []
    reference_rows = []
    behavior_rows = []
    mask_rows = []
    alignments = []
    checkpoint_cases = []
    report.data["current_partial_gradient_accumulation"] = dict(
        label=label, expected_answers=total_answers, completed_backward_answers=0,
        optimizer_update_performed=False)

    for index, rollout in enumerate(rollouts):
        check_deadline("before_probability_answer_" + str(index + 1))
        mask = rollout["completion_mask"]
        width = rollout["prompt_width"]
        independent = completion_token_mask(
            rollout["input_ids"], rollout["attention_mask"],
            torch.full((1,), width, device=device), [tokenizer.eos_token_id])[:, width:]
        prefix = label + "_answer_" + str(index + 1)
        report.require(prefix + "_causal_completion_mask", torch.equal(mask, independent))
        statistics = {}
        report.data["current_partial_gradient_accumulation"].update(
            current_answer_index=index, current_answer_checkpoint_statistics=statistics,
        )

        def recompute(scoring_model, stage, gradient_enabled):
            report.data["current_stage"] = prefix + "_" + stage
            report.write()

            def allocated_memory_gib():
                return (torch.cuda.memory_allocated(device) / 2**30
                        if torch.device(device).type == "cuda" else 0.0)

            def guard(progress):
                # Checkpoint backward invokes this closure again. Bind the
                # replay stage explicitly instead of relying on an observer
                # that intentionally runs only on the original forward.
                guard_stage = prefix + ("_backward_cache_recompute" if progress["recomputing"]
                                        else "_" + stage)
                check_deadline(guard_stage + "_" + progress["phase"]
                               + "_token_" + str(progress["step"]))
                allocated_gib = allocated_memory_gib()
                if allocated_gib > config["scoring_memory_guard_gib"]:
                    report.data["memory_guard_failure"] = dict(
                        stage=guard_stage, step=progress["step"],
                        phase=progress["phase"], recomputing=progress["recomputing"],
                        allocated_memory_gib=allocated_gib,
                        limit_gib=config["scoring_memory_guard_gib"])
                    report.write()
                    raise RuntimeError("Functional cached scoring exceeded its explicit memory guard")

            def observe(progress):
                allocated_gib = allocated_memory_gib()
                if progress["step"] % 64 == 0:
                    summary = dict(answer=index + 1, step=progress["step"],
                                   allocated_memory_gib=allocated_gib)
                    report.data["last_scoring_progress"] = dict(
                        stage=report.data["current_stage"], **summary)
                    report.write()
                    print(label, stage, summary, flush=True)

            return score_completion_logps_block_checkpointed(
                scoring_model, rollout, device, frozen_dtype,
                gradient_enabled=gradient_enabled, trace_callback=observe, runtime_guard=guard,
                checkpoint_statistics=statistics if gradient_enabled else None,
                block_size=block_size)

        scoring_started = time.perf_counter()
        old_logp = recompute(model, "old_probabilities", False)
        reference_logp = recompute(reference, "reference_probabilities", False)
        sampled_alignment = likelihood_gate(
            report, prefix + "_sampled_and_old_likelihood", old_logp,
            rollout["behavior_logps"], mask, config)
        if label == "step_1":
            likelihood_gate(report, prefix + "_initial_policy_matches_original_reference",
                            old_logp, reference_logp, mask, config)
        current_logp = recompute(model, "current_probabilities", True)
        likelihood_gate(report, prefix + "_current_and_old_likelihood",
                        current_logp, old_logp, mask, config)
        terms = grpo_loss(current_logp, old_logp, reference_logp,
                          advantages[index:index + 1], mask,
                          config["clip_epsilon"], config["kl_coefficient"])
        report.require(prefix + "_finite_loss", all(torch.isfinite(terms[key]).item()
                       for key in ("loss", "policy_loss", "kl", "clip_fraction")),
                       loss=terms["loss"].item(), sampled_old_token_k3=terms["kl"].item())
        scoring_seconds += time.perf_counter() - scoring_started
        check_deadline("before_backward_answer_" + str(index + 1))
        backward_started = time.perf_counter()
        report.data["current_stage"] = prefix + "_backward"
        report.write()
        # Equal answer weighting preserves the original mean-of-token-means
        # loss when completion lengths differ. Clip once after the entire group.
        report.data["physical_backward_calls"] += 1
        (terms["loss"] / total_answers).backward()
        backward_seconds += time.perf_counter() - backward_started
        report.require(prefix + "_functional_cache_blocks_recomputed_with_matching_autocast",
                       statistics["recompute_blocks"] == statistics["checkpoint_blocks"]
                       and statistics["autocast_states_match"]
                       and statistics["runtime_guard_forward_checks"]
                       >= statistics["forward_cache_forwards"]
                       and statistics["runtime_guard_recompute_checks"]
                       >= statistics["recompute_cache_forwards"] > 0, **statistics)
        report.require(prefix + "_accumulated_FP32_AB_gradients_finite",
                       all(p.grad is not None and p.grad.dtype == torch.float32
                           and torch.isfinite(p.grad).all().item() for p in trainable))
        count = int(mask.sum().item())
        for name in loss_totals:
            loss_totals[name] += terms[name].item() / total_answers
        clip_token_total += terms["clip_fraction"].item() * count
        completed_tokens += count
        alignments.append(sampled_alignment)
        checkpoint_cases.append(dict(answer=index + 1, **statistics))
        old_rows.append(old_logp[0].tolist())
        reference_rows.append(reference_logp[0].tolist())
        behavior_rows.append(rollout["behavior_logps"][0].tolist())
        mask_rows.append(mask[0].tolist())
        report.data["current_partial_gradient_accumulation"]["completed_backward_answers"] = index + 1
        report.write()
        # All tensors holding a differentiable history are released now. Only
        # the accumulated A/B gradients and fixed sampled-token data survive.
        del current_logp, old_logp, reference_logp, terms
        check_deadline("completed_backward_answer_" + str(index + 1))

    check_deadline("before_group_optimizer_update")
    report.data["current_stage"] = label + "_optimizer_update"
    report.write()
    previous = [parameter.detach().clone() for parameter in trainable]
    update_started = time.perf_counter()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable, config["gradient_clip_norm"], error_if_nonfinite=True).item()
    optimizer.step()
    scheduler.step()
    report.data["physical_optimizer_step_calls"] += 1
    changed = sum(not torch.equal(parameter, before) for parameter, before in zip(trainable, previous))
    report.require(label + "_finite_AB_parameters", all(torch.isfinite(p).all().item()
                   for p in trainable), changed_parameter_tensors=changed,
                   gradient_norm_before_clipping=gradient_norm)
    optimizer.zero_grad(set_to_none=True)
    backward_seconds += time.perf_counter() - update_started
    report.data.pop("current_partial_gradient_accumulation", None)
    sequence_differences = [value["sequence_logp_differences"][0] for value in alignments]
    alignment = dict(
        max_absolute_logp_difference=max(value["max_absolute_logp_difference"] for value in alignments),
        mean_absolute_logp_difference=sum(value["mean_absolute_logp_difference"] * answer["token_count"]
            for value, answer in zip(alignments, answers)) / completed_tokens,
        max_absolute_ratio_minus_one=max(value["max_absolute_ratio_minus_one"] for value in alignments),
        sequence_logp_differences=sequence_differences)
    record = dict(
        label=label, question_ids=[row["id"] for row in questions], answers=answers,
        rewards=grouped_rewards.tolist(), advantages=advantages.tolist(),
        equal_reward_groups=(grouped_rewards.std(1, correction=0) == 0).tolist(),
        parse_failure_count=sum(answer["predicted_answer"] is None for answer in answers),
        truncated_answer_count=sum(answer["truncated"] for answer in answers),
        valid_completion_tokens=completed_tokens,
        fixed_input_width=max(rollout["fixed_width"] for rollout in rollouts),
        loss=loss_totals["loss"], policy_loss=loss_totals["policy_loss"],
        sampled_old_token_k3=loss_totals["kl"], clip_fraction=clip_token_total / completed_tokens,
        gradient_norm_before_clipping=gradient_norm, changed_parameter_tensors=changed,
        learning_rate=optimizer.param_groups[0]["lr"], rollout_seconds=rollout_seconds,
        scoring_seconds=scoring_seconds, backward_and_step_seconds=backward_seconds,
        likelihood_alignment=alignment, behavior_logp=behavior_rows, old_logp=old_rows,
        reference_logp=reference_rows, completion_mask=mask_rows,
        rollout_backend=config["rollout_backend"],
        rollout_path=rollouts[0]["path"] + "_sequential_batch1",
        recompute_function="cached_block_score.score_completion_logps_block_checkpointed",
        attention_checkpoint_statistics={}, functional_block_checkpoint_statistics=checkpoint_cases,
        effective_question_batch=len(questions), answers_per_question=group_size,
        generation_microbatch_size=1, gradient_microbatch_size=1,
        gradient_accumulation_answer_count=total_answers,
        mean_of_per_answer_token_means=True)
    print(label, "rewards:", record["rewards"], "grad:", gradient_norm,
          "seconds:", round(time.perf_counter() - started, 2), flush=True)
    return record
