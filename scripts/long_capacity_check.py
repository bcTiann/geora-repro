"""Full native-cache forward/backward capacity gate without an update.

The caller must keep its already-loaded frozen reference resident while this
function uses the policy. This artificial teacher-forced sequence measures
memory and execution capacity; it is not sampling, GRPO, or task accuracy.
The CUDA gate uses 1024 prompt tokens and 4096 non-EOS completion tokens.
Shorter CPU configurations exercise the same path as a small wiring fixture.
"""

import hashlib
import math
import time
import traceback

import torch

from cached_block_score import score_completion_logps_block_checkpointed
from geora_layers import GeoRALinear, TARGET_PATTERN


def check_long_capacity(
    policy, tokenizer, config, report, deadline, device="cuda", frozen_dtype=torch.bfloat16,
):
    """Score a fixed maximal sequence, backpropagate, then clear its gradients.

    No optimizer is accepted or used. Entry requires empty gradients, so clearing
    temporary gradients restores the caller's state. CPU snapshots verify that
    every A/B parameter is unchanged without adding a GPU snapshot allocation.
    RNG state is preserved even if a runtime or memory guard interrupts replay.
    """
    started = time.perf_counter()
    target_device = torch.device(device)
    cuda = target_device.type == "cuda"
    prompt_tokens = config["max_prompt_tokens"]
    completion_tokens = config["max_new_tokens"]
    block_tokens = config["checkpoint_block_tokens"]
    memory_limit_gib = config["scoring_memory_guard_gib"]
    statistics = {}
    measurement = dict(
        status="running", passed=False,
        scope="Artificial teacher-forced native-cache capacity only; no task accuracy or GRPO update",
        device=str(target_device), frozen_dtype=str(frozen_dtype),
        synthetic_prompt_tokens=prompt_tokens, synthetic_completion_tokens=completion_tokens,
        fixed_input_width=prompt_tokens + completion_tokens,
        expected_final_cache_length=prompt_tokens + completion_tokens - 1,
        checkpoint_block_tokens=block_tokens, checkpoint_statistics=statistics,
        synthetic_token_text=" apples", batch_size=1, optimizer_steps=0,
        backward_calls=0, gradients_cleared=False, parameters_unchanged=False,
        frozen_reference_forward_calls=0,
        reference_residency_requirement="Caller retains its already-loaded frozen reference on the same device",
        full_1024_prompt_4096_completion_gate=(cuda and prompt_tokens == 1024
                                              and completion_tokens == 4096 and block_tokens == 16),
        cpu_fixture_only=not cuda, memory_guard_gib=memory_limit_gib,
        GPU_memory_measurements_available=cuda,
    )
    report.data["capacity_check"] = measurement
    report.data["current_stage"] = "capacity_check_prepare"
    report.write()
    original_mode = policy.training
    original_optimizer_steps = report.data.get("physical_optimizer_step_calls", 0)
    trainable = [(name, parameter) for name, parameter in policy.named_parameters()
                 if parameter.requires_grad]
    snapshot = {}
    logps = None
    loss = None
    rollout = None
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = None
    cuda_index = None
    clear_temporary_gradients = False

    def sync_device():
        if cuda:
            torch.cuda.synchronize(target_device)

    def memory_measurements():
        if not cuda:
            return dict(allocated_memory_gib=None, reserved_memory_gib=None,
                        peak_allocated_memory_gib=None, peak_reserved_memory_gib=None)
        return dict(
            allocated_memory_gib=torch.cuda.memory_allocated(target_device) / 2**30,
            reserved_memory_gib=torch.cuda.memory_reserved(target_device) / 2**30,
            peak_allocated_memory_gib=torch.cuda.max_memory_allocated(target_device) / 2**30,
            peak_reserved_memory_gib=torch.cuda.max_memory_reserved(target_device) / 2**30,
        )

    def check_limits(stage, progress=None, *, memory_check=True):
        if time.monotonic() >= deadline:
            measurement.update(deadline_stage=stage)
            report.data["deadline_stage"] = stage
            report.write()
            raise TimeoutError("Long capacity deadline reached at " + stage)
        if not memory_check:
            return
        usage = memory_measurements()
        measurement.update(usage)
        # The peak catches a transient allocation released inside the model
        # before after_forward runs. Current-only checks can miss that peak.
        if cuda and usage["peak_allocated_memory_gib"] > memory_limit_gib:
            failure = dict(stage=stage, limit_gib=memory_limit_gib, **usage)
            if progress is not None:
                failure.update(progress)
            measurement["memory_guard_failure"] = failure
            report.data["memory_guard_failure"] = failure
            report.write()
            raise RuntimeError("Long capacity check exceeded its explicit memory guard")

    def guard(progress):
        phase = "backward_cache_recompute" if progress["recomputing"] else "forward"
        stage = "capacity_" + phase + "_" + progress["phase"] + "_token_" + str(progress["step"])
        check_limits(stage, progress)
        if progress["phase"] == "after_forward" and progress["step"] % 64 == 0:
            measurement["last_progress"] = dict(stage=stage, **progress)
            report.write()
            print("CAPACITY:", dict(stage=stage, step=progress["step"],
                                     peak_allocated_memory_gib=measurement["peak_allocated_memory_gib"]),
                  flush=True)

    try:
        if not (isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
                and prompt_tokens >= 1 and isinstance(completion_tokens, int)
                and not isinstance(completion_tokens, bool) and 1 <= completion_tokens <= 4096
                and isinstance(block_tokens, int) and not isinstance(block_tokens, bool)
                and block_tokens >= 1 and math.isfinite(memory_limit_gib) and memory_limit_gib > 0):
            raise ValueError("Capacity dimensions and memory limit must be positive and bounded")
        if cuda and (prompt_tokens, completion_tokens, block_tokens) != (1024, 4096, 16):
            raise ValueError("The full GPU capacity gate requires 1024 prompt + 4096 completion, block16")
        if cuda and frozen_dtype != torch.bfloat16:
            raise ValueError("The full GPU capacity gate requires the reviewed BF16 frozen model")
        if cuda and memory_limit_gib > 55:
            raise ValueError("The full GPU capacity gate cannot relax the reviewed 55 GiB guard")
        if frozen_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Capacity activations use FP32 or reviewed BF16, never FP64")
        if target_device.type not in ("cuda", "cpu"):
            raise ValueError("Capacity checking supports CUDA production or a short CPU fixture")
        if any(parameter.grad is not None for parameter in policy.parameters()):
            raise ValueError("Capacity checking requires empty gradients at entry")
        clear_temporary_gradients = True
        if not trainable or any(parameter.dtype != torch.float32 for _, parameter in trainable):
            raise ValueError("Capacity checking requires FP32 A/B parameters")
        target_layers = [(name, layer) for name, layer in policy.named_modules()
                         if isinstance(layer, GeoRALinear) and TARGET_PATTERN.search(name)]
        expected_names = {name + "." + factor for name, layer in target_layers for factor in ("A", "B")}
        expected_count = policy.config.num_hidden_layers * 7 * 2
        if ({name for name, _ in trainable} != expected_names
                or len(trainable) != expected_count
                or any(not isinstance(layer, GeoRALinear) for _, layer in target_layers)):
            raise ValueError("Capacity gate requires all seven GeoRA projections and only their A/B trainable")
        if cuda and expected_count != 392:
            raise ValueError("The full Qwen capacity gate requires all 392 trainable A/B tensors")
        adapter_parameter_count = sum(parameter.numel() for _, parameter in trainable)
        if cuda and adapter_parameter_count != 18464768:
            raise ValueError("The full Qwen rank16 capacity gate requires 18,464,768 A/B elements")
        if any(parameter.dtype != frozen_dtype for parameter in policy.parameters()
               if not parameter.requires_grad):
            raise ValueError("Frozen parameter dtype differs from the declared capacity precision")
        check_limits("capacity_before_prepare", memory_check=False)
        sync_device()
        if cuda:
            cuda_index = target_device.index if target_device.index is not None else torch.cuda.current_device()
            cuda_rng = torch.cuda.get_rng_state(cuda_index).clone()
            torch.cuda.reset_peak_memory_stats(target_device)
        check_limits("capacity_after_peak_reset")
        measurement.update(starting_memory=memory_measurements(),
                           expected_adapter_tensor_count=expected_count,
                           adapter_parameter_count=adapter_parameter_count)
        snapshot = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
        measurement["adapter_snapshot_bytes_CPU"] = sum(
            tensor.numel() * tensor.element_size() for tensor in snapshot.values())
        eos_ids = tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        generation_eos = getattr(policy.generation_config, "eos_token_id", None)
        generation_eos = [generation_eos] if isinstance(generation_eos, int) else list(generation_eos or [])
        excluded = set(getattr(tokenizer, "all_special_ids", None) or []) | set(eos_ids) | set(generation_eos)
        pattern_ids = [token for token in tokenizer.encode(" apples", add_special_tokens=False)
                       if isinstance(token, int) and token not in excluded
                       and 0 <= token < policy.config.vocab_size]
        if not pattern_ids:
            raise ValueError("Synthetic token pattern has no non-special non-EOS tokens")
        width = prompt_tokens + completion_tokens
        pattern = torch.tensor(pattern_ids, dtype=torch.long)
        cpu_ids = pattern.repeat((width + len(pattern_ids) - 1) // len(pattern_ids))[:width].reshape(1, width)
        measurement.update(pattern_token_ids=pattern_ids,
                           input_ids_sha256=hashlib.sha256(cpu_ids.numpy().tobytes()).hexdigest(),
                           contains_EOS=False, generated_steps=completion_tokens)
        ids = cpu_ids.to(target_device)
        rollout = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                       completion_mask=torch.ones((1, completion_tokens), dtype=torch.bool, device=target_device),
                       prompt_width=prompt_tokens, max_new_tokens=completion_tokens,
                       generated_steps=completion_tokens, temperature=config.get("temperature", 1.0),
                       use_cache=True)
        policy.zero_grad(set_to_none=True)
        report.data["current_stage"] = "capacity_check_forward"
        report.write()
        scoring_started = time.perf_counter()
        with torch.random.fork_rng(devices=[cuda_index] if cuda else []):
            logps = score_completion_logps_block_checkpointed(
                policy, rollout, device, frozen_dtype, gradient_enabled=True,
                runtime_guard=guard, checkpoint_statistics=statistics, block_size=block_tokens,
            )
            sync_device()
            measurement["scoring_seconds"] = time.perf_counter() - scoring_started
            check_limits("capacity_after_forward")
            finite_logps = torch.isfinite(logps).all().item()
            measurement.update(all_selected_logps_finite=finite_logps,
                               selected_logp_dtype=str(logps.dtype))
            report.require("capacity_all_selected_logps_finite", finite_logps,
                           completion_tokens=completion_tokens)
            measurement.update(selected_logp_min=logps.detach().min().item(),
                               selected_logp_max=logps.detach().max().item(),
                               max_absolute_selected_logp=logps.detach().abs().max().item())
            loss = -logps.mean()
            loss_value = loss.detach().item()
            measurement["synthetic_mean_negative_logp_loss"] = loss_value if math.isfinite(loss_value) else None
            report.require("capacity_synthetic_mean_loss_finite", math.isfinite(loss_value))
            report.data["current_stage"] = "capacity_check_backward"
            report.write()
            backward_started = time.perf_counter()
            check_limits("capacity_before_backward")
            loss.backward()
            sync_device()
            measurement["backward_calls"] = 1
            measurement["backward_seconds"] = time.perf_counter() - backward_started
            check_limits("capacity_after_backward")
            finite_gradients = sum(
                parameter.grad is not None and parameter.grad.dtype == torch.float32
                and torch.isfinite(parameter.grad).all().item()
                for _, parameter in trainable
            )
            measurement["finite_FP32_adapter_gradient_tensors"] = finite_gradients
            report.require("capacity_every_AB_gradient_finite_FP32",
                           finite_gradients == expected_count,
                           expected_adapter_tensor_count=expected_count,
                           finite_gradient_tensors=finite_gradients)
            gradient_norm = math.sqrt(sum(parameter.grad.norm().item() ** 2 for _, parameter in trainable))
            measurement["adapter_gradient_l2_norm"] = gradient_norm if math.isfinite(gradient_norm) else None
            report.require("capacity_nonzero_finite_adapter_gradient_norm",
                           math.isfinite(gradient_norm) and gradient_norm > 0)
            expected_blocks = (completion_tokens + block_tokens - 1) // block_tokens
            replay_ok = (statistics["checkpoint_blocks"] == expected_blocks
                         and statistics["recompute_blocks"] == expected_blocks
                         and statistics["forward_cache_forwards"] == completion_tokens
                         and statistics["recompute_cache_forwards"] == completion_tokens
                         and statistics["autocast_states_match"]
                         and statistics["boundary_cache_inputs_are_explicit"]
                         and statistics["runtime_guard_forward_checks"] >= completion_tokens
                         and statistics["runtime_guard_recompute_checks"] >= completion_tokens)
            report.require("capacity_every_cache_block_replayed_with_guards", replay_ok,
                           expected_blocks=expected_blocks, **statistics)
            check_limits("capacity_after_gradient_checks")
        # Keep this verification outside fork_rng so the reported RNG property
        # refers to the caller's state after restoration, including failures.
        measurement["parameters_unchanged"] = all(
            torch.equal(parameter.detach().cpu(), snapshot[name]) for name, parameter in trainable)
        report.require("capacity_AB_parameters_identical_without_optimizer",
                       measurement["parameters_unchanged"]
                       and report.data.get("physical_optimizer_step_calls", 0) == original_optimizer_steps,
                       optimizer_steps=0)
        check_limits("capacity_after_parameter_verification")
    except TimeoutError as error:
        measurement.update(status="interrupted", passed=False, error=str(error))
        # Keep diagnostic stack locations while dropping completed scorer
        # frames that could otherwise retain activation tensors in the error.
        traceback.clear_frames(error.__traceback__)
        raise
    except Exception as error:
        measurement.update(status="failed", passed=False, error=str(error), error_type=type(error).__name__)
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        # Release activation histories and gradients before the caller saves a
        # clean step0 or starts its first real optimizer update.
        logps = None
        loss = None
        rollout = None
        if clear_temporary_gradients:
            policy.zero_grad(set_to_none=True)
        measurement["gradients_cleared"] = all(parameter.grad is None for _, parameter in trainable)
        measurement["model_training_mode_unchanged"] = policy.training == original_mode
        measurement["CPU_rng_restored"] = torch.equal(torch.get_rng_state(), cpu_rng)
        measurement["CUDA_rng_restored"] = (cuda_rng is None or
                                           torch.equal(torch.cuda.get_rng_state(cuda_index), cuda_rng))
        if snapshot and not measurement["parameters_unchanged"]:
            measurement["parameters_unchanged"] = all(
                torch.equal(parameter.detach().cpu(), snapshot[name]) for name, parameter in trainable)
        measurement.update(memory_measurements(), elapsed_seconds=time.perf_counter() - started)
        report.write()
    restored = (measurement["gradients_cleared"] and measurement["parameters_unchanged"]
                and measurement["CPU_rng_restored"] and measurement["CUDA_rng_restored"]
                and measurement["model_training_mode_unchanged"])
    try:
        report.require("capacity_temporary_gradients_cleared_and_rng_preserved", restored)
        check_limits("capacity_before_completion")
        measurement.update(status="passed", passed=True)
        report.data["current_stage"] = "capacity_check_completed"
        report.write()
    except TimeoutError as error:
        measurement.update(status="interrupted", passed=False, error=str(error))
        report.write()
        raise
    except Exception as error:
        measurement.update(status="failed", passed=False, error=str(error), error_type=type(error).__name__)
        report.write()
        raise
    return measurement
