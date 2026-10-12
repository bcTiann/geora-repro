"""Validate answer-weighted gradient accumulation independently of Qwen.

A trainable softmax model supplies probabilities. Four different completion
lengths and signed group advantages exercise clipping and reference KL. The
full-batch mean and four sequential loss/4 backward calls must agree, followed
by one common gradient clipping and one AdamW/scheduler update.
"""

import copy
import tempfile
import json
import os
from pathlib import Path
import sys
import time

import torch

REPOSITORY = Path(os.environ.get("GEORA_REPOSITORY", Path.cwd())).resolve()
if not (REPOSITORY / "scripts/grpo_math.py").exists():
    REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))
from grpo_math import group_advantages, grpo_loss


def selected_logps(parameters, features, targets):
    logits = features @ parameters
    return logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.manual_seed(1901)
    parameters = torch.nn.Parameter(torch.randn(3, 9, dtype=torch.float64) * 0.2)
    micro_parameters = torch.nn.Parameter(parameters.detach().clone())
    features = torch.randn(4, 7, 3, dtype=torch.float64)
    targets = torch.randint(0, 9, (4, 7))
    lengths = torch.tensor([1, 4, 7, 3])
    mask = torch.arange(7)[None, :] < lengths[:, None]
    advantages = group_advantages(torch.tensor([[1.0, 0.0, 1.0, 0.0]])).flatten()
    initial = selected_logps(parameters, features, targets).detach()
    delta = torch.tensor([0.45, -0.35, -0.4, 0.5], dtype=torch.float64)[:, None]
    old = initial - delta
    reference = initial + torch.linspace(-0.2, 0.3, 7)[None, :]
    full_optimizer = torch.optim.AdamW([parameters], lr=1e-3, weight_decay=0.0)
    micro_optimizer = torch.optim.AdamW([micro_parameters], lr=1e-3, weight_decay=0.0)
    full_scheduler = torch.optim.lr_scheduler.LambdaLR(full_optimizer, lambda _: 1.0)
    micro_scheduler = torch.optim.lr_scheduler.LambdaLR(micro_optimizer, lambda _: 1.0)
    full_terms = grpo_loss(selected_logps(parameters, features, targets), old,
                           reference, advantages, mask, kl_coefficient=0.07)
    full_terms["loss"].backward()
    total_loss = 0.0
    micro_losses = []
    for row in range(4):
        current = selected_logps(micro_parameters, features[row:row + 1], targets[row:row + 1])
        terms = grpo_loss(current, old[row:row + 1], reference[row:row + 1],
                          advantages[row:row + 1], mask[row:row + 1], kl_coefficient=0.07)
        (terms["loss"] / 4).backward()
        total_loss += terms["loss"].item() / 4
        micro_losses.append(terms["loss"].item())
    gradient_difference = (parameters.grad - micro_parameters.grad).abs().max().item()
    checks = [
        dict(name="unequal_completion_lengths_and_signed_advantages", passed=len(set(lengths.tolist())) == 4
             and (advantages > 0).any().item() and (advantages < 0).any().item(),
             lengths=lengths.tolist(), advantages=advantages.tolist()),
        dict(name="clipping_and_KL_are_exercised", passed=full_terms["clip_fraction"].item() > 0
             and full_terms["kl"].item() > 0, clip_fraction=full_terms["clip_fraction"].item(),
             sampled_reference_KL=full_terms["kl"].item()),
        dict(name="mean_of_per_answer_token_means_loss_equal", passed=abs(total_loss - full_terms["loss"].item()) < 1e-12,
             full_loss=full_terms["loss"].item(), accumulated_loss=total_loss),
        dict(name="four_micro_backward_gradients_equal_full_batch", passed=bool(torch.allclose(
             parameters.grad, micro_parameters.grad, rtol=1e-12, atol=1e-12)),
             max_absolute_gradient_difference=gradient_difference),
    ]
    full_norm = torch.nn.utils.clip_grad_norm_([parameters], 0.03)
    micro_norm = torch.nn.utils.clip_grad_norm_([micro_parameters], 0.03)
    full_optimizer.step()
    micro_optimizer.step()
    full_scheduler.step()
    micro_scheduler.step()
    checks.append(dict(name="one_group_clip_AdamW_scheduler_update_equal", passed=bool(torch.allclose(
        parameters, micro_parameters, rtol=1e-12, atol=1e-12))
        and full_scheduler.state_dict() == micro_scheduler.state_dict()
        and bool(torch.isclose(full_norm, micro_norm, rtol=1e-12, atol=1e-12)),
        gradient_norm_before_clipping=full_norm.item(),
        max_absolute_parameter_difference=(parameters - micro_parameters).abs().max().item()))
    full_state = full_optimizer.state[parameters]
    micro_state = micro_optimizer.state[micro_parameters]
    checks.append(dict(name="one_AdamW_step_and_moments_equal", passed=all(torch.allclose(
        full_state[name], micro_state[name], rtol=1e-12, atol=1e-12)
        for name in ("step", "exp_avg", "exp_avg_sq")) and full_state["step"].item() == 1))
    status = "passed" if all(case["passed"] for case in checks) else "failed"
    report = dict(status=status, stage="independent_long_GRPO_accumulation_math_CPU", checks=checks,
                  physical_backward_calls=4, optimizer_steps=1,
                  dtype="FP64 toy softmax model for stringent algebra comparison; production precision unchanged",
                  scope="Loss weighting, gradients, clipping and optimizer semantics; not full-Qwen GPU acceptance",
                  elapsed_seconds=time.perf_counter() - started)
    path = Path(os.environ.get("GEORA_CHECK_OUTPUT", str(Path(tempfile.gettempdir()) / "geora_accumulation_cpu.json")))
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if status != "passed":
        raise RuntimeError("Independent accumulation check failed")


if __name__ == "__main__":
    main()
