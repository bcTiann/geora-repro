"""CPU hand checks for GRPO rewards, masks, surrogate loss and gradients.

Run: python tests/check_grpo_math.py --output-dir outputs/grpo-math-cpu
When staged outside the repository, grpo_math.py may be alongside this script.
These artificial examples verify formulas; they are not GSM8K training scores.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grpo_math import causal_token_log_probs, completion_token_mask, group_advantages, grpo_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    report = {
        "status": "running",
        "device": "cpu",
        "torch_version": torch.__version__,
        "scope": "Hand-computable artificial examples only; no task reward or model training.",
        "source": "https://arxiv.org/html/2402.03300v3#S4.SS1",
        "checks": [],
    }

    def passed(name: str, **metrics: object) -> None:
        report["checks"].append({"name": name, "passed": True, **metrics})
        print(f"PASS: {name}", flush=True)

    try:
        rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], requires_grad=True)
        advantages = group_advantages(rewards)
        expected = torch.tensor([[math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3)], [0.0] * 4])
        torch.testing.assert_close(advantages, expected, rtol=1e-6, atol=1e-7)
        assert not advantages.requires_grad
        passed("population_group_normalization_and_fixed_labels", advantages=advantages.tolist())

        old = torch.full((4, 1), -2.0, dtype=torch.float64, requires_grad=True)
        reference = old.detach().clone().requires_grad_()
        current = (old.detach() + torch.log(torch.tensor([[1.1], [1.3], [0.9], [0.7]], dtype=torch.float64))).requires_grad_()
        advantages = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float64, requires_grad=True)
        terms = grpo_loss(current, old, reference, advantages, torch.ones_like(current, dtype=torch.bool), kl_coefficient=0.0)
        torch.testing.assert_close(terms["per_token_surrogate"].flatten(), torch.tensor([2.2, 2.4, -1.8, -1.6], dtype=torch.float64))
        torch.testing.assert_close(terms["loss"], torch.tensor(-0.3, dtype=torch.float64))
        terms["loss"].backward()
        torch.testing.assert_close(current.grad.flatten(), torch.tensor([-0.55, 0.0, 0.45, 0.0], dtype=torch.float64))
        assert old.grad is None and reference.grad is None and advantages.grad is None
        passed("positive_and_negative_clipping_values_and_gradients", loss=terms["loss"].item(), current_gradients=current.grad.flatten().tolist())
        passed("old_reference_and_advantages_detached")

        current = torch.log(torch.tensor([[0.7], [1.3]], dtype=torch.float64)).requires_grad_()
        terms = grpo_loss(current, torch.zeros_like(current), torch.zeros_like(current), torch.tensor([2.0, -2.0]), torch.ones_like(current, dtype=torch.bool), kl_coefficient=0.0)
        terms["loss"].backward()
        torch.testing.assert_close(current.grad.flatten(), torch.tensor([-0.7, 1.3], dtype=torch.float64))
        assert terms["clip_fraction"].item() == 0.0
        passed("clipping_does_not_block_wrong_direction_recovery")

        input_ids = torch.tensor([[0, 2, 11, 7, 2, 8, 0], [0, 0, 10, 9, 8, 2, 0], [10, 11, 12, 13, 14, 15, 16]])
        attention = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0], [1] * 7])
        mask = completion_token_mask(input_ids, attention, torch.tensor([3, 3, 4]), [2])
        expected_mask = torch.tensor([[False, False, False, True, True, False, False], [False, False, False, True, True, True, False], [False, False, False, False, True, True, True]])
        assert torch.equal(mask, expected_mask)
        assert torch.equal(mask[:, 1:].sum(dim=1), torch.tensor([2, 3, 3]))
        passed("completion_mask_excludes_prompt_padding_and_post_eos_includes_first_eos", lengths=mask.sum(dim=1).tolist())

        same_eos_pad_ids = torch.tensor([[2, 10, 7, 2, 2]])
        same_eos_pad_mask = completion_token_mask(same_eos_pad_ids, torch.tensor([[0, 1, 1, 1, 0]]), torch.tensor([2]), [2])
        assert torch.equal(same_eos_pad_mask, torch.tensor([[False, False, True, True, False]]))
        passed("eos_equals_pad_uses_attention_to_distinguish_padding")

        logits = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [-10.0, 10.0, 0.0]]], requires_grad=True)
        ids = torch.tensor([[0, 2, 1]])
        actual = causal_token_log_probs(logits, ids)
        expected = torch.tensor([[2.0 - math.log(1.0 + math.e + math.e**2), 1.0 - math.log(math.e**2 + math.e + 1.0)]])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        actual.sum().backward()
        assert torch.equal(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1]))
        passed("causal_logits_predict_next_token_without_final_logit")

        current = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, False, False], [True, True, True]])
        terms = grpo_loss(current, torch.zeros_like(current), torch.zeros_like(current), torch.tensor([1.0, -1.0]), mask, kl_coefficient=0.0)
        assert terms["loss"].item() == 0.0
        terms["loss"].backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-0.5, 0.0, 0.0], [1 / 6, 1 / 6, 1 / 6]], dtype=torch.float64))
        passed("token_mean_then_completion_mean_not_global_token_mean")

        current = torch.tensor([[-1.0, float("nan"), float("inf")]], requires_grad=True)
        terms = grpo_loss(current, torch.tensor([[-1.0, float("nan"), float("inf")]]), torch.tensor([[-1.0, float("nan"), float("inf")]]), torch.tensor([1.0]), torch.tensor([[True, False, False]]))
        terms["loss"].backward()
        assert torch.isfinite(terms["loss"]) and torch.equal(current.grad, torch.tensor([[-1.0, 0.0, 0.0]]))
        passed("masked_nonfinite_positions_do_not_contaminate_loss_or_gradients")

        current = torch.tensor([[-1.0], [-2.0]], dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([[-2.0], [-1.0]], dtype=torch.float64, requires_grad=True)
        terms = grpo_loss(current, current.detach(), reference, torch.zeros(2), torch.ones_like(current, dtype=torch.bool), kl_coefficient=0.001)
        expected_kl = torch.tensor([[math.exp(-1)], [math.e - 2]], dtype=torch.float64)
        torch.testing.assert_close(terms["per_token_kl"], expected_kl)
        terms["loss"].backward()
        expected_gradient = 0.001 * torch.tensor([[1 - math.exp(-1)], [1 - math.e]], dtype=torch.float64) / 2
        torch.testing.assert_close(current.grad, expected_gradient)
        assert reference.grad is None
        passed("sampled_k3_value_and_direct_surrogate_gradient", kl_values=terms["per_token_kl"].tolist(), gradients=current.grad.tolist())

        for dtype in (torch.float32, torch.float64):
            tiny_difference = torch.tensor([[-1e-7, 0.0, 1e-7]], dtype=dtype)
            terms = grpo_loss(torch.zeros_like(tiny_difference), torch.zeros_like(tiny_difference), tiny_difference, torch.zeros(1), torch.ones_like(tiny_difference, dtype=torch.bool))
            assert torch.isfinite(terms["per_token_kl"]).all()
            assert torch.all(terms["per_token_kl"] >= 0)
            assert terms["per_token_kl"][0, 1].item() == 0.0
        passed("unreduced_k3_finite_nonnegative_near_zero_fp32_fp64")

        current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        identical_advantages = group_advantages(torch.ones((1, 2))).flatten()
        terms = grpo_loss(current, current.detach(), current.detach(), identical_advantages[:1], torch.ones_like(current, dtype=torch.bool))
        terms["loss"].backward()
        assert terms["loss"].item() == 0.0 and torch.equal(current.grad, torch.zeros_like(current))
        passed("identical_rewards_and_identical_reference_zero_policy_and_kl_signal")

        # Even an all-zero advantage group can get a KL gradient after policy drift.
        current = torch.tensor([[-1.0]], requires_grad=True)
        terms = grpo_loss(current, current.detach(), torch.tensor([[-2.0]]), torch.zeros(1), torch.ones_like(current, dtype=torch.bool))
        terms["loss"].backward()
        assert terms["policy_loss"].item() == 0.0 and current.grad.item() > 0.0
        passed("zero_advantage_can_still_have_reference_kl_gradient")

        invalid_cases = [
            (torch.tensor([[float("nan")]]), torch.ones((1, 1), dtype=torch.bool)),
            (torch.tensor([[-1.0]]), torch.zeros((1, 1), dtype=torch.bool)),
        ]
        for invalid_current, invalid_mask in invalid_cases:
            try:
                grpo_loss(invalid_current, torch.zeros_like(invalid_current), torch.zeros_like(invalid_current), torch.ones(1), invalid_mask)
            except ValueError:
                continue
            raise AssertionError("Invalid valid-token values or empty completions must fail.")
        passed("nonfinite_valid_tokens_and_empty_completions_rejected")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        if arguments.output_dir is not None:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = arguments.output_dir / "grpo_math_checks.json"
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Report: {output_path}", flush=True)
    print(f"GRPO CPU hand checks passed: {len(report['checks'])} checks.", flush=True)


if __name__ == "__main__":
    main()
