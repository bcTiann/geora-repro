"""CPU tests for three-way decomposition and restoration of learner ownership."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.nn import functional as F

from native_sync import source_guard
from three_way_probability import (
    canonical_merged_model, hf_sequence_logprobs, score_three_way,
    validate_fixed_rows, vllm_sequence_logprobs,
)


class GeoRALinear(nn.Module):
    def __init__(self, inputs=4, outputs=11):
        super().__init__()
        self.A = nn.Parameter(torch.randn(2, inputs))
        self.B = nn.Parameter(torch.randn(outputs, 2))
        self.register_buffer("W_res", torch.randn(outputs, inputs).bfloat16())
        self.register_buffer("bias", torch.randn(outputs).bfloat16())
        self.scaling = 0.5

    def forward(self, x):
        return (F.linear(x.bfloat16(), self.W_res, self.bias) +
                self.scaling * F.linear(F.linear(x.float(), self.A), self.B)).to(x.dtype)


class LoRALinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(4, 11, dtype=torch.bfloat16)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 11, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(0)})
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged_adapters = []
        self.fan_in_fan_out = False
        self.use_dora = {"default": False}
        self.scaling = {"default": 0.5}

    def forward(self, x):
        return (self.base_layer(x) + self.scaling["default"] *
                self.lora_B["default"](self.lora_A["default"](x.float()))).to(x.dtype)


class TinyModel(nn.Module):
    def __init__(self, kind="geora"):
        super().__init__()
        self.embedding = nn.Embedding(11, 4, dtype=torch.bfloat16)
        self.proj = GeoRALinear() if kind == "geora" else LoRALinear()
        self.fail_merged = False

    def forward(self, input_ids, use_cache=False):
        if self.fail_merged and isinstance(self.proj, nn.Linear):
            raise RuntimeError("injected merged forward failure")
        return SimpleNamespace(logits=self.proj(self.embedding(input_ids)))


class PeftWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.peft_config = {"default": {}}
        self.core = TinyModel("lora")

    def get_base_model(self):
        return self.core

    def forward(self, **kwargs):
        return self.core(**kwargs)


def schema():
    return {"embedding.weight": ((11, 4), torch.bfloat16),
            "proj.weight": ((11, 4), torch.bfloat16),
            "proj.bias": ((11,), torch.bfloat16)}


def rows():
    return [{"id": "short", "category": "short", "prompt_ids": [1, 2], "completion_ids": [3, 4, 0],
             "terminated_with_eos": True, "eos_token_id": 0},
            {"id": "long", "category": "long", "prompt_ids": [5, 6, 7, 8], "completion_ids": [9, 10, 0],
             "terminated_with_eos": True, "eos_token_id": 0}]


def client_for(values, fixed_rows, offset=0.0):
    client = Mock()
    client.get_sequence_logprobs.return_value = {
        "actual_logprobs": [[[value + offset] for value in row] for row in values],
        "actual_token_ids": [[[token] for token in row["completion_ids"]] for row in fixed_rows],
    }
    return client


class ThreeWayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)

    def test_geora_three_way_decomposition_and_parameter_identity(self):
        model = TinyModel()
        model.train()
        model.embedding.eval()  # Preserve heterogeneous modes as well as values.
        before = source_guard(model)
        parameters = list(model.parameters())
        original_proj = model.proj
        values = {k: v.clone() for k, v in model.state_dict().items()}
        with canonical_merged_model(model, schema(), device="cpu") as merged:
            reference = hf_sequence_logprobs(merged, rows(), device="cpu")
        result = score_three_way(model, schema(), rows(), client_for(reference, rows(), offset=0.01),
                                 device="cpu", version=0)
        self.assertEqual(before, source_guard(model))
        self.assertIs(model.proj, original_proj)
        self.assertTrue(all(a is b for a, b in zip(parameters, model.parameters(), strict=True)))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, values[key]))
        self.assertEqual(result["tokens"], 6)
        self.assertTrue(all(row["includes_eos"] for row in result["rows"]))
        self.assertAlmostEqual(result["comparisons"]["hf_merged_minus_vllm"]["mean_abs_logprob_delta"], 0.01)
        self.assertIsNone(result["numerical_accepted"])
        for actual, expected in zip(result["rows"], reference, strict=True):
            self.assertEqual(actual["hf_merged_logprobs"], expected)

    def test_peft_wrapper_uses_root_names_and_restores_adapters(self):
        model = PeftWrapper()
        original = model.core.proj
        before = source_guard(model)
        with canonical_merged_model(model, schema(), device="cpu") as merged:
            self.assertIs(merged, model.core)
            self.assertIsInstance(merged.proj, nn.Linear)
            reference = hf_sequence_logprobs(merged, rows(), device="cpu")
        result = score_three_way(model, schema(), rows(), client_for(reference, rows()), device="cpu")
        self.assertTrue(result["structural_checks_passed"])
        self.assertIs(model.core.proj, original)
        self.assertEqual(before, source_guard(model))

    def test_failure_in_merged_forward_restores_originals(self):
        model = TinyModel()
        model.fail_merged = True
        original = model.proj
        before = source_guard(model)
        with self.assertRaisesRegex(RuntimeError, "injected merged forward"):
            score_three_way(model, schema(), rows(), Mock(), device="cpu")
        self.assertIs(model.proj, original)
        self.assertEqual(before, source_guard(model))

    def test_failure_during_partial_preparation_restores_originals(self):
        model = TinyModel()
        original = model.proj
        altered = schema()
        altered["embedding.weight"] = ((11, 4), torch.float32)
        before = source_guard(model)
        with self.assertRaisesRegex(ValueError, "Temporary canonical HF tensor"):
            with canonical_merged_model(model, altered, device="cpu"):
                self.fail("Should reject inconsistent frozen tensor dtype")
        self.assertIs(model.proj, original)
        self.assertEqual(before, source_guard(model))

    def test_server_must_score_actual_requested_token_including_eos(self):
        values = [[-1.0] * 3, [-2.0] * 3]
        client = client_for(values, rows())
        client.get_sequence_logprobs.return_value["actual_token_ids"][0][-1] = [7]
        with self.assertRaisesRegex(ValueError, "omitted or changed"):
            vllm_sequence_logprobs(client, rows())
        client = client_for(values, rows())
        client.get_sequence_logprobs.return_value["actual_logprobs"][0][-1] = [None]
        with self.assertRaisesRegex(ValueError, "invalid actual-token"):
            vllm_sequence_logprobs(client, rows())

    def test_teacher_forcing_uses_prediction_at_previous_position(self):
        class PositionModel(nn.Module):
            def forward(self, input_ids, use_cache=False):
                logits = torch.zeros(1, input_ids.shape[1], 11)
                for position in range(input_ids.shape[1]):
                    logits[0, position, (position + 1) % 11] = 2.0
                return SimpleNamespace(logits=logits)

        result = hf_sequence_logprobs(PositionModel(), rows(), device="cpu")
        normalizer = torch.logsumexp(torch.tensor([2.0] + [0.0] * 10), dim=0).item()
        # Prompt length2: token3 is predicted by position1 (highest token2),
        # hence all selected targets3,4,0 have zero logit here.
        for value in result[0]:
            self.assertAlmostEqual(value, -normalizer, places=6)

    def test_version_change_and_frozen_threshold_are_not_silent(self):
        model = TinyModel()
        with canonical_merged_model(model, schema(), device="cpu") as merged:
            reference = hf_sequence_logprobs(merged, rows(), device="cpu")
        client = client_for(reference, rows(), offset=0.01)
        versions = iter(["0", "1"])
        before = source_guard(model)
        with self.assertRaisesRegex(ValueError, "Policy changed"):
            score_three_way(model, schema(), rows(), client, device="cpu", version=0,
                            version_getter=lambda: next(versions))
        self.assertEqual(before, source_guard(model))
        result = score_three_way(model, schema(), rows(), client, device="cpu", acceptance={
            "policy_id": "cpu-test-limit", "engine_max_abs_logprob_delta": 0.001,
            "engine_mean_abs_logprob_delta": 0.001})
        self.assertFalse(result["numerical_accepted"])

    def test_fixture_rejects_unscored_claimed_eos(self):
        malformed = rows()
        malformed[0]["completion_ids"][-1] = 5
        with self.assertRaisesRegex(ValueError, "claimed EOS"):
            validate_fixed_rows(malformed)


if __name__ == "__main__":
    unittest.main()
