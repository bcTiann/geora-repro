"""Validate original-base GeoRA cancellation, gradients, and checkpoint mode.

Use independent dense FP64 effective weights as the algebra/gradient oracle,
then run the existing update/reload gates on a tiny Qwen in FP32 and BF16.
No model download is required; generated checkpoints live in a temporary dir.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))

from geora_layers import GeoRALinear, export_adapter_state, install_geora, load_geora_state
from geora_check_utils import CheckReport, inference_logits, move_with_precision
from check_geora_training import validate_model_and_update


def dense_fp64_oracle(layers, original_weights, original_biases, x):
    """Compute one or two updated effective matrices without split linears."""
    output = x
    for layer, weight, bias in zip(layers, original_weights, original_biases):
        effective = weight + layer.scaling * (
            layer.B.double() @ layer.A.double()
            - layer.B0.double() @ layer.A0.double()
        )
        output = nn.functional.linear(output, effective, bias)
    return output


def check_algebra_and_gradients():
    torch.manual_seed(41)
    layers = []
    original_weights = []
    original_biases = []
    for input_size, output_size in ((5, 7), (7, 4)):
        base = nn.Linear(input_size, output_size).float()
        original_weights.append(base.weight.detach().double().clone())
        original_biases.append(base.bias.detach().double().clone())
        A0 = torch.randn(2, input_size) * 0.1
        B0 = torch.randn(output_size, 2) * 0.1
        layer = GeoRALinear(base, A0, B0, scaling=2, forward_mode="difference")
        with torch.no_grad():
            layer.A.add_(torch.randn_like(layer.A) * 0.03)
            layer.B.add_(torch.randn_like(layer.B) * 0.02)
        layers.append(layer)
    x = torch.randn(3, 5, requires_grad=True)
    x_oracle = x.detach().double().requires_grad_(True)
    output = layers[1](layers[0](x))
    expected = dense_fp64_oracle(layers, original_weights, original_biases, x_oracle)
    parameters = [parameter for layer in layers for parameter in (layer.A, layer.B)]
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad((output * upstream).sum(), [x] + parameters)
    expected_gradients = torch.autograd.grad(
        (expected * upstream.double()).sum(), [x_oracle] + parameters
    )
    torch.testing.assert_close(output.double(), expected, rtol=3e-6, atol=5e-7)
    errors = []
    for measured, oracle in zip(gradients, expected_gradients):
        torch.testing.assert_close(measured.double(), oracle.double(), rtol=5e-6, atol=5e-7)
        errors.append((measured.double() - oracle.double()).abs().max().item())
    assert all(layer.A0.grad is None and layer.B0.grad is None for layer in layers)
    print("PASS: updated two-layer values and input/A/B gradients match independent dense FP64 oracle", flush=True)
    return {"output_max_absolute_error": (output.double()-expected).abs().max().item(),
            "gradient_max_absolute_errors": errors, "layer_count": 2}


def run(output_path):
    torch.set_num_threads(2)
    summary = {"status": "running", "torch_version": torch.__version__, "device": "cpu",
               "algebra": check_algebra_and_gradients(), "tiny_models": {}}
    with tempfile.TemporaryDirectory(prefix="geora-difference-") as temporary:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(7)
            configuration = Qwen2Config(
                vocab_size=64, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=64, attention_dropout=0.0,
            )
            configuration._attn_implementation = "eager"
            original = Qwen2ForCausalLM(configuration).float().eval()
            original.requires_grad_(False)

            def fresh():
                return copy.deepcopy(original)

            initialized = fresh()
            legacy_manifest = install_geora(initialized, rank=2, alpha=4, rho=0.2)
            # Remote saved initialization predates forward-mode tags. Reproduce
            # that old format and select the candidate explicitly exactly once.
            legacy_manifest.pop("forward_mode", None)
            state = export_adapter_state(initialized)
            model = fresh()
            load_geora_state(model, state, legacy_manifest, forward_mode="difference")
            manifest = dict(legacy_manifest, forward_mode="difference")
            reference = fresh()
            move_with_precision(reference, "cpu", dtype)
            move_with_precision(model, "cpu", dtype)
            ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            labels = ids.clone()
            labels[:, :3] = -100
            actual = inference_logits(model, inputs, "cpu", dtype)
            expected = inference_logits(reference, inputs, "cpu", dtype)
            assert torch.equal(actual, expected), "Initial logits must exactly match the original path"
            directory = Path(temporary) / str(dtype)
            report = CheckReport(directory, "checks.json")
            report.require("difference_initial_logits_exact", torch.equal(actual, expected))
            validate_model_and_update(
                model, reference, manifest, fresh, inputs, labels, directory,
                report, "cpu", dtype,
            )
            report.finish()
            assert json.loads((directory/"manifest.json").read_text())["forward_mode"] == "difference"
            summary["tiny_models"][str(dtype)] = report.data
            print("PASSED TINY DIFFERENCE MODEL:", dtype, len(report.data["checks"]), flush=True)

            wrong_mode = fresh()
            try:
                load_geora_state(wrong_mode, state, manifest, forward_mode="residual")
            except ValueError as error:
                assert "conflicts" in str(error)
                assert not any(isinstance(module, GeoRALinear) for module in wrong_mode.modules())
            else:
                raise AssertionError("Tagged mode conflict was not rejected")
            default_legacy = fresh()
            load_geora_state(default_legacy, state, legacy_manifest)
            assert all(module.forward_mode == "residual" for module in default_legacy.modules()
                       if isinstance(module, GeoRALinear))
            print("PASS: tagged conflicts rejected before mutation; old untagged checkpoints retain residual default", flush=True)
    summary.update(status="passed", scope="Tiny CPU models only; each dtype performed one mechanical AdamW step.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2)+"\n")
        print("Report:", output_path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
