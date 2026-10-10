"""Check the LoRA baseline before spending GPU time on comparison training."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

PROJECT_DIRECTORY = next(
    directory for directory in (Path(__file__).resolve().parents[1], Path.cwd())
    if (directory / "geora_layers.py").exists()
)
sys.path.insert(0, str(PROJECT_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from geora_check_utils import forward_context, move_with_precision
from lora_layers import LoRALinear, export_lora_state, install_lora, iter_lora_layers, load_lora_state


def run_checks(report_path: Path) -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(12)
    report = {"stage": "lora_cpu_correctness", "status": "running", "device": "cpu", "torch_version": torch.__version__, "checks": []}

    def require(name, condition, **details):
        report["checks"].append({"name": name, "passed": bool(condition), **details})
        print(f"{'PASS' if condition else 'FAIL'}: {name}", flush=True)
        if not condition:
            report["status"] = "failed"
            report["error"] = name
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if not condition:
            raise AssertionError(name)

    # A small 28-block shape fixture tests the actual 196-name target selector.
    fixture = nn.Module()
    fixture.model = nn.Module()
    fixture.model.layers = nn.ModuleList()
    for _ in range(28):
        block = nn.Module()
        block.self_attn = nn.Module()
        block.mlp = nn.Module()
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, projection, nn.Linear(8, 8))
        for projection in ("gate_proj", "up_proj", "down_proj"):
            setattr(block.mlp, projection, nn.Linear(8, 8))
        fixture.model.layers.append(block)
    fixture.lm_head = nn.Linear(8, 64)
    fixture.extra_proj = nn.Linear(8, 8)
    original_fixture = copy.deepcopy(fixture)
    global_rng = torch.get_rng_state().clone()
    fixture_manifest = install_lora(fixture, rank=2, alpha=4, seed=123)
    require("196_targets_392_AB_tensors_lm_head_excluded",
            len(list(iter_lora_layers(fixture))) == 196
            and len([p for p in fixture.parameters() if p.requires_grad]) == 392
            and isinstance(fixture.lm_head, nn.Linear) and not fixture.lm_head.weight.requires_grad
            and isinstance(fixture.extra_proj, nn.Linear) and not fixture.extra_proj.weight.requires_grad)
    require("private_initialization_generator_preserves_global_rng", torch.equal(torch.get_rng_state(), global_rng))
    repeat = copy.deepcopy(original_fixture)
    repeat_manifest = install_lora(repeat, rank=2, alpha=4, seed=123)
    require("same_seed_factors_and_manifest_reproducible", fixture_manifest == repeat_manifest
            and all(torch.equal(value, export_lora_state(repeat)[name]) for name, value in export_lora_state(fixture).items()))
    require("Gaussian_A_zero_B_no_initial_buffers", all(
        layer.A.dtype == torch.float32 and layer.B.dtype == torch.float32
        and torch.count_nonzero(layer.A) > 0 and torch.count_nonzero(layer.B) == 0
        and not hasattr(layer, "A0") and not hasattr(layer, "B0") for _, layer in iter_lora_layers(fixture)))
    require("all_original_base_values_preserved", all(
        torch.equal(layer.base_layer.weight, original_fixture.get_submodule(name).weight)
        and torch.equal(layer.base_layer.bias, original_fixture.get_submodule(name).bias)
        for name, layer in iter_lora_layers(fixture)))

    # A projection gives the B-first/A-second gradient mechanism a direct check.
    for frozen_dtype in (torch.float32, torch.bfloat16):
        label = str(frozen_dtype)
        base = nn.Linear(4, 3).float()
        reference = copy.deepcopy(base)
        A = torch.tensor([[.03, -.01, .02, .04], [-.02, .05, .01, -.03]])
        layer = LoRALinear(base, A, torch.zeros(3, 2), scaling=2)
        move_with_precision(layer, "cpu", frozen_dtype)
        reference.to(dtype=frozen_dtype).requires_grad_(False)
        x = torch.tensor([[.5, -.25, .75, .2], [-.4, .8, -.2, .6]], requires_grad=True)
        native_x = x.detach().clone().requires_grad_(True)
        with forward_context("cpu", frozen_dtype):
            actual, expected = layer(x), reference(native_x)
        require(label + "_initial_linear_output_exact", torch.equal(actual, expected))
        actual.float().sum().backward()
        expected.float().sum().backward()
        require(label + "_initial_input_gradient_exact_native", torch.equal(x.grad, native_x.grad))
        optimizer = torch.optim.SGD((layer.A, layer.B), lr=0.5)
        initial_A = layer.A.detach().clone()
        initial_B = layer.B.detach().clone()
        frozen_weight = layer.base_layer.weight.detach().clone()
        target = torch.tensor([[.2, -.4, .3], [-.3, .1, -.2]])
        optimizer.zero_grad(set_to_none=True)
        with forward_context("cpu", frozen_dtype):
            loss = (layer(x).float() - target).square().sum()
        loss.backward()
        require(label + "_initial_A_gradient_zero_B_gradient_nonzero",
                torch.count_nonzero(layer.A.grad) == 0 and torch.count_nonzero(layer.B.grad) > 0
                and layer.A.grad.dtype == torch.float32 and layer.B.grad.dtype == torch.float32)
        optimizer.step()
        require(label + "_first_step_only_B_changes", torch.equal(layer.A, initial_A) and not torch.equal(layer.B, initial_B))
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        native_x.grad = None
        with forward_context("cpu", frozen_dtype):
            loss = (layer(x).float() - target).square().sum()
        loss.backward()
        require(label + "_second_step_A_gradient_nonzero_and_input_gradient_finite",
                torch.count_nonzero(layer.A.grad) > 0 and torch.isfinite(x.grad).all())
        optimizer.step()
        require(label + "_second_step_A_changes_base_unchanged", not torch.equal(layer.A, initial_A)
                and torch.equal(layer.base_layer.weight, frozen_weight) and layer.base_layer.weight.grad is None)
        input_lora = x.detach().clone().requires_grad_(True)
        input_native = x.detach().clone().requires_grad_(True)
        with forward_context("cpu", frozen_dtype):
            lora_input_loss = layer(input_lora).float().sum()
            native_input_loss = reference(input_native).float().sum()
        lora_input_loss.backward()
        native_input_loss.backward()
        expected_branch_gradient = 2 * (torch.ones(2, 3) @ layer.B.detach() @ layer.A.detach())
        require(label + "_input_gradient_includes_trained_adapter", torch.allclose(
            input_lora.grad, input_native.grad + expected_branch_gradient, rtol=1e-5, atol=1e-6))

    # Use tiny real Qwen blocks to check complete forward/backward and reload.
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = "eager"
    torch.manual_seed(12)
    original = Qwen2ForCausalLM(config).float().eval()
    provenance = {"model_repository": "local_random_tiny_Qwen", "seed": 12}
    input_ids = torch.tensor([[1, 7, 9, 11, 3, 5], [2, 8, 10, 12, 4, 6]])
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "use_cache": False}
    for dtype in (torch.float32, torch.bfloat16):
        label = str(dtype)
        reference = copy.deepcopy(original).requires_grad_(False)
        policy = copy.deepcopy(original)
        manifest = install_lora(policy, rank=2, alpha=4, seed=456, provenance=provenance)
        move_with_precision(reference, "cpu", dtype)
        move_with_precision(policy, "cpu", dtype)
        policy.train()
        with torch.no_grad(), forward_context("cpu", dtype):
            reference_logits = reference(**inputs).logits
            initial_logits = policy(**inputs).logits
        require(label + "_tiny_Qwen_initial_logits_exact", torch.equal(initial_logits, reference_logits))
        adapter_state = export_lora_state(policy)
        expected_trainable = set(adapter_state)
        actual_trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
        require(label + "_tiny_Qwen_only_AB_trainable_FP32", actual_trainable == expected_trainable
                and len(actual_trainable) == 28
                and all(p.dtype == torch.float32 for p in policy.parameters() if p.requires_grad))
        frozen = {name: p.detach().clone() for name, p in policy.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
        A_names = {name for name in adapter_state if name.endswith(".A")}
        B_names = {name for name in adapter_state if name.endswith(".B")}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with forward_context("cpu", dtype):
                loss = policy(**inputs, labels=input_ids).loss
            loss.backward()
            parameters = dict(policy.named_parameters())
            require(label + f"_tiny_Qwen_step_{step + 1}_gradients_finite", torch.isfinite(loss)
                    and all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
                            for p in policy.parameters() if p.requires_grad))
            if step == 0:
                require(label + "_tiny_Qwen_first_B_gradient_then_A_gradient",
                        all(torch.count_nonzero(parameters[name].grad) == 0 for name in A_names)
                        and any(torch.count_nonzero(parameters[name].grad) > 0 for name in B_names))
            else:
                require(label + "_tiny_Qwen_second_A_gradient_nonzero",
                        any(torch.count_nonzero(parameters[name].grad) > 0 for name in A_names))
            optimizer.step()
        updated_state = export_lora_state(policy)
        require(label + "_tiny_Qwen_AB_changed_frozen_unchanged",
                any(not torch.equal(updated_state[name], adapter_state[name]) for name in A_names)
                and any(not torch.equal(updated_state[name], adapter_state[name]) for name in B_names)
                and all(torch.equal(dict(policy.named_parameters())[name], value) for name, value in frozen.items()))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_file(updated_state, str(directory / "adapter.safetensors"))
            (directory / "manifest.json").write_text(json.dumps(manifest))
            restored = copy.deepcopy(original)
            load_lora_state(restored, load_file(str(directory / "adapter.safetensors")),
                            json.loads((directory / "manifest.json").read_text()), expected_provenance=provenance)
            move_with_precision(restored, "cpu", dtype)
            restored.train()
            with torch.no_grad(), forward_context("cpu", dtype):
                actual_logits = policy(**inputs).logits
                restored_logits = restored(**inputs).logits
            require(label + "_tiny_Qwen_FP32_safetensors_reload_exact",
                    all(value.dtype == torch.float32 for value in load_file(str(directory / "adapter.safetensors")).values())
                    and torch.equal(actual_logits, restored_logits)
                    and all(torch.equal(value, export_lora_state(restored)[name]) for name, value in updated_state.items()))
        for invalid_kind in ("dtype", "provenance", "unexpected_key"):
            bad_state = {name: value.clone() for name, value in updated_state.items()}
            fresh = copy.deepcopy(original)
            expected = provenance
            if invalid_kind == "dtype":
                first_key = next(iter(bad_state))
                bad_state[first_key] = bad_state[first_key].bfloat16()
            elif invalid_kind == "provenance":
                expected = {"model_repository": "another_base"}
            else:
                bad_state["unexpected.A"] = torch.zeros(2, 32)
            try:
                load_lora_state(fresh, bad_state, manifest, expected_provenance=expected)
            except ValueError:
                require(label + "_bad_" + invalid_kind + "_rejected_before_modification",
                        not any(isinstance(layer, LoRALinear) for layer in fresh.modules()))
            else:
                raise AssertionError(f"Invalid {invalid_kind} checkpoint accepted")
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "lora_cpu_checks.json"
    result = run_checks(report_path)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print("LoRA CPU checks passed:", len(result["checks"]), "Report:", report_path, flush=True)
