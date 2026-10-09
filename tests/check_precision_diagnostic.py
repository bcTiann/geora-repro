"""Check diagnostic interpretation and causal answer alignment without downloads."""

import copy
import json
from pathlib import Path
import sys
import tempfile

project_directory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_directory))
sys.path.insert(0, str(project_directory / "scripts"))

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from diagnose_geora_precision import diagnose_initialization, position_metrics
from geora_check_utils import CheckReport, logits_difference, move_with_precision
from geora_layers import export_adapter_state, install_geora

torch.set_num_threads(2)

# Adding an arbitrary constant at each position changes raw logits, but preserves
# probabilities. The centered error and distribution metrics must reveal this.
reference_logits = torch.tensor([[[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]]])
shifted_logits = reference_logits + torch.tensor([[[4.0], [-8.0]]])
metrics = logits_difference(reference_logits, shifted_logits)
assert metrics["relative_l2_error"] > 1
assert metrics["centered_relative_l2_error"] < 1e-6
assert abs(metrics["all_positions_max_kl_nats"]) < 1e-12
assert metrics["all_positions_max_total_variation"] < 1e-12

# Matching the final position must not hide a different distribution earlier.
different_logits = reference_logits.clone()
different_logits[:, 0] = torch.tensor([9.0, -9.0, -9.0])
metrics = logits_difference(reference_logits, different_logits)
assert metrics["last_token_reference_to_actual_kl_nats"] == 0
assert metrics["all_positions_max_kl_nats"] > 1

# A label at position 1 is predicted by logits at position 0, not position 1.
rows = position_metrics(reference_logits, reference_logits, torch.tensor([[-100, 2]]))
assert rows[0]["predicts_supervised_answer"] and rows[0]["target_token"] == 2
assert not rows[1]["predicts_supervised_answer"]
print("Distribution invariance and causal answer alignment passed.")

torch.manual_seed(0)
configuration = Qwen2Config(
    vocab_size=64, hidden_size=32, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=64, attention_dropout=0.0,
)
configuration._attn_implementation = "eager"
# Match the real checkpoint's BF16 storage before creating the FP32 base factory.
original = Qwen2ForCausalLM(configuration).bfloat16().float().eval()
original.requires_grad_(False)


def fresh():
    return copy.deepcopy(original)


reference = fresh()
model = fresh()
manifest = install_geora(model, rank=2, alpha=4, rho=0.2)
before = export_adapter_state(model)
move_with_precision(reference, "cpu", torch.bfloat16)
move_with_precision(model, "cpu", torch.bfloat16)
ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
labels = ids.clone()
labels[:, :3] = -100

with tempfile.TemporaryDirectory(prefix="geora-diagnostic-") as directory:
    output = Path(directory)
    report = CheckReport(output, "gpu_checks.json")
    diagnose_initialization(model, reference, manifest, fresh, inputs, labels, output, report, "cpu")
    diagnostic = json.loads((output / "precision_diagnostics.json").read_text())
    assert diagnostic["status"] == "completed" and diagnostic["optimizer_steps"] == 0
    assert len(diagnostic["layers"]) == 14
    assert diagnostic["fp32_logits"]["relative_l2_error"] < 1e-4
    answer_positions = [row["logit_position"] for row in diagnostic["bf16_positions"]
                        if row["predicts_supervised_answer"]]
    assert answer_positions == [2, 3, 4]
    after = export_adapter_state(model)
    assert all(torch.equal(value, after[name]) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not any(module._forward_hooks for module in model.modules())
    assert not any(module._forward_hooks for module in reference.modules())

print("Small-model BF16/FP32 diagnostics passed; factors unchanged, no gradients or SVD reload.")
