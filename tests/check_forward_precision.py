"""Check the diagnostic projection policy on a tiny Qwen; no downloads/GPU."""

import argparse
import copy
from pathlib import Path
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-directory", type=Path, default=Path(__file__).resolve().parents[1])
arguments = parser.parse_args()
sys.path.insert(0, str(arguments.project_directory))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch
from torch.nn import functional
from transformers import Qwen2Config, Qwen2ForCausalLM

from geora_layers import (
    GeoRALinear, export_adapter_state, install_geora, load_geora_state,
)
from forward_precision import (
    fp32_attention_scores, fp32_projection_forwards,
    prepare_fp32_projections, projection_modules,
)
import forward_precision
import transformers.models.qwen2.modeling_qwen2 as qwen

torch.set_num_threads(2)
torch.manual_seed(0)
configuration = Qwen2Config(
    vocab_size=64, hidden_size=32, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=64, attention_dropout=0.0,
)
configuration._attn_implementation = "eager"
original = Qwen2ForCausalLM(configuration).float().eval()
original.requires_grad_(False)

# Save initial factors once, then rebuild every candidate from the original.
initialized = copy.deepcopy(original)
manifest = install_geora(initialized, rank=2, alpha=4, rho=0.2)
state = export_adapter_state(initialized)
reference = copy.deepcopy(original)
geo = copy.deepcopy(original)
load_geora_state(geo, state, manifest)
# Preserve the A/B flags: other runtime policies identify them by trainability.
# inference_mode(), rather than changing these flags, prevents diagnostic grads.
ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

for model_name, model in (("reference", reference), ("geora", geo)):
    target_names = prepare_fp32_projections(model, "cpu")
    assert len(target_names) == 14
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.model.layers[0].input_layernorm.weight.dtype == torch.bfloat16
    if model_name == "geora":
        for _, module in projection_modules(model):
            assert module.A0.dtype == module.B0.dtype == torch.float32

    targets = projection_modules(model)
    original_forwards = {name: module.forward for name, module in targets}
    head_forward = model.lm_head.forward
    parameter_snapshots = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    target_weights = {
        id(parameter): f"{name}.{parameter_name}"
        for name, module in targets
        for parameter_name, parameter in module.named_parameters()
        if parameter.ndim == 2
    }
    observed_linear_weights = set()
    observed_outputs = []
    original_linear = functional.linear

    def observe_linear(value, weight, bias=None):
        output = original_linear(value, weight, bias)
        if id(weight) in target_weights:
            assert value.dtype == weight.dtype == output.dtype == torch.float32
            if bias is not None:
                assert bias.dtype == torch.float32
            assert not torch.is_autocast_enabled("cpu")
            observed_linear_weights.add(id(weight))
        return output

    def observe_output(module, args, output):
        assert args[0].dtype == output.dtype == torch.bfloat16
        observed_outputs.append(module)

    handles = [module.register_forward_hook(observe_output) for _, module in targets]
    try:
        with fp32_projection_forwards(model) as patched_names:
            assert patched_names == target_names
            assert model.lm_head.forward == head_forward
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                with patch("torch.nn.functional.linear", side_effect=observe_linear):
                    logits = model(**inputs, use_cache=False).logits
            assert logits.dtype == torch.bfloat16
            assert not logits.requires_grad and torch.isfinite(logits).all()
            assert observed_linear_weights == set(target_weights)
            assert len(observed_outputs) == 14
    finally:
        for handle in handles:
            handle.remove()
    for name, module in targets:
        assert module.forward == original_forwards[name]
        assert "forward" not in module.__dict__
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter, parameter_snapshots[name])
    print(f"PASS: {model_name}, all 14 targets compute FP32 and emit BF16")

# Failure inside the probe must also restore original methods.
try:
    with fp32_projection_forwards(geo):
        raise RuntimeError("intentional probe failure")
except RuntimeError as error:
    assert str(error) == "intentional probe failure"
for _, module in projection_modules(geo):
    assert "forward" not in module.__dict__
print("PASS: exception restores projection forwards")

# Never pretend that an upcast can restore an already rounded residual.
rounded = copy.deepcopy(original)
rounded.bfloat16()
try:
    prepare_fp32_projections(rounded, "cpu")
except ValueError as error:
    assert "must already be FP32" in str(error)
else:
    raise AssertionError("Already rounded targets were accepted")
print("PASS: previously rounded projections rejected")

# Reach the replacement through actual Qwen forwards, including repeated KV
# heads, mask addition and standard output shape. Observe the underlying ops.
original_eager = qwen.eager_attention_forward
original_score_forward = forward_precision._eager_attention_fp32_scores
original_matmul = torch.matmul
original_softmax = functional.softmax
score_calls = 0
active_attention = False
attention_matmuls = []
softmax_input_dtypes = []

def observe_matmul(left, right, *args, **kwargs):
    output = original_matmul(left, right, *args, **kwargs)
    if active_attention:
        attention_matmuls.append((
            left.dtype, right.dtype, output.dtype,
            torch.is_autocast_enabled("cpu"),
        ))
    return output

def observe_softmax(value, *args, **kwargs):
    if active_attention:
        softmax_input_dtypes.append(value.dtype)
    return original_softmax(value, *args, **kwargs)

def observe_score_forward(module, query, key, value, *args, **kwargs):
    global score_calls, active_attention
    score_calls += 1
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    active_attention = True
    try:
        output, probabilities = original_score_forward(
            module, query, key, value, *args, **kwargs
        )
    finally:
        active_attention = False
    assert output.shape == (
        query.shape[0], query.shape[2], query.shape[1], query.shape[3]
    )
    assert probabilities.shape == (
        query.shape[0], query.shape[1], query.shape[2], key.shape[2]
    )
    assert output.dtype == probabilities.dtype == torch.bfloat16
    return output, probabilities

with patch("forward_precision._eager_attention_fp32_scores", side_effect=observe_score_forward):
    with fp32_attention_scores():
        assert qwen.eager_attention_forward is not original_eager
        for model in (reference, geo):
            with fp32_projection_forwards(model):
                with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                    with patch("torch.matmul", side_effect=observe_matmul):
                        with patch("torch.nn.functional.softmax", side_effect=observe_softmax):
                            logits = model(**inputs, use_cache=False).logits
                assert torch.isfinite(logits).all() and not logits.requires_grad
assert qwen.eager_attention_forward is original_eager
assert score_calls == 4  # Reference + GeoRA, two layers each.
assert len(attention_matmuls) == 8
for index in range(0, len(attention_matmuls), 2):
    assert attention_matmuls[index] == (
        torch.float32, torch.float32, torch.float32, False
    )
    assert attention_matmuls[index + 1] == (
        torch.bfloat16, torch.bfloat16, torch.bfloat16, True
    )
assert softmax_input_dtypes == [torch.float32] * 4
assert all(parameter.grad is None for model in (reference, geo) for parameter in model.parameters())
print("PASS: matched Qwen eager forwards use FP32 QK/mask scores and native BF16 PV")

try:
    with fp32_attention_scores():
        raise RuntimeError("intentional attention probe failure")
except RuntimeError as error:
    assert str(error) == "intentional attention probe failure"
assert qwen.eager_attention_forward is original_eager
print("PASS: normal and exceptional exit restore global eager attention")

# Although this experiment uses inference_mode, the wrapper is differentiable.
# Exercise a single target rather than training or running an optimizer.
projection = geo.model.layers[0].self_attn.q_proj
projection.A.requires_grad_(True)
projection.B.requires_grad_(True)
with fp32_projection_forwards(geo), torch.autocast("cpu", dtype=torch.bfloat16):
    input_vector = torch.randn(1, 2, configuration.hidden_size).bfloat16()
    loss = projection(input_vector).float().square().mean()
    loss.backward()
assert projection.A.grad.dtype == projection.B.grad.dtype == torch.float32
assert torch.isfinite(projection.A.grad).all()
assert torch.isfinite(projection.B.grad).all()
assert projection.base_layer.weight.grad is None
print("PASS: wrapper retains FP32 A/B gradients; no optimizer step")
