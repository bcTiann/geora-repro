"""Probe layer-0 attention precision using saved factors, without a full model or SVD."""

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
import transformers
import transformers.models.qwen2.modeling_qwen2 as qwen

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))
from geora_layers import GeoRALinear

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def relative_error(reference, actual):
    reference = reference.float()
    actual = actual.float()
    return (torch.linalg.vector_norm(actual - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()


def attention_trace(attention, hidden_states, position_embeddings, mode):
    """Each mode changes one named numerical part while retaining identical input."""
    full_fp32 = mode == "fp32_all"
    fp32_projections = mode in ("fp32_projections", "fp32_all")
    fp32_scores = mode in ("fp32_scores", "fp32_core", "fp32_all")
    fp32_value_sum = mode in ("fp32_core", "fp32_all")
    device_type = hidden_states.device.type

    def projection(layer, value):
        if fp32_projections:
            with torch.autocast(device_type=device_type, enabled=False):
                output = layer(value.float())
            return output if full_fp32 else output.bfloat16()
        return layer(value)

    with torch.inference_mode(), torch.autocast(
        device_type=device_type, dtype=torch.bfloat16, enabled=not full_fp32
    ):
        batch, length, _ = hidden_states.shape
        head_shape = (batch, length, -1, attention.head_dim)
        query = projection(attention.q_proj, hidden_states).view(head_shape).transpose(1, 2)
        key = projection(attention.k_proj, hidden_states).view(head_shape).transpose(1, 2)
        value = projection(attention.v_proj, hidden_states).view(head_shape).transpose(1, 2)
        query, key = qwen.apply_rotary_pos_emb(query, key, *position_embeddings)
        key = qwen.repeat_kv(key, attention.num_key_value_groups)
        value = qwen.repeat_kv(value, attention.num_key_value_groups)
        valid = torch.ones(length, length, device=hidden_states.device, dtype=torch.bool).tril()

        if fp32_scores:
            with torch.autocast(device_type=device_type, enabled=False):
                scores = (query.float() @ key.float().transpose(-2, -1)) * attention.scaling
        else:
            scores = (query @ key.transpose(-2, -1)) * attention.scaling
        masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probabilities = masked_scores.softmax(dim=-1, dtype=torch.float32)
        if fp32_value_sum:
            with torch.autocast(device_type=device_type, enabled=False):
                context = probabilities.float() @ value.float()
            if not full_fp32:
                context = context.bfloat16()
        else:
            probabilities = probabilities.to(query.dtype)
            context = probabilities @ value
        context = context.transpose(1, 2).contiguous().reshape(batch, length, -1)
        output = projection(attention.o_proj, context)
        # Recalculate the exact same quantized Q/K in FP32 to isolate score rounding.
        with torch.autocast(device_type=device_type, enabled=False):
            fp32_scores_from_same_qk = (query.float() @ key.float().transpose(-2, -1)) * attention.scaling
        return {
            "query": query.float(), "key": key.float(), "value": value.float(),
            "scores": scores.float(), "probabilities": probabilities.float(),
            "context": context.float(), "output": output.float(), "valid": valid,
            "score_rounding_max_absolute_error": (
                scores.float() - fp32_scores_from_same_qk
            ).abs()[..., valid].max().item(),
        }


def compare_traces(reference, actual):
    tv = (reference["probabilities"] - actual["probabilities"]).abs().sum(dim=-1) / 2
    valid = reference["valid"]
    result = {f"{name}_relative_error": relative_error(reference[name], actual[name])
              for name in ("query", "key", "value", "context", "output")}
    flat_index = tv.reshape(-1).argmax().item()
    batch, head, position = (flat_index // (tv.shape[1] * tv.shape[2]),
                             (flat_index // tv.shape[2]) % tv.shape[1],
                             flat_index % tv.shape[2])
    visible = position + 1
    result.update(
        reference_valid_scores_max_absolute_value=reference["scores"][..., valid].abs().max().item(),
        worst_attention_row={
            "batch": batch, "head": head, "query_position": position,
            "reference_scores": reference["scores"][batch, head, position, :visible].cpu().tolist(),
            "actual_scores": actual["scores"][batch, head, position, :visible].cpu().tolist(),
            "reference_probabilities": reference["probabilities"][batch, head, position, :visible].cpu().tolist(),
            "actual_probabilities": actual["probabilities"][batch, head, position, :visible].cpu().tolist(),
        },
        valid_scores_relative_error=relative_error(reference["scores"][..., valid], actual["scores"][..., valid]),
        valid_scores_max_absolute_error=(reference["scores"] - actual["scores"]).abs()[..., valid].max().item(),
        attention_probability_mean_total_variation=tv.mean().item(),
        attention_probability_max_total_variation=tv.max().item(),
        reference_score_rounding_max_absolute_error=reference["score_rounding_max_absolute_error"],
        actual_score_rounding_max_absolute_error=actual["score_rounding_max_absolute_error"],
    )
    return result


def run(arguments, report):
    started = time.perf_counter()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if arguments.device == "cuda" and (torch.version.hip is None or not torch.cuda.is_available()):
        raise RuntimeError("A visible ROCm GPU is required for --device cuda.")
    configuration = json.loads((PROJECT_DIRECTORY / "configs/base_model.json").read_text())
    checkpoint = PROJECT_DIRECTORY / configuration["checkpoint_directory"]
    revision_record = checkpoint / ".cache/huggingface/download/model.safetensors.metadata"
    if revision_record.read_text().splitlines()[0] != configuration["model_revision"]:
        raise RuntimeError("Checkpoint revision differs from the pinned model.")
    manifest = json.loads((arguments.init_dir / "manifest.json").read_text())
    for name in ("model_repository", "model_revision", "rank", "alpha", "rho"):
        expected = configuration.get(name, configuration["initialization"].get(name))
        if manifest[name] != expected:
            raise RuntimeError(f"Initialization {name} differs from the configuration.")
    model_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    model_config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 3? Answer briefly."}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids += tokenizer.encode("5", add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([ids])

    # Read only the four projections, one norm, and selected embedding rows.
    # safe_open maps the checkpoint; it does not instantiate the 1.5B model.
    reference = qwen.Qwen2Attention(model_config, layer_idx=0).float().eval()
    norm = qwen.Qwen2RMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps)
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as weights:
        embeddings = weights.get_tensor("model.embed_tokens.weight")[input_ids].clone()
        with torch.no_grad():
            norm.weight.copy_(weights.get_tensor("model.layers.0.input_layernorm.weight"))
            for projection_name in PROJECTIONS:
                layer = getattr(reference, projection_name)
                prefix = f"model.layers.0.self_attn.{projection_name}"
                layer.weight.copy_(weights.get_tensor(prefix + ".weight"))
                if layer.bias is not None:
                    layer.bias.copy_(weights.get_tensor(prefix + ".bias"))
    geo = copy.deepcopy(reference)
    with safe_open(str(arguments.init_dir / "adapter.safetensors"), framework="pt", device="cpu") as factors:
        for projection_name in PROJECTIONS:
            prefix = f"model.layers.0.self_attn.{projection_name}"
            A0, B0 = (factors.get_tensor(prefix + suffix) for suffix in (".A0", ".B0"))
            if not torch.equal(A0, factors.get_tensor(prefix + ".A")) or not torch.equal(B0, factors.get_tensor(prefix + ".B")):
                raise RuntimeError("The probe requires untrained initial factors.")
            setattr(geo, projection_name, GeoRALinear(
                getattr(geo, projection_name), A0, B0, manifest["alpha"] / manifest["rank"]
            ))
    reference.requires_grad_(False)
    geo.requires_grad_(False)
    reference.to(arguments.device)
    geo.to(arguments.device)
    norm.requires_grad_(False)
    norm32 = copy.deepcopy(norm).float().to(arguments.device)
    norm16 = norm.bfloat16().to(arguments.device)
    rotary = qwen.Qwen2RotaryEmbedding(model_config).to(arguments.device)
    embeddings = embeddings.to(arguments.device)
    positions = torch.arange(len(ids), device=arguments.device).unsqueeze(0)
    with torch.inference_mode():
        hidden16 = norm16(embeddings.bfloat16())
        hidden32 = norm32(embeddings.float())
        rope16 = rotary(embeddings.bfloat16(), positions)
        rope32 = rotary(embeddings.float(), positions)

    report.update(
        stage="first_attention_precision_probe", optimizer_steps=0, svd_calls=0,
        token_count=len(ids), torch_version=torch.__version__, hip_version=torch.version.hip,
        transformers_version=transformers.__version__, device=arguments.device,
        gpu_name=torch.cuda.get_device_name(0) if arguments.device == "cuda" else None,
        model_repository=configuration["model_repository"], model_revision=configuration["model_revision"],
        cases={}, reference_controls={},
        eager_attention_source=inspect.getsource(qwen.eager_attention_forward),
    )
    references = {}
    compute_started = time.perf_counter()
    for mode in ("native", "fp32_scores", "fp32_core", "fp32_projections", "fp32_all"):
        hidden, rope = (hidden32, rope32) if mode == "fp32_all" else (hidden16, rope16)
        reference_trace = attention_trace(reference, hidden, rope, mode)
        actual_trace = attention_trace(geo, hidden, rope, mode)
        row = compare_traces(reference_trace, actual_trace)
        report["cases"][mode] = row
        references[mode] = reference_trace
        print("PROBE", mode + ":", json.dumps({key: value for key, value in row.items()
                                                if key != "worst_attention_row"}), flush=True)
        if mode == "native":
            # Check the manual trace against the installed Transformers implementation.
            mask = torch.zeros(len(ids), len(ids), device=arguments.device, dtype=torch.bfloat16)
            mask.masked_fill_(~reference_trace["valid"], torch.finfo(torch.bfloat16).min)
            with torch.inference_mode(), torch.autocast(device_type=arguments.device, dtype=torch.bfloat16):
                for name, attention, trace in (("reference", reference, reference_trace), ("geora", geo, actual_trace)):
                    library_output, _ = attention(hidden, rope, mask[None, None])
                    error = relative_error(library_output, trace["output"])
                    report[name + "_manual_vs_library_relative_error"] = error
                    if error > 1e-6:
                        raise RuntimeError(f"Manual attention differs from installed implementation: {name}, {error}")
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    for mode in references:
        if mode == "native":
            continue
        report["reference_controls"][mode] = compare_traces(references[mode], references["native"])
        print("REFERENCE CONTROL", mode + ":", json.dumps({key: value for key, value in report["reference_controls"][mode].items()
                                                            if key != "worst_attention_row"}), flush=True)
    if arguments.device == "cuda":
        torch.cuda.synchronize()
        report["peak_allocated_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report.update(status="completed", compute_seconds=time.perf_counter() - compute_started,
                  script_seconds=time.perf_counter() - started)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PROBE COMPLETED:", json.dumps({key: report[key] for key in ("compute_seconds", "script_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running"}
    try:
        run(arguments, report)
    except Exception as error:
        report.update(status="failed", error=str(error))
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
