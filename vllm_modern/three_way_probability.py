"""Fixed-token learner / HF merged / vLLM merged probability diagnostics.

The temporary HF model reuses the learner's frozen backbone and replaces only
adapter modules. Original module and Parameter objects are always restored;
neither PEFT's in-place merge nor a second full model allocation is used.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from native_sync import canonical_sources, dense_tensor, source_guard


def _root(model):
    return model.get_base_model() if hasattr(model, "peft_config") else model


@contextmanager
def evaluating(model):
    modes = [(layer, layer.training) for layer in model.modules()]
    model.eval()
    try:
        yield
    finally:
        # Direct assignment preserves deliberately heterogeneous submodule modes.
        for layer, mode in modes:
            layer.training = mode


@contextmanager
def canonical_merged_model(model, expected_schema, *, device):
    """Temporarily replace adapters with exact exported canonical dense layers.

    References owned by the optimizer keep pointing to the original Parameter
    objects. A failure during preparation, forward, or validation restores every
    already-replaced module before propagating the error.
    """
    before = source_guard(model)
    root = _root(model)
    sources = canonical_sources(model, expected_schema)
    replacements = []
    try:
        for name, source in sources.items():
            if source[0] not in ("geora", "lora"):
                continue
            if not name.endswith(".weight"):
                raise ValueError("An adapter export must have a canonical weight name")
            module_path = name.removesuffix(".weight")
            if not module_path:
                raise ValueError("Root-level adapter replacement is unsupported")
            parent_path, _, child_name = module_path.rpartition(".")
            parent = root.get_submodule(parent_path) if parent_path else root
            original = parent._modules[child_name]
            shape, dtype = expected_schema[name]
            if len(shape) != 2 or dtype != torch.bfloat16:
                raise ValueError("This diagnostic requires canonical BF16 Linear weights")
            weight = dense_tensor(source, shape=shape, dtype=dtype, device=device)
            bias_name = module_path + ".bias"
            bias = None
            if bias_name in expected_schema:
                bias_shape, bias_dtype = expected_schema[bias_name]
                bias = dense_tensor(sources[bias_name], shape=bias_shape, dtype=bias_dtype, device=device)
            # Meta construction avoids allocating a random second dense tensor.
            dense = nn.Linear(shape[1], shape[0], bias=bias is not None, device="meta", dtype=dtype)
            dense.weight = nn.Parameter(weight, requires_grad=False)
            if bias is not None:
                dense.bias = nn.Parameter(bias, requires_grad=False)
            dense.training = original.training
            replacements.append((parent, child_name, original))
            parent._modules[child_name] = dense

        state = root.state_dict()
        if set(state) != set(expected_schema):
            raise ValueError("The temporary HF model is not the complete canonical policy")
        for name, tensor in state.items():
            shape, dtype = expected_schema[name]
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                raise ValueError(f"Temporary canonical HF tensor differs from schema: {name}")
        yield root
    finally:
        for parent, child_name, original in reversed(replacements):
            parent._modules[child_name] = original
        if source_guard(model) != before:
            raise RuntimeError("Merged probability scoring mutated learner state or Parameter references")


def validate_fixed_rows(fixed_rows):
    if not fixed_rows:
        raise ValueError("Nonempty fixed-token rows are required")
    seen = set()
    for index, row in enumerate(fixed_rows):
        identifier = str(row.get("id", index))
        if identifier in seen:
            raise ValueError("Fixed-token row IDs must be unique")
        seen.add(identifier)
        for field in ("prompt_ids", "completion_ids"):
            values = row[field]
            if not values or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in values):
                raise ValueError("Token rows must contain nonempty nonnegative integer IDs")
        if row.get("terminated_with_eos"):
            eos = row.get("eos_token_id")
            if eos is None or row["completion_ids"][-1] != eos:
                raise ValueError("The claimed EOS must be present and scored as the last completion token")


def hf_sequence_logprobs(model, fixed_rows, *, device):
    """Teacher-force every requested completion token, including a final EOS."""
    validate_fixed_rows(fixed_rows)
    scored = []
    with torch.no_grad():
        for row in fixed_rows:
            prompt = row["prompt_ids"]
            completion = row["completion_ids"]
            ids = torch.tensor([prompt + completion], device=device, dtype=torch.long)
            logits = model(input_ids=ids, use_cache=False).logits
            if logits.ndim != 3 or logits.shape[:2] != ids.shape:
                raise ValueError("HF logits do not align with the requested sequence")
            if max(prompt + completion) >= logits.shape[-1]:
                raise ValueError("Fixed token ID is outside the model vocabulary")
            selected_logits = logits[0, len(prompt) - 1:-1].float()
            targets = ids[0, len(prompt):]
            values = selected_logits.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(selected_logits, dim=-1)
            if values.numel() != len(completion) or not torch.isfinite(values).all():
                raise ValueError("HF did not return finite logprobs for every completion token")
            scored.append(values.cpu().tolist())
            del ids, logits, selected_logits, targets, values
    return scored


def vllm_sequence_logprobs(client, fixed_rows):
    validate_fixed_rows(fixed_rows)
    response = client.get_sequence_logprobs(
        [row["prompt_ids"] + row["completion_ids"] for row in fixed_rows],
        [len(row["prompt_ids"]) for row in fixed_rows],
        top_logprobs=1,
        temperature=1.0,
    )
    actual = response["actual_logprobs"]
    token_rows = response["actual_token_ids"]
    if len(actual) != len(fixed_rows) or len(token_rows) != len(fixed_rows):
        raise ValueError("vLLM omitted or added fixed sequences")
    scored = []
    for row, probabilities, token_ids in zip(fixed_rows, actual, token_rows, strict=True):
        if token_ids != [[token] for token in row["completion_ids"]] or len(probabilities) != len(row["completion_ids"]):
            raise ValueError("vLLM omitted or changed fixed completion tokens")
        if any(len(values) != 1 or values[0] is None or not math.isfinite(values[0]) or values[0] > 1e-6
               for values in probabilities):
            raise ValueError("vLLM returned missing or invalid actual-token logprobs")
        scored.append([values[0] for values in probabilities])
    return scored


def _difference(left, right):
    differences = [a - b for a, b in zip(left, right, strict=True)]
    if not differences or any(not math.isfinite(value) for value in differences):
        raise ValueError("Probability comparison requires nonempty finite values")
    absolute = torch.tensor([abs(value) for value in differences], dtype=torch.float64)
    return {
        "tokens": len(differences),
        "mean_abs_logprob_delta": absolute.mean().item(),
        "max_abs_logprob_delta": absolute.max().item(),
        "p95_abs_logprob_delta": torch.quantile(absolute, 0.95).item(),
        "p99_abs_logprob_delta": torch.quantile(absolute, 0.99).item(),
        "mean_signed_logprob_delta": sum(differences) / len(differences),
        "fraction_abs_logratio_above_log2": sum(abs(value) > math.log(2) for value in differences) / len(differences),
    }


def score_three_way(model, expected_schema, fixed_rows, client, *, device,
                    audit_path=None, version=None, version_getter=None, acceptance=None):
    """Return raw per-token values and separately measured merge/engine deltas.

    acceptance is optional: {"engine_max_abs_logprob_delta": x,
    "engine_mean_abs_logprob_delta": y, "policy_id": "frozen-control-v1"}.
    No default equivalence threshold is invented. Without an explicit frozen
    policy the result reports measurements and leaves numerical acceptance null.
    """
    validate_fixed_rows(fixed_rows)
    if version_getter is not None and version_getter() != str(version):
        raise ValueError("Unexpected policy version before the three-way check")
    before = source_guard(model)
    with evaluating(model):
        learner = hf_sequence_logprobs(model, fixed_rows, device=device)
        with canonical_merged_model(model, expected_schema, device=device) as merged_model:
            hf_merged = hf_sequence_logprobs(merged_model, fixed_rows, device=device)
        vllm_merged = vllm_sequence_logprobs(client, fixed_rows)
    if source_guard(model) != before:
        raise RuntimeError("Three-way scoring changed the learner state")
    if version_getter is not None and version_getter() != str(version):
        raise ValueError("Policy changed during the three-way check")
    flattened = [[value for row in values for value in row] for values in (learner, hf_merged, vllm_merged)]
    comparisons = {
        "learner_minus_hf_merged": _difference(flattened[0], flattened[1]),
        "hf_merged_minus_vllm": _difference(flattened[1], flattened[2]),
        "learner_minus_vllm": _difference(flattened[0], flattened[2]),
    }
    numerical_accepted = None
    if acceptance is not None:
        required = ("policy_id", "engine_max_abs_logprob_delta", "engine_mean_abs_logprob_delta")
        if any(key not in acceptance for key in required):
            raise ValueError("A complete, identified numerical acceptance policy is required")
        maximum = acceptance["engine_max_abs_logprob_delta"]
        mean = acceptance["engine_mean_abs_logprob_delta"]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (maximum, mean)):
            raise ValueError("Numerical acceptance limits must be finite and nonnegative")
        engine = comparisons["hf_merged_minus_vllm"]
        numerical_accepted = engine["max_abs_logprob_delta"] <= maximum and engine["mean_abs_logprob_delta"] <= mean
    token_rows = []
    for row, native_values, hf_values, server_values in zip(fixed_rows, learner, hf_merged, vllm_merged, strict=True):
        token_rows.append({
            "id": row.get("id"), "category": row.get("category"),
            "prompt_ids": row["prompt_ids"], "completion_ids": row["completion_ids"],
            "includes_eos": bool(row.get("terminated_with_eos", False)),
            "learner_logprobs": native_values, "hf_merged_logprobs": hf_values, "vllm_logprobs": server_values,
            "low_probability_tokens_logp_below_minus10": sum(value < -10 for value in native_values),
            "comparisons": {
                "learner_minus_hf_merged": _difference(native_values, hf_values),
                "hf_merged_minus_vllm": _difference(hf_values, server_values),
                "learner_minus_vllm": _difference(native_values, server_values),
            },
        })
    payload = {
        "policy_version": version, "structural_checks_passed": True,
        "numerical_accepted": numerical_accepted, "acceptance_policy": acceptance,
        "interpretation": "Merge delta is learner minus HF merged; engine delta is HF merged minus vLLM. No exact dual-branch/merged equality is assumed.",
        "fixed_rows_sha256": hashlib.sha256(json.dumps(fixed_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sequences": len(fixed_rows), "tokens": len(flattened[0]),
        "comparisons": comparisons, "rows": token_rows,
    }
    if audit_path is not None:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload
