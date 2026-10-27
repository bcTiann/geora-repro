"""Independently predict vLLM 0.32 Qwen2 packed parameter byte hashes.

The upstream /weight_checker endpoint hashes actual resident parameters. This
module constructs the corresponding hashes from the canonical HF weight stream
without retaining a second full policy. No vLLM code is imported on the learner.
The layout contract is deliberately limited to unquantized TP=PP=DP=PCP=1 Qwen2.
Unknown shapes, vocabulary padding, missing tensors, or extra resident keys fail.
"""

from __future__ import annotations

import hashlib

import torch


CHECKSUM_PREFIX = "dp0:pp0:pcp0:tp0:"
LAYOUT_SOURCES = {
    "qwen2": "https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/qwen2.py",
    "checksum": "https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/model_loader/weight_checksum.py",
    "prefix": "https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_worker.py",
    "vocabulary": "https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/vocab_parallel_embedding.py",
}


def qwen2_layout(config):
    """Return canonical shapes and ordered row partitions per resident tensor."""
    if getattr(config, "model_type", None) != "qwen2":
        raise ValueError("Resident verification requires the inspected Qwen2 layout")
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("Resident verification requires unquantized weights")
    h, vocab, inter = config.hidden_size, config.vocab_size, config.intermediate_size
    if vocab % 64:
        raise ValueError("Vocabulary padding requires a separately verified layout")
    head = getattr(config, "head_dim", None) or h // config.num_attention_heads
    if h != config.num_attention_heads * head:
        raise ValueError("Unexpected Qwen2 head dimensions")
    q, kv = config.num_attention_heads * head, config.num_key_value_heads * head
    shapes = {"model.embed_tokens.weight": (vocab, h), "lm_head.weight": (vocab, h),
              "model.norm.weight": (h,)}
    groups = {"model.embed_tokens.weight": ["model.embed_tokens.weight"],
              "model.norm.weight": ["model.norm.weight"]}
    if not config.tie_word_embeddings:
        groups["lm_head.weight"] = ["lm_head.weight"]
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        for suffix in ("weight", "bias"):
            group = []
            for part, out in (("q_proj", q), ("k_proj", kv), ("v_proj", kv)):
                name = prefix + f"self_attn.{part}.{suffix}"
                shapes[name] = (out, h) if suffix == "weight" else (out,)
                group.append(name)
            groups[prefix + f"self_attn.qkv_proj.{suffix}"] = group
        for suffix, shape in (("self_attn.o_proj.weight", (h, q)),
                              ("mlp.down_proj.weight", (h, inter)),
                              ("input_layernorm.weight", (h,)),
                              ("post_attention_layernorm.weight", (h,))):
            name = prefix + suffix
            shapes[name] = shape
            groups[name] = [name]
        group = []
        for part in ("gate_proj", "up_proj"):
            name = prefix + f"mlp.{part}.weight"
            shapes[name] = (inter, h)
            group.append(name)
        groups[prefix + "mlp.gate_up_proj.weight"] = group
    return shapes, groups


class PackedPolicyHasher:
    """Hash canonical tensors in an explicit, bounded-memory packing order.

    Concatenating contiguous q/k/v rows or gate/up rows is byte concatenation.
    Each source tensor crosses CPU once for both its own hash and packed hash.
    The tied output head is checked even though named_parameters() deduplicates
    its resident storage. The default aligned Qwen vocabulary needs no padding.
    """

    def __init__(self, config, schema):
        self.shapes, self.groups = qwen2_layout(config)
        if set(schema) != set(self.shapes):
            raise ValueError(f"Canonical Qwen2 schema mismatch: {sorted(set(schema) ^ set(self.shapes))}")
        for name, (shape, dtype) in schema.items():
            if tuple(shape) != self.shapes[name] or dtype != torch.bfloat16:
                raise ValueError(f"Unsupported canonical shape/dtype: {name}")
        self.canonical_order = [name for parts in self.groups.values() for name in parts]
        self.tied = bool(config.tie_word_embeddings)
        if self.tied:
            self.canonical_order.append("lm_head.weight")
        self.group_for = {name: group for group, parts in self.groups.items() for name in parts}
        self.hashers = {name: hashlib.sha256() for name in self.groups}
        self.canonical_hashes = {}
        self.position = 0

    def consume(self, name, tensor):
        if self.position >= len(self.canonical_order) or name != self.canonical_order[self.position]:
            raise ValueError(f"Out-of-order or duplicate canonical weight: {name}")
        if tuple(tensor.shape) != self.shapes[name] or tensor.dtype != torch.bfloat16:
            raise ValueError(f"Canonical weight shape/dtype changed: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite canonical weight: {name}")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest = hashlib.sha256(raw).hexdigest()
        if name in self.group_for:
            self.hashers[self.group_for[name]].update(raw)
        self.canonical_hashes[name] = digest
        self.position += 1
        return digest

    def expected_checksums(self):
        if self.position != len(self.canonical_order):
            raise ValueError("Incomplete canonical policy stream")
        if self.tied and self.canonical_hashes["lm_head.weight"] != self.canonical_hashes["model.embed_tokens.weight"]:
            raise ValueError("Tied embedding and output-head exports disagree")
        return {CHECKSUM_PREFIX + name: value.hexdigest() for name, value in self.hashers.items()}

    def verify(self, actual):
        expected = self.expected_checksums()
        if not isinstance(actual, dict) or not actual:
            raise ValueError("Missing resident parameter checksums")
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        different = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        if missing or extra or different:
            raise ValueError(f"Resident policy does not match canonical export: missing={missing}, extra={extra}, different={different}")
        return {"passed": True, "canonical_tensor_count": len(self.shapes),
                "resident_parameter_count": len(expected), "comparison": "All packed resident parameter bytes SHA256 equal independently packed canonical exports",
                "layout": "Unquantized Qwen2; TP=PP=DP=PCP=1; aligned vocabulary; tied head deduplicated",
                "expected_resident_checksums": expected, "layout_sources": LAYOUT_SOURCES}
