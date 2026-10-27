"""Complete GeoRA/LoRA policies over TRL's native vLLM 0.32 weight transport.

TRL owns generation scheduling, GRPO, backward, and optimizer updates. This
module changes only the instance's policy publication and audits generation.
The server stays alive throughout updates 0 -> 1 -> 2. No checkpoint files are
used for weight transfer, and no old bridge package is imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading
import time

import torch

from resident_verification import PackedPolicyHasher


def tensor_hash(tensor):
    """Hash exact bytes, including BF16, without converting their values."""
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def canonical_sources(model, expected_schema):
    """Map every pristine HF tensor name to a source tensor or adapter layer.

    expected_schema maps names to (shape, dtype). GeoRA residuals and biases are
    buffers, so walking named_parameters() would silently omit policy state.
    """
    root = model.get_base_model() if hasattr(model, "peft_config") else model
    state = root.state_dict()
    sources = {}
    consumed = set()
    for name, layer in root.named_modules():
        prefix = name + "." if name else ""
        if all(hasattr(layer, key) for key in ("W_res", "A", "B", "scaling")):
            keys = {prefix + key for key in ("A", "B", "W_res")}
            if layer.bias is not None:
                keys.add(prefix + "bias")
                sources[prefix + "bias"] = ("tensor", layer.bias)
            actual = {key for key in state if key.startswith(prefix)}
            if keys != actual:
                raise ValueError(f"Unknown GeoRA state: {name}")
            sources[prefix + "weight"] = ("geora", layer)
            consumed.update(keys)
        elif hasattr(layer, "lora_A") and hasattr(layer, "lora_B"):
            active = list(layer.active_adapters)
            if len(active) != 1 or layer.disable_adapters or layer.merged_adapters:
                raise ValueError("Require one enabled, unmerged LoRA adapter")
            adapter = active[0]
            if set(layer.lora_A) != {adapter} or set(layer.lora_B) != {adapter}:
                raise ValueError("Unexpected additional LoRA adapters")
            if layer.fan_in_fan_out or getattr(layer, "use_dora", {}).get(adapter, False):
                raise ValueError("Only ordinary linear LoRA is supported")
            dropout = layer.lora_dropout[adapter]
            if getattr(dropout, "p", 0) != 0:
                raise ValueError("The frozen protocol requires zero adapter dropout")
            if layer.lora_A[adapter].bias is not None or layer.lora_B[adapter].bias is not None:
                raise ValueError("Adapter biases need an explicit export rule")
            keys = {
                prefix + "base_layer.weight",
                prefix + f"lora_A.{adapter}.weight",
                prefix + f"lora_B.{adapter}.weight",
            }
            if layer.base_layer.bias is not None:
                keys.add(prefix + "base_layer.bias")
                sources[prefix + "bias"] = ("tensor", layer.base_layer.bias)
            actual = {key for key in state if key.startswith(prefix)}
            if keys != actual:
                raise ValueError(f"Unknown LoRA state: {name}")
            sources[prefix + "weight"] = ("lora", (layer, adapter))
            consumed.update(keys)
    for name, value in state.items():
        if name not in consumed:
            if name in sources:
                raise ValueError(f"Ambiguous canonical tensor: {name}")
            sources[name] = ("tensor", value)
    if set(sources) != set(expected_schema):
        missing = sorted(set(expected_schema) - set(sources))
        extra = sorted(set(sources) - set(expected_schema))
        raise ValueError(f"Incomplete canonical policy: missing={missing}, extra={extra}")
    return sources


def dense_tensor(source, *, shape, dtype, device):
    """Materialize one canonical tensor; never merge into the learner in place."""
    kind, value = source
    with torch.no_grad():
        if kind == "geora":
            value = value.W_res.float() + float(value.scaling) * (value.B.float() @ value.A.float())
        elif kind == "lora":
            layer, adapter = value
            value = layer.base_layer.weight.float() + float(layer.scaling[adapter]) * (
                layer.lora_B[adapter].weight.float() @ layer.lora_A[adapter].weight.float()
            )
        value = value.detach().to(device=device, dtype=dtype).contiguous()
        if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
            raise ValueError("Canonical tensor has an unexpected shape or nonfinite values")
        return value


def validate_generation(result, expected_prompts, num_generations):
    """Require sampled-token probabilities for every active token, including EOS."""
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise ValueError("Expected the native TRL four-field generation result")
    prompts, completions, logprobs, token_ids = result
    if prompts != expected_prompts or not completions or logprobs is None or token_ids is None:
        raise ValueError("Missing or reordered prompts/completions/behavior probabilities")
    if any(len(rows) != len(prompts) for rows in (completions, logprobs, token_ids)):
        raise ValueError("Generation fields have different batch sizes")
    if not isinstance(num_generations, int) or num_generations < 1 or len(prompts) % num_generations:
        raise ValueError("Incomplete GRPO groups")
    for start in range(0, len(prompts), num_generations):
        if any(row != prompts[start] for row in prompts[start:start + num_generations]):
            raise ValueError("One GRPO group contains different prompts")
    token_count = 0
    for completion, values, ids in zip(completions, logprobs, token_ids, strict=True):
        if not completion or len(completion) != len(values) or len(values) != len(ids):
            raise ValueError("Missing completion token probabilities")
        for token, candidates, candidate_ids in zip(completion, values, ids, strict=True):
            if candidate_ids != [token] or len(candidates) != 1:
                raise ValueError("The sampled token does not match its probability")
            probability = candidates[0]
            if probability is None or not math.isfinite(probability) or probability > 1e-6:
                raise ValueError("Nonfinite or invalid sampled-token log probability")
        token_count += len(completion)
    return {"responses": len(completions), "tokens": token_count, "all_behavior_logprobs_finite": True}


def source_guard(model):
    """Detect source storage/version, trainability, or module-mode changes."""
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    return {
        "tensors": {name: (value.data_ptr(), value._version, value.dtype, value.device, tuple(value.shape))
                    for name, value in tensors},
        "parameters": {name: (id(value), value.requires_grad) for name, value in model.named_parameters()},
        "modes": {name: layer.training for name, layer in model.named_modules()},
    }


class NativePolicySync:
    def __init__(self, trainer, *, expected_schema, audit_dir, fixed_rows,
                 expected_sync_versions=(0, 1, 2),
                 expected_generation_versions=(0, 0, 0, 0, 1, 1, 1, 1),
                 fixed_token_checker=None, base_sampling_seed=None,
                 responses_per_rollout=32, num_generations=8, rollouts_per_update=4):
        self.trainer = trainer
        self.generation = trainer.vllm_generation
        self.client = self.generation.vllm_client
        self.model = trainer.accelerator.unwrap_model(self.generation.model)
        self.expected_schema = expected_schema
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_rows = fixed_rows
        self.expected_sync_versions = tuple(expected_sync_versions)
        self.expected_generation_versions = tuple(expected_generation_versions)
        self.fixed_token_checker = fixed_token_checker
        self.base_sampling_seed = base_sampling_seed
        self.responses_per_rollout = responses_per_rollout
        self.num_generations = num_generations
        self.rollouts_per_update = rollouts_per_update
        self.applied_version = None
        self.sync_records = []
        self.generation_records = []
        self.poisoned = False
        self.lock = threading.Lock()
        if trainer.accelerator.num_processes != 1 or self.generation.mode != "server":
            raise ValueError("This closure requires one learner and one separate vLLM server")
        if self.generation.logprobs != 0 or not trainer.args.vllm_importance_sampling_correction:
            raise ValueError("Sampled-token probabilities and explicit importance correction are required")
        if not fixed_rows:
            raise ValueError("Fixed token sequences are required before native synchronization")
        for name, value in (("responses_per_rollout", responses_per_rollout),
                            ("num_generations", num_generations),
                            ("rollouts_per_update", rollouts_per_update)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_generations < 2 or responses_per_rollout % num_generations:
            raise ValueError("A generation rollout must contain complete GRPO groups of at least two responses")
        self.answers_per_update = responses_per_rollout * rollouts_per_update
        self.generation_schedule = {
            "responses_per_rollout": responses_per_rollout,
            "num_generations": num_generations,
            "questions_per_rollout": responses_per_rollout // num_generations,
            "rollouts_per_update": rollouts_per_update,
            "responses_per_update": self.answers_per_update,
            "questions_per_update": self.answers_per_update // num_generations,
            "request_seed_formula": "base_seed + num_generations * (global_step * rollouts_per_update + rollout_in_update)",
            "batch_invariant_sampling_claim": False,
        }
        if base_sampling_seed is not None and (type(base_sampling_seed) is not int or base_sampling_seed < 0):
            raise ValueError("The base sampling seed must be a nonnegative integer")
        if (not self.expected_sync_versions
                or (not self.expected_generation_versions and len(self.expected_sync_versions) != 1)
                or any(type(value) is not int or value < 0 for value in self.expected_sync_versions)
                or any(type(value) is not int or value < 0 for value in self.expected_generation_versions)
                or any(b != a + 1 for a, b in zip(self.expected_sync_versions, self.expected_sync_versions[1:]))
                or any(value not in self.expected_sync_versions[:-1] for value in self.expected_generation_versions)):
            raise ValueError("Require consecutive updates, or one fixed publication for a generation-free audit")
        expected_schedule = tuple(version for version in self.expected_sync_versions[:-1]
                                  for _ in range(rollouts_per_update))
        if self.expected_generation_versions != expected_schedule:
            raise ValueError("Generation version schedule must contain every configured rollout for each update")
        if self.expected_generation_versions:
            # Real GRPOConfig exposes all these fields. Tiny CPU transport fixtures
            # omit them, so independently validate every field that is available.
            args = trainer.args
            if hasattr(args, "generation_batch_size") and args.generation_batch_size != responses_per_rollout:
                raise ValueError("responses_per_rollout differs from TRL generation_batch_size")
            if hasattr(args, "num_generations") and args.num_generations != num_generations:
                raise ValueError("num_generations differs from the TRL generation group")
            batch_fields = ("per_device_train_batch_size", "gradient_accumulation_steps")
            present = [hasattr(args, name) for name in batch_fields]
            if any(present) and not all(present):
                raise ValueError("Both training micro batch and gradient accumulation are required")
            if all(present):
                micro, accumulation = (getattr(args, name) for name in batch_fields)
                if any(type(value) is not int or value < 1 for value in (micro, accumulation)):
                    raise ValueError("Training micro batch and gradient accumulation must be positive integers")
                if micro * trainer.accelerator.num_processes * accumulation != self.answers_per_update:
                    raise ValueError("Configured responses per update differ from the effective training batch")
                if responses_per_rollout % (micro * trainer.accelerator.num_processes):
                    raise ValueError("Generation responses must be divisible by the global training micro batch")
                steps = responses_per_rollout // (micro * trainer.accelerator.num_processes)
                if hasattr(args, "steps_per_generation") and args.steps_per_generation != steps:
                    raise ValueError("Configured rollout differs from TRL steps_per_generation")
                if getattr(args, "num_iterations", 1) != 1:
                    raise ValueError("This synchronous schedule requires one iteration per generated batch")
                self.generation_schedule.update({"training_micro_batch_size": micro,
                                                 "gradient_accumulation_steps": accumulation,
                                                 "steps_per_generation": steps})
        canonical_sources(self.model, expected_schema)
        # Do not replay a potentially partially applied POST after a timeout.
        from requests.adapters import HTTPAdapter
        self.client.session.mount("http://", HTTPAdapter(max_retries=0))
        self.client.session.mount("https://", HTTPAdapter(max_retries=0))
        self.original_generate = self.generation.generate
        self.generation.sync_weights = self.sync_weights
        self.generation.generate = self.generate
        trainer._last_loaded_step = -1

    def _emit(self, name, record):
        with (self.audit_dir / name).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _request(self, method, endpoint, **kwargs):
        response = self.client.session.request(
            method, self.client.base_url + endpoint, timeout=(15, 600), **kwargs
        )
        response.raise_for_status()
        return response.json()

    def _version(self):
        return self._request("GET", "/weight_info")["weight_version"]

    def _enter(self):
        if self.poisoned or not self.lock.acquire(blocking=False):
            raise RuntimeError("Failed or concurrent policy publication/generation")

    def sync_weights(self):
        self._enter()
        started = time.monotonic()
        phase_started = started
        phase_seconds = {}
        active_phase = "prepare"

        def finish_phase(name, next_phase):
            nonlocal phase_started, active_phase
            now = time.monotonic()
            phase_seconds[name] = now - phase_started
            phase_started = now
            active_phase = next_phase

        current = int(self.trainer.state.global_step)
        try:
            if current == self.applied_version:
                return
            index = len(self.sync_records)
            if index >= len(self.expected_sync_versions) or current != self.expected_sync_versions[index]:
                raise ValueError(f"Policy publications must follow {self.expected_sync_versions}")
            if self.trainer.accelerator.device.type == "cuda":
                # Packed transfer uses its own streams; finish the optimizer first.
                torch.cuda.synchronize(self.trainer.accelerator.device)
            before = source_guard(self.model)
            sources = canonical_sources(self.model, self.expected_schema)
            packed = PackedPolicyHasher(self.model.config, self.expected_schema)
            metadata = [(name, str(dtype).removeprefix("torch."), list(shape))
                        for name in packed.canonical_order
                        for shape, dtype in [self.expected_schema[name]]]
            descriptors = {}

            def tensor_stream():
                for name in packed.canonical_order:
                    shape, dtype = self.expected_schema[name]
                    tensor = dense_tensor(
                        sources[name], shape=shape, dtype=dtype, device=self.trainer.accelerator.device
                    )
                    descriptors[name] = packed.consume(name, tensor)
                    yield name, tensor

            finish_phase("prepare", "pause")
            self._request("POST", "/pause", params={"mode": "wait", "clear_cache": True})
            finish_phase("pause", "export_expected_hash_and_transfer")
            self.client.update_named_params(metadata, tensor_stream())
            finish_phase("export_expected_hash_and_transfer", "learner_integrity")
            if set(descriptors) != set(self.expected_schema) or source_guard(self.model) != before:
                raise ValueError("Incomplete weight stream or mutated learner state")
            finish_phase("learner_integrity", "cache_and_version")
            if self._request("POST", "/reset_prefix_cache").get("success") is not True:
                raise ValueError("The server did not confirm prefix-cache invalidation")
            self._request("POST", "/update_weight_version", json={"new_version": str(current)})
            if self._version() != str(current):
                raise ValueError("The server did not acknowledge the new policy version")
            finish_phase("cache_and_version", "resident_checksum_and_verify")
            resident = self._request("POST", "/weight_checker", json={"action": "checksum"})["checksums"]
            resident_verification = packed.verify(resident)
            changed = None if not self.sync_records else resident != self.sync_records[-1]["resident_checksums"]
            finish_phase("resident_checksum_and_verify", "resume")
            self._request("POST", "/resume")
            finish_phase("resume", "fixed_probability")
            comparison = (self.fixed_token_checker(self, current) if self.fixed_token_checker is not None
                          else self._fixed_token_check(current))
            finish_phase("fixed_probability", "record")
            record = {
                "policy_version": current,
                "total_seconds": time.monotonic() - started,
                "phase_seconds": phase_seconds,
                "timing_scope": "Host wall-clock phases include RPC waits. export_expected_hash_and_transfer includes canonical dense materialization, validation, expected byte hashing/host copies and the native transfer call; it is not isolated RCCL time or device-kernel profiling.",
                "generation_schedule": self.generation_schedule,
                "canonical_tensor_count": len(descriptors),
                "canonical_tensor_sha256": descriptors,
                "resident_checksums": resident,
                "resident_verification": resident_verification,
                "resident_policy_bytes_changed_since_previous_publication": changed,
                "resident_hash_layout": "Independently predicted native Qwen2 packed layout, exact byte comparison",
                "fixed_token_comparison": comparison,
                "transport": "native TRL packed NCCL/RCCL; no checkpoint transfer",
            }
            self._emit("sync.jsonl", record)
            self.sync_records.append(record)
            self.applied_version = current
            print(f"[sync] policy v{current} resident; {len(descriptors)} canonical tensors published", flush=True)
        except BaseException as error:
            self.poisoned = True
            self._emit("failures.jsonl", {"stage": "sync", "version": current, "error": repr(error),
                                          "active_phase": active_phase, "completed_phase_seconds": phase_seconds,
                                          "active_phase_elapsed_seconds": time.monotonic() - phase_started})
            raise
        finally:
            self.lock.release()

    def _fixed_token_check(self, current):
        sequences = [row["prompt_ids"] + row["completion_ids"] for row in self.fixed_rows]
        prompt_lengths = [len(row["prompt_ids"]) for row in self.fixed_rows]
        result = self.client.get_sequence_logprobs(
            sequences, prompt_lengths, top_logprobs=1, temperature=1.0
        )
        native = []
        modes = [(layer, layer.training) for layer in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                for sequence, prompt_length in zip(sequences, prompt_lengths, strict=True):
                    ids = torch.tensor([sequence], device=self.trainer.accelerator.device)
                    logits = self.model(input_ids=ids, use_cache=False).logits[0, prompt_length - 1:-1].float()
                    selected = logits.gather(1, ids[0, prompt_length:, None]).squeeze(1)
                    values = selected - torch.logsumexp(logits, dim=-1)
                    native.append(values.cpu().tolist())
                    del ids, logits, selected, values
        finally:
            for layer, mode in modes:
                layer.training = mode
        deltas = []
        server_rows = []
        for row, native_values, server_values, token_ids in zip(
            self.fixed_rows, native, result["actual_logprobs"], result["actual_token_ids"], strict=True
        ):
            if token_ids != [[token] for token in row["completion_ids"]] or len(server_values) != len(native_values):
                raise ValueError("Fixed-token scoring omitted or changed tokens")
            server_values = [values[0] for values in server_values]
            if any(not math.isfinite(value) for value in native_values + server_values):
                raise ValueError("Missing or nonfinite fixed-token probabilities")
            server_rows.append(server_values)
            deltas.extend(a - b for a, b in zip(native_values, server_values, strict=True))
        if self._version() != str(current):
            raise ValueError("Policy changed while scoring fixed sequences")
        payload = {"policy_version": current, "native_logprobs": native, "server_logprobs": server_rows,
                   "mean_abs_log_ratio": sum(abs(value) for value in deltas) / len(deltas),
                   "max_abs_log_ratio": max(abs(value) for value in deltas),
                   "threshold_policy": "Descriptive; no fitted equivalence threshold"}
        (self.audit_dir / f"fixed_tokens_v{current}.json").write_text(json.dumps(payload, indent=2) + "\n")
        return {key: value for key, value in payload.items() if key not in ("native_logprobs", "server_logprobs")}

    def generate(self, *args, **kwargs):
        self._enter()
        current = int(self.trainer.state.global_step)
        started = time.monotonic()
        original_generation_kwargs = None
        try:
            if self.applied_version != current or self._version() != str(current):
                raise RuntimeError("Generation attempted before the current policy was published")
            prompts = kwargs.get("prompts", args[0] if args else None)
            count = kwargs.get("num_generations", args[2] if len(args) > 2 else None)
            index = len(self.generation_records)
            if (count != self.num_generations or prompts is None or len(prompts) != self.responses_per_rollout
                    or index >= len(self.expected_generation_versions)
                    or current != self.expected_generation_versions[index]):
                raise ValueError("Generation is outside the configured GRPO rollout/version schedule")
            rollout_in_update = sum(row["policy_version"] == current for row in self.generation_records)
            request_seed = None
            if self.base_sampling_seed is not None:
                previous = getattr(self.generation, "generation_kwargs", None)
                if not isinstance(previous, dict):
                    raise ValueError("Seeded generation requires TRL generation_kwargs")
                # Request-local seed choice depends on saved global_step and
                # the frozen batching schedule, not a restarted server RNG.
                # A different batching schedule need not sample identical tokens.
                request_seed = self.base_sampling_seed + self.num_generations * (
                    current * self.rollouts_per_update + rollout_in_update)
                original_generation_kwargs = previous
                self.generation.generation_kwargs = {**previous, "seed": request_seed}
            result = self.original_generate(*args, **kwargs)
            summary = validate_generation(result, prompts, count)
            if summary["responses"] != self.responses_per_rollout:
                raise ValueError("Response count differs from the configured rollout")
            if self._version() != str(current):
                raise ValueError("Server policy changed during generation")
            record = {"policy_version": current, "seconds": time.monotonic() - started, **summary,
                      "request_seed": request_seed, "rollout_in_update": rollout_in_update,
                      "generation_schedule": self.generation_schedule,
                      "prompt_ids": result[0], "completion_ids": result[1],
                      "behavior_logprobs": [[values[0] for values in row] for row in result[2]]}
            self._emit("rollouts.jsonl", record)
            self.generation_records.append(record)
            return result
        except BaseException as error:
            self.poisoned = True
            self._emit("failures.jsonl", {"stage": "generation", "version": current, "error": repr(error)})
            raise
        finally:
            if original_generation_kwargs is not None:
                self.generation.generation_kwargs = original_generation_kwargs
            self.lock.release()

    def verify_final(self):
        final = self.expected_sync_versions[-1]
        if self.poisoned or tuple(row["policy_version"] for row in self.sync_records) != self.expected_sync_versions:
            raise ValueError("All configured policy publications must complete")
        if tuple(row["policy_version"] for row in self.generation_records) != self.expected_generation_versions:
            raise ValueError("Updates must consume the configured version-matched rollouts")
        if self._version() != str(final):
            raise ValueError("Final policy is not resident")
        prompt = self.fixed_rows[0]["prompt_ids"]
        output = self.client.generate(prompts=[prompt], n=1, max_tokens=32, temperature=1.0, logprobs=0)
        converted = (output["prompt_ids"], output["completion_ids"], output["logprobs"], output["logprob_token_ids"])
        summary = validate_generation(converted, [prompt], 1)
        if self._version() != str(final):
            raise ValueError("Final generation used a different policy version")
        (self.audit_dir / f"final_generation_v{final}.json").write_text(json.dumps(output, indent=2) + "\n")
        return {"passed": True, "sync_versions": list(self.expected_sync_versions),
                "generation_schedule": self.generation_schedule,
                "training_rollouts": len(self.generation_records),
                "training_responses": sum(row["responses"] for row in self.generation_records),
                "final_generation": summary, "all_resident_weights_verified": True,
                "resident_policy_bytes_changed_between_versions": [
                    row["resident_policy_bytes_changed_since_previous_publication"] for row in self.sync_records[1:]]}


def install_native_sync(trainer, *, expected_schema, audit_dir, fixed_rows,
                        expected_sync_versions=(0, 1, 2),
                        expected_generation_versions=(0, 0, 0, 0, 1, 1, 1, 1),
                        fixed_token_checker=None, base_sampling_seed=None,
                        responses_per_rollout=32, num_generations=8, rollouts_per_update=4):
    return NativePolicySync(trainer, expected_schema=expected_schema, audit_dir=audit_dir, fixed_rows=fixed_rows,
                            expected_sync_versions=expected_sync_versions,
                            expected_generation_versions=expected_generation_versions,
                            fixed_token_checker=fixed_token_checker,
                            base_sampling_seed=base_sampling_seed,
                            responses_per_rollout=responses_per_rollout,
                            num_generations=num_generations,
                            rollouts_per_update=rollouts_per_update)
