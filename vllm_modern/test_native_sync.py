"""Policy coverage and failure-barrier tests; no vLLM installation or GPU needed."""

import tempfile
import hashlib
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from native_sync import NativePolicySync, canonical_sources, dense_tensor, validate_generation
from resident_verification import CHECKSUM_PREFIX, PackedPolicyHasher, qwen2_layout


class GeoRALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        self.B = nn.Parameter(torch.tensor([[0.5], [-0.25]]))
        self.register_buffer("W_res", torch.eye(2, dtype=torch.bfloat16))
        self.register_buffer("bias", torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
        self.scaling = 0.5


class LoRALayer(nn.Module):
    """Ordinary PEFT Linear's state contract, without installing PEFT locally."""
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, dtype=torch.bfloat16)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged_adapters = []
        self.fan_in_fan_out = False
        self.use_dora = {"default": False}
        self.scaling = {"default": 0.5}


def fixture():
    model = nn.Sequential(GeoRALayer(), nn.Linear(2, 1, dtype=torch.bfloat16))
    schema = {
        "0.weight": ((2, 2), torch.bfloat16),
        "0.bias": ((2,), torch.bfloat16),
        "1.weight": ((1, 2), torch.bfloat16),
        "1.bias": ((1,), torch.bfloat16),
    }
    return model, schema


def put_parameter(model, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in model._modules:
            model.add_module(part, nn.Module())
        model = model._modules[part]
    model.register_parameter(parts[-1], nn.Parameter(value, requires_grad=False))


def qwen_fixture(tied=True):
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen2", hidden_size=4, vocab_size=64,
                                   intermediate_size=6, num_attention_heads=2,
                                   num_key_value_heads=1, num_hidden_layers=1,
                                   tie_word_embeddings=tied)
    shapes, _ = qwen2_layout(model.config)
    for index, (name, shape) in enumerate(shapes.items()):
        put_parameter(model, name, torch.full(shape, (index + 1) / 16, dtype=torch.bfloat16))
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model, {name: (shape, torch.bfloat16) for name, shape in shapes.items()}


def fake_resident_weights(canonical, tied=True):
    """A fixture loader independent of PackedPolicyHasher's stream logic."""
    packed = {}
    prefix = "model.layers.0."
    for suffix in ("weight", "bias"):
        names = [prefix + f"self_attn.{part}_proj.{suffix}" for part in ("q", "k", "v")]
        packed[prefix + f"self_attn.qkv_proj.{suffix}"] = torch.cat([canonical[name] for name in names])
    names = [prefix + f"mlp.{part}_proj.weight" for part in ("gate", "up")]
    packed[prefix + "mlp.gate_up_proj.weight"] = torch.cat([canonical[name] for name in names])
    for name, tensor in canonical.items():
        if any(f".{part}_proj." in name for part in ("q", "k", "v", "gate", "up")):
            continue
        if name == "lm_head.weight" and tied:
            continue
        packed[name] = tensor.clone()
    return packed


def checksums(weights):
    return {CHECKSUM_PREFIX + name: hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
            for name, tensor in weights.items()}


class FakeNativeClient:
    def __init__(self, *, corrupt=False):
        self.base_url = "http://127.0.0.1:1"
        self.session = SimpleNamespace(mount=Mock(), request=self.request)
        self.calls = []
        self.version = None
        self.canonical = {}
        self.corrupt = corrupt

    def update_named_params(self, metadata, stream):
        self.canonical = {name: tensor.clone() for name, tensor in stream}
        if [row[0] for row in metadata] != list(self.canonical):
            raise ValueError("Transport metadata and tensors differ")

    def request(self, method, url, **kwargs):
        endpoint = url.removeprefix(self.base_url)
        self.calls.append(endpoint)
        if endpoint == "/reset_prefix_cache":
            value = {"success": True}
        elif endpoint == "/update_weight_version":
            self.version = kwargs["json"]["new_version"]
            value = {"success": True}
        elif endpoint == "/weight_info":
            value = {"weight_version": self.version}
        elif endpoint == "/weight_checker":
            resident = fake_resident_weights(self.canonical)
            if self.corrupt:
                resident["model.layers.0.self_attn.qkv_proj.bias"][0] += 1
            value = {"checksums": checksums(resident)}
        else:
            value = {}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)


def native_trainer(model, client):
    generation = SimpleNamespace(vllm_client=client, model=model, mode="server", logprobs=0,
                                 generate=Mock())
    return SimpleNamespace(vllm_generation=generation,
                           accelerator=SimpleNamespace(num_processes=1, device=torch.device("cpu"),
                                                       unwrap_model=lambda value: value),
                           args=SimpleNamespace(vllm_importance_sampling_correction=True),
                           state=SimpleNamespace(global_step=0))


class NativeSyncTests(unittest.TestCase):
    def setUp(self):
        requests_stub = ModuleType("requests")
        adapters_stub = ModuleType("requests.adapters")
        adapters_stub.HTTPAdapter = Mock()
        modules = patch.dict("sys.modules", {"requests": requests_stub, "requests.adapters": adapters_stub})
        modules.start()
        self.addCleanup(modules.stop)

    def test_one_fixed_publication_supports_probability_only_probe(self):
        model, schema = qwen_fixture()
        client = FakeNativeClient()
        trainer = native_trainer(model, client)
        trainer.state.global_step = 1
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(
                trainer, expected_schema=schema, audit_dir=directory,
                fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                expected_sync_versions=(1,), expected_generation_versions=(),
                fixed_token_checker=lambda session, current: {"probability_probe_version": current},
            )
            sync.sync_weights()
            self.assertEqual(sync.applied_version, 1)
            self.assertEqual(sync.generation_records, [])
            self.assertTrue(sync.sync_records[0]["resident_verification"]["passed"])

    def test_generation_free_probe_rejects_multiple_policy_versions(self):
        model, schema = qwen_fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "one fixed publication"):
                NativePolicySync(
                    native_trainer(model, FakeNativeClient()), expected_schema=schema,
                    audit_dir=directory, fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                    expected_sync_versions=(0, 1), expected_generation_versions=(),
                )

    def test_full_policy_includes_frozen_buffers_and_preserves_learner(self):
        model, schema = fixture()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parameters = [id(value) for value in model.parameters()]
        sources = canonical_sources(model, schema)
        exported = {name: dense_tensor(sources[name], shape=shape, dtype=dtype, device="cpu")
                    for name, (shape, dtype) in schema.items()}
        expected = (model[0].W_res.float() + 0.5 * (model[0].B @ model[0].A)).bfloat16()
        self.assertTrue(torch.equal(exported["0.weight"], expected))
        self.assertTrue(torch.equal(exported["0.bias"], model[0].bias))
        self.assertEqual(parameters, [id(value) for value in model.parameters()])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertTrue(all(value.grad is None for value in model.parameters()))

    def test_rejects_incomplete_canonical_schema(self):
        model, schema = fixture()
        del schema["0.bias"]
        with self.assertRaisesRegex(ValueError, "Incomplete canonical policy"):
            canonical_sources(model, schema)

    def test_lora_export_includes_frozen_base_and_bias_without_merging_learner(self):
        model = nn.Sequential(LoRALayer())
        schema = {"0.weight": ((2, 2), torch.bfloat16), "0.bias": ((2,), torch.bfloat16)}
        before = {name: value.clone() for name, value in model.state_dict().items()}
        sources = canonical_sources(model, schema)
        exported = {name: dense_tensor(sources[name], shape=shape, dtype=dtype, device="cpu")
                    for name, (shape, dtype) in schema.items()}
        layer = model[0]
        expected = (layer.base_layer.weight.float() + 0.5 *
                    (layer.lora_B["default"].weight @ layer.lora_A["default"].weight)).bfloat16()
        self.assertTrue(torch.equal(exported["0.weight"], expected))
        self.assertTrue(torch.equal(exported["0.bias"], layer.base_layer.bias))
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))

    def test_rejects_unknown_geora_buffer(self):
        model, schema = fixture()
        model[0].register_buffer("unmapped_policy_state", torch.ones(1))
        with self.assertRaisesRegex(ValueError, "Unknown GeoRA state"):
            canonical_sources(model, schema)

    def test_sampled_probabilities_cover_eos(self):
        result = ([[1, 2], [1, 2]], [[3, 0], [4, 0]],
                  [[[-0.2], [-0.1]], [[-0.3], [-0.1]]], [[[3], [0]], [[4], [0]]])
        summary = validate_generation(result, result[0], 2)
        self.assertEqual(summary["tokens"], 4)
        result[2][0][-1] = [None]
        with self.assertRaisesRegex(ValueError, "log probability"):
            validate_generation(result, result[0], 2)

    def test_rejects_token_and_group_mismatch(self):
        result = ([[1], [2]], [[3], [4]], [[[-0.2]], [[-0.3]]], [[[3]], [[4]]])
        with self.assertRaisesRegex(ValueError, "group"):
            validate_generation(result, result[0], 2)
        result[0][1] = [1]
        result[3][1] = [[9]]
        with self.assertRaisesRegex(ValueError, "sampled token"):
            validate_generation(result, result[0], 2)

    def test_partial_publication_poisoned_and_cannot_generate(self):
        model, schema = qwen_fixture()
        session = SimpleNamespace(mount=Mock())
        response = Mock()
        response.json.return_value = {}
        session.request = Mock(return_value=response)

        def interrupted_update(metadata, stream):
            next(stream)
            raise RuntimeError("connection failed after first tensor")

        original_generate = Mock()
        client = SimpleNamespace(session=session, base_url="http://127.0.0.1:1",
                                 update_named_params=interrupted_update)
        generation = SimpleNamespace(vllm_client=client, model=model, mode="server",
                                     logprobs=0, generate=original_generate)
        accelerator = SimpleNamespace(num_processes=1, device=torch.device("cpu"), unwrap_model=lambda value: value)
        trainer = SimpleNamespace(vllm_generation=generation, accelerator=accelerator,
                                  args=SimpleNamespace(vllm_importance_sampling_correction=True),
                                  state=SimpleNamespace(global_step=0))
        requests_stub = ModuleType("requests")
        adapters_stub = ModuleType("requests.adapters")
        adapters_stub.HTTPAdapter = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"requests": requests_stub, "requests.adapters": adapters_stub}
        ):
            sync = NativePolicySync(trainer, expected_schema=schema, audit_dir=directory,
                                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}])
            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                sync.sync_weights()
            self.assertTrue(sync.poisoned)
            self.assertIsNone(sync.applied_version)
            with self.assertRaisesRegex(RuntimeError, "Failed or concurrent"):
                sync.generate(prompts=[[1]], images=None, num_generations=1)
            original_generate.assert_not_called()
            self.assertEqual(session.request.call_count, 1)
            self.assertTrue(session.request.call_args.args[1].endswith("/pause"))

    def test_complete_native_publication_checks_all_resident_bytes(self):
        model, schema = qwen_fixture()
        client = FakeNativeClient()
        trainer = native_trainer(model, client)
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(trainer, expected_schema=schema, audit_dir=directory,
                                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                                    fixed_token_checker=lambda sync, version: {"version": version, "fixture": True})
            sync.sync_weights()
            self.assertEqual(sync.applied_version, 0)
            self.assertTrue(sync.sync_records[0]["resident_verification"]["passed"])
            self.assertEqual(sync.sync_records[0]["resident_verification"]["resident_parameter_count"], 9)
            self.assertEqual(set(sync.sync_records[0]["canonical_tensor_sha256"]), set(schema))
            self.assertIn("/resume", client.calls)
            record = sync.sync_records[0]
            self.assertEqual(record["generation_schedule"]["responses_per_update"], 128)
            self.assertEqual(record["generation_schedule"]["rollouts_per_update"], 4)
            self.assertEqual(set(record["phase_seconds"]), {
                "prepare", "pause", "export_expected_hash_and_transfer", "learner_integrity",
                "cache_and_version", "resident_checksum_and_verify", "resume", "fixed_probability",
            })
            self.assertTrue(all(value >= 0 for value in record["phase_seconds"].values()))
            self.assertLessEqual(sum(record["phase_seconds"].values()), record["total_seconds"])
            self.assertIn("not isolated RCCL", record["timing_scope"])

    def test_corrupt_resident_cannot_resume_or_generate(self):
        model, schema = qwen_fixture()
        client = FakeNativeClient(corrupt=True)
        trainer = native_trainer(model, client)
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(trainer, expected_schema=schema, audit_dir=directory,
                                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}])
            with self.assertRaisesRegex(ValueError, "Resident policy does not match"):
                sync.sync_weights()
            self.assertTrue(sync.poisoned)
            self.assertIsNone(sync.applied_version)
            self.assertNotIn("/resume", client.calls)

    def test_resume_segment_can_start_at_saved_policy_version(self):
        model, schema = qwen_fixture()
        client = FakeNativeClient()
        trainer = native_trainer(model, client)
        trainer.state.global_step = 1
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(trainer, expected_schema=schema, audit_dir=directory,
                                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                                    expected_sync_versions=(1, 2), expected_generation_versions=(1, 1, 1, 1),
                                    fixed_token_checker=lambda sync, version: {"version": version})
            sync.sync_weights()
            trainer.state.global_step = 2
            sync.sync_weights()
            self.assertEqual([row["policy_version"] for row in sync.sync_records], [1, 2])
            self.assertFalse(sync.sync_records[-1]["resident_policy_bytes_changed_since_previous_publication"])

    def test_request_seed_uses_saved_version_and_restores_generation_kwargs(self):
        model, schema = qwen_fixture()
        client = FakeNativeClient()
        trainer = native_trainer(model, client)
        trainer.state.global_step = 1
        previous = {"seed": 999, "fixture_option": "preserve"}
        trainer.vllm_generation.generation_kwargs = previous
        prompts = [[value] for value in (11, 22, 33, 44) for _ in range(8)]
        observed_seeds = []

        def generated(**kwargs):
            observed_seeds.append(trainer.vllm_generation.generation_kwargs["seed"])
            return (prompts, [[5, 0] for _ in prompts],
                    [[[-0.2], [-0.3]] for _ in prompts], [[[5], [0]] for _ in prompts])

        trainer.vllm_generation.generate = Mock(side_effect=generated)
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(trainer, expected_schema=schema, audit_dir=directory,
                                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                                    expected_sync_versions=(1, 2), expected_generation_versions=(1, 1, 1, 1),
                                    fixed_token_checker=lambda sync, version: {"version": version},
                                    base_sampling_seed=42)
            sync.sync_weights()
            for _ in range(2):
                sync.generate(prompts=prompts, images=None, num_generations=8)
                self.assertIs(trainer.vllm_generation.generation_kwargs, previous)
            self.assertEqual(observed_seeds, [74, 82])
            self.assertEqual([row["request_seed"] for row in sync.generation_records], [74, 82])
            self.assertEqual([row["rollout_in_update"] for row in sync.generation_records], [0, 1])

    def test_request_seed_restores_generation_kwargs_on_generation_failure(self):
        model, schema = qwen_fixture()
        client = FakeNativeClient()
        trainer = native_trainer(model, client)
        previous = {"fixture_option": "preserve"}
        trainer.vllm_generation.generation_kwargs = previous
        trainer.vllm_generation.generate = Mock(side_effect=RuntimeError("server generation failed"))
        prompts = [[11] for _ in range(32)]
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(trainer, expected_schema=schema, audit_dir=directory,
                                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                                    fixed_token_checker=lambda sync, version: {"version": version},
                                    base_sampling_seed=42)
            sync.sync_weights()
            with self.assertRaisesRegex(RuntimeError, "server generation failed"):
                sync.generate(prompts=prompts, images=None, num_generations=8)
            self.assertIs(trainer.vllm_generation.generation_kwargs, previous)
            self.assertTrue(sync.poisoned)
            self.assertEqual(sync.generation_records, [])

    def test_configured_generation64_and128_preserve128_response_updates(self):
        for responses, rollouts in ((64, 2), (128, 1)):
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as directory:
                model, schema = qwen_fixture()
                client = FakeNativeClient()
                trainer = native_trainer(model, client)
                trainer.args = SimpleNamespace(
                    vllm_importance_sampling_correction=True, per_device_train_batch_size=8,
                    gradient_accumulation_steps=16, generation_batch_size=responses,
                    num_generations=8, steps_per_generation=responses // 8, num_iterations=1,
                )
                trainer.state.global_step = 1
                previous = {"fixture_option": "preserve"}
                trainer.vllm_generation.generation_kwargs = previous
                prompts = [[question] for question in range(responses // 8) for _ in range(8)]
                observed = []

                def generated(**kwargs):
                    observed.append(trainer.vllm_generation.generation_kwargs["seed"])
                    return (kwargs["prompts"], [[5, 0] for _ in prompts],
                            [[[-0.2], [-0.3]] for _ in prompts], [[[5], [0]] for _ in prompts])

                trainer.vllm_generation.generate = Mock(side_effect=generated)
                sync = NativePolicySync(
                    trainer, expected_schema=schema, audit_dir=directory,
                    fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                    expected_sync_versions=(1, 2), expected_generation_versions=(1,) * rollouts,
                    fixed_token_checker=lambda session, version: {"version": version},
                    base_sampling_seed=42, responses_per_rollout=responses,
                    num_generations=8, rollouts_per_update=rollouts,
                )
                sync.sync_weights()
                for _ in range(rollouts):
                    sync.generate(prompts=prompts, images=None, num_generations=8)
                    self.assertIs(trainer.vllm_generation.generation_kwargs, previous)
                self.assertEqual(observed, [42 + 8 * (rollouts + index) for index in range(rollouts)])
                self.assertEqual(sum(row["responses"] for row in sync.generation_records), 128)
                self.assertEqual(sync.generation_schedule["steps_per_generation"], responses // 8)
                self.assertTrue(all(row["generation_schedule"] == sync.generation_schedule
                                    for row in sync.generation_records))
                trainer.state.global_step = 2
                sync.sync_weights()
                with self.assertRaisesRegex(ValueError, "schedule"):
                    sync.generate(prompts=prompts, images=None, num_generations=8)
                self.assertTrue(sync.poisoned)
                self.assertEqual(len(observed), rollouts)

    def test_configured_generation_group_is_used_in_request_and_seed(self):
        model, schema = qwen_fixture()
        trainer = native_trainer(model, FakeNativeClient())
        trainer.state.global_step = 1
        trainer.vllm_generation.generation_kwargs = {}
        prompts = [[question] for question in range(16) for _ in range(4)]
        observed = []

        def generated(**kwargs):
            observed.append(trainer.vllm_generation.generation_kwargs["seed"])
            return (prompts, [[5] for _ in prompts], [[[-0.2]] for _ in prompts], [[[5]] for _ in prompts])

        trainer.vllm_generation.generate = Mock(side_effect=generated)
        with tempfile.TemporaryDirectory() as directory:
            sync = NativePolicySync(
                trainer, expected_schema=schema, audit_dir=directory,
                fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                expected_sync_versions=(1, 2), expected_generation_versions=(1, 1),
                fixed_token_checker=lambda session, version: {"version": version}, base_sampling_seed=42,
                responses_per_rollout=64, num_generations=4, rollouts_per_update=2,
            )
            sync.sync_weights()
            sync.generate(prompts=prompts, images=None, num_generations=4)
            self.assertEqual(observed, [50])
            self.assertEqual(sync.generation_schedule["questions_per_update"], 32)

    def test_invalid_batching_or_version_schedules_fail_before_publication(self):
        cases = [
            ({"responses_per_rollout": 63}, "complete GRPO"),
            ({"responses_per_rollout": 0}, "positive integer"),
            ({"num_generations": True}, "positive integer"),
            ({"num_generations": 1}, "complete GRPO"),
            ({"rollouts_per_update": 0}, "positive integer"),
            ({"expected_generation_versions": (0, 0, 0, 1, 1, 1, 1)}, "version schedule"),
            ({"expected_generation_versions": (0, 1, 0, 0, 1, 1, 1, 1)}, "version schedule"),
            ({"responses_per_rollout": 64}, "effective training batch"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                model, schema = qwen_fixture()
                client = FakeNativeClient()
                trainer = native_trainer(model, client)
                trainer.args.per_device_train_batch_size = 4
                trainer.args.gradient_accumulation_steps = 32
                with self.assertRaisesRegex(ValueError, message):
                    NativePolicySync(
                        trainer, expected_schema=schema, audit_dir=directory,
                        fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}], **kwargs,
                    )
                self.assertEqual(client.calls, [])

    def test_trl_batch_contract_mismatch_is_rejected(self):
        for field, wrong in (("generation_batch_size", 64), ("num_generations", 4),
                             ("steps_per_generation", 16), ("num_iterations", 2),
                             ("gradient_accumulation_steps", 16)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                model, schema = qwen_fixture()
                trainer = native_trainer(model, FakeNativeClient())
                trainer.args = SimpleNamespace(
                    vllm_importance_sampling_correction=True, per_device_train_batch_size=4,
                    gradient_accumulation_steps=32, generation_batch_size=32,
                    num_generations=8, steps_per_generation=8, num_iterations=1,
                )
                setattr(trainer.args, field, wrong)
                with self.assertRaises(ValueError):
                    NativePolicySync(
                        trainer, expected_schema=schema, audit_dir=directory,
                        fixed_rows=[{"prompt_ids": [1], "completion_ids": [2]}],
                    )


class PackedPolicyTests(unittest.TestCase):
    def fixture(self, tied=True):
        model, schema = qwen_fixture(tied=tied)
        hasher = PackedPolicyHasher(model.config, schema)
        state = model.state_dict()
        for name in hasher.canonical_order:
            hasher.consume(name, state[name])
        return model, state, hasher

    def test_qkv_gateup_bias_norm_and_tied_head_match_independent_loader(self):
        _, state, hasher = self.fixture()
        actual = checksums(fake_resident_weights(state))
        result = hasher.verify(actual)
        self.assertTrue(result["passed"])
        self.assertEqual(result["canonical_tensor_count"], 15)
        self.assertEqual(result["resident_parameter_count"], 9)
        self.assertNotIn(CHECKSUM_PREFIX + "lm_head.weight", actual)

    def test_target_qwen15b_has339_canonical_and198_resident_tensors(self):
        config = SimpleNamespace(model_type="qwen2", hidden_size=1536, vocab_size=151936,
                                 intermediate_size=8960, num_attention_heads=12,
                                 num_key_value_heads=2, num_hidden_layers=28, tie_word_embeddings=True)
        shapes, groups = qwen2_layout(config)
        self.assertEqual(len(shapes), 339)
        self.assertEqual(len(groups), 198)
        self.assertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], (1536, 1536))
        self.assertEqual(shapes["model.layers.0.self_attn.k_proj.bias"], (256,))

    def test_untied_output_head_requires_its_own_resident_checksum(self):
        _, state, hasher = self.fixture(tied=False)
        actual = checksums(fake_resident_weights(state, tied=False))
        self.assertTrue(hasher.verify(actual)["passed"])
        self.assertIn(CHECKSUM_PREFIX + "lm_head.weight", actual)

    def test_wrong_packed_bias_order_and_stale_weight_are_rejected(self):
        _, state, hasher = self.fixture()
        packed = fake_resident_weights(state)
        name = "model.layers.0.self_attn.qkv_proj.bias"
        packed[name] = packed[name].flip(0)
        with self.assertRaisesRegex(ValueError, "different="):
            hasher.verify(checksums(packed))
        packed = fake_resident_weights(state)
        packed["model.layers.0.mlp.down_proj.weight"].zero_()
        with self.assertRaisesRegex(ValueError, "different="):
            hasher.verify(checksums(packed))

    def test_missing_extra_or_second_rank_resident_keys_are_rejected(self):
        _, state, hasher = self.fixture()
        original = checksums(fake_resident_weights(state))
        missing = dict(original)
        missing.pop(next(iter(missing)))
        extra = {**original, CHECKSUM_PREFIX + "unaccounted.weight": "0" * 64}
        rank = {key.replace("tp0:", "tp1:"): value for key, value in original.items()}
        for actual in (missing, extra, rank):
            with self.assertRaisesRegex(ValueError, "Resident policy does not match"):
                hasher.verify(actual)

    def test_malformed_stream_tied_disagreement_and_padding_fail_closed(self):
        model, schema = qwen_fixture()
        hasher = PackedPolicyHasher(model.config, schema)
        first = hasher.canonical_order[0]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            hasher.expected_checksums()
        hasher.consume(first, model.state_dict()[first])
        with self.assertRaisesRegex(ValueError, "Out-of-order or duplicate"):
            hasher.consume(first, model.state_dict()[first])
        hasher = PackedPolicyHasher(model.config, schema)
        for name in hasher.canonical_order:
            value = model.state_dict()[name]
            hasher.consume(name, value + 1 if name == "lm_head.weight" else value)
        with self.assertRaisesRegex(ValueError, "Tied embedding"):
            hasher.expected_checksums()
        model.config.vocab_size = 65
        with self.assertRaisesRegex(ValueError, "Vocabulary padding"):
            PackedPolicyHasher(model.config, schema)


if __name__ == "__main__":
    unittest.main()
