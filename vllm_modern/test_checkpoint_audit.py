"""CPU tests of the resume observer, including corrupted restoration paths."""

import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn

from checkpoint_audit import (
    ResumeAuditMixin, checkpoint_inventory, load_torch, state_digest, verify_loaded_policy,
)


class GeoRALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor([[0.5, -0.25]]))
        self.B = nn.Parameter(torch.tensor([[0.75], [0.5]]))
        self.register_buffer("W_res", torch.eye(2, dtype=torch.bfloat16))
        self.register_buffer("bias", torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
        self.scaling = 0.5


class ParentTrainer:
    """Fixture only: the real stock hook runs in the subsequent GPU acceptance."""

    def _load_from_checkpoint(self, checkpoint, model=None):
        model = self.model if model is None else model
        tensors = load_file(str(Path(checkpoint) / "geora.safetensors"))
        model.load_state_dict(tensors)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is not None:
            self.optimizer.load_state_dict(load_torch(Path(checkpoint) / "optimizer.pt"))
            self.lr_scheduler.load_state_dict(load_torch(Path(checkpoint) / "scheduler.pt"))
        if getattr(self, "corrupt_optimizer", False):
            self.optimizer.param_groups[0]["lr"] *= 2

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return
        saved = load_torch(Path(checkpoint) / "rng_state.pth")
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.random.set_rng_state(saved["cpu"])
        if getattr(self, "corrupt_rng", False):
            torch.rand(1)

    def _init_training_state(self, max_steps, num_update_steps_per_epoch, num_train_epochs,
                             resume_from_checkpoint, trial):
        if resume_from_checkpoint is None:
            self.state = SimpleNamespace(global_step=0)
            return (0, 0)
        step = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())["global_step"]
        self.state = SimpleNamespace(global_step=step)
        return (step // num_update_steps_per_epoch,
                (step % num_update_steps_per_epoch) * self.args.gradient_accumulation_steps)


class ObservedTrainer(ResumeAuditMixin, ParentTrainer):
    def __init__(self, directory):
        self.model = nn.Sequential(GeoRALayer())
        self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        self.args = SimpleNamespace(ignore_data_skip=False, gradient_accumulation_steps=32,
                                    steps_per_generation=8)
        self.num_iterations = 1
        self.resume_audit_dir = Path(directory) / "audit"
        self.resume_audit_variant = "geora_mixed"


def make_checkpoint(directory):
    trainer = ObservedTrainer(directory)
    loss = sum(value.square().sum() for value in trainer.model.parameters())
    loss.backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    checkpoint = Path(directory) / "checkpoint-1"
    checkpoint.mkdir()
    save_file(trainer.model.state_dict(), str(checkpoint / "geora.safetensors"))
    (checkpoint / "geora_config.json").write_text('{"format_version": 1}')
    (checkpoint / "trainer_state.json").write_text('{"global_step": 1, "max_steps": 2}')
    torch.save(trainer.optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(trainer.lr_scheduler.state_dict(), checkpoint / "scheduler.pt")
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state()}, checkpoint / "rng_state.pth")
    return checkpoint


class CheckpointAuditTests(unittest.TestCase):
    def test_digest_is_order_stable_typed_and_byte_sensitive(self):
        a = {"numpy": np.array([1, 2], dtype=np.uint32), 5: [torch.tensor([1.0]), (True, None)]}
        b = {5: [torch.tensor([1.0]), (True, None)], "numpy": np.array([1, 2], dtype=np.uint32)}
        self.assertEqual(state_digest(a), state_digest(b))
        self.assertNotEqual(state_digest(torch.tensor([1.0])), state_digest(torch.tensor([1.0], dtype=torch.bfloat16)))
        self.assertNotEqual(state_digest([1, 2]), state_digest((1, 2)))
        b[5][0][0] = 2.0
        self.assertNotEqual(state_digest(a), state_digest(b))

    def test_real_cpu_optimizer_scheduler_rng_and_policy_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            trainer = ObservedTrainer(directory)
            original_parameters = [id(value) for value in trainer.model.parameters()]
            trainer._load_from_checkpoint(checkpoint)
            trainer._load_optimizer_and_scheduler(checkpoint)
            trainer._init_training_state(2, 460, 1, checkpoint, None)
            random.random()
            np.random.rand()
            torch.rand(1)
            trainer._load_rng_state(checkpoint)
            summary = trainer.resume_audit_summary()
            self.assertTrue(summary["passed"])
            self.assertEqual({row["kind"] for row in summary["records"]},
                             {"policy", "optimizer", "scheduler", "rng", "training_state"})
            self.assertEqual(original_parameters, [id(value) for value in trainer.model.parameters()])
            state = next(row for row in summary["records"] if row["kind"] == "training_state")
            self.assertEqual(state["microbatches_skipped"], 32)
            self.assertEqual(checkpoint_inventory(checkpoint, "geora_mixed")["global_step"], 1)

    def test_optimizer_that_was_not_really_restored_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            trainer = ObservedTrainer(directory)
            trainer.corrupt_optimizer = True
            with self.assertRaisesRegex(ValueError, "Restored optimizer differs"):
                trainer._load_optimizer_and_scheduler(checkpoint)

    def test_rng_drift_after_load_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            trainer = ObservedTrainer(directory)
            trainer.corrupt_rng = True
            with self.assertRaisesRegex(ValueError, "Restored learner RNG differs"):
                trainer._load_rng_state(checkpoint)

    def test_missing_policy_tensor_or_different_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            trainer = ObservedTrainer(directory)
            with self.assertRaisesRegex(ValueError, "differs bitwise"):
                verify_loaded_policy(trainer.model, checkpoint, "geora_mixed")
            tensors = load_file(str(checkpoint / "geora.safetensors"))
            del tensors["0.bias"]
            save_file(tensors, str(checkpoint / "geora.safetensors"))
            with self.assertRaisesRegex(ValueError, "tensor coverage"):
                verify_loaded_policy(trainer.model, checkpoint, "geora_mixed")

    def test_lora_uses_peft_saved_state_contract(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM

        model = get_peft_model(
            LlamaForCausalLM(LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, vocab_size=32)),
            LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            self.assertEqual(verify_loaded_policy(model, directory, "lora")["tensor_count"], 4)
            with torch.no_grad():
                next(value for value in model.parameters() if value.requires_grad).add_(0.01)
            with self.assertRaisesRegex(ValueError, "differs bitwise"):
                verify_loaded_policy(model, directory, "lora")

    def test_incomplete_checkpoint_and_incomplete_audit_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete checkpoint: optimizer"):
                checkpoint_inventory(checkpoint, "geora_mixed")
            trainer = ObservedTrainer(directory)
            with self.assertRaisesRegex(ValueError, "Incomplete resume audit"):
                trainer.resume_audit_summary()

    def test_resume_requires_exhausted_grpo_buffer_and_data_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            trainer = ObservedTrainer(directory)
            trainer.args.gradient_accumulation_steps = 31
            with self.assertRaisesRegex(ValueError, "unsaved GRPO buffered"):
                trainer._init_training_state(2, 460, 1, checkpoint, None)
            trainer.args.ignore_data_skip = True
            with self.assertRaisesRegex(ValueError, "actual data skip"):
                trainer._init_training_state(2, 460, 1, checkpoint, None)


if __name__ == "__main__":
    unittest.main()
