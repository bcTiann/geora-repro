"""Exercise real tiny Qwen generation/cache/padding APIs without downloads."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))

from geora_layers import export_adapter_state, install_geora, load_geora_state
from geora_check_utils import CheckReport, forward_context, move_with_precision
from geora_generation_checks import check_generation_paths, left_padded_inputs


def check_supervised_padding(model, reference, report, dtype):
    """Construct known prompt/answer boundaries, then count shifted CE targets.

    Three records have prompt/answer lengths (2, 2), (3, 1), and (4, 2).
    The first two rows need two left pads to match the third row's length six.
    All pad/prompt labels are ignored; five answer tokens enter the shifted loss.
    """
    prompts = [[3, 4], [7, 8, 9]]
    answers = [[5, 63], [10]]
    full_tokens = [prompt + answer for prompt, answer in zip(prompts, answers)]
    # The longer third record forces actual left padding in the first two.
    prompts.append([11, 12, 13, 14])
    answers.append([15, 63])
    full_tokens.append(prompts[-1] + answers[-1])
    inputs = left_padded_inputs(full_tokens, pad_token_id=0, device="cpu")
    labels = torch.full_like(inputs["input_ids"], -100)
    for row, answer in enumerate(answers):
        labels[row, -len(answer):] = torch.tensor(answer)
    shifted_labels = labels[:, 1:]
    answer_count = sum(map(len, answers))
    report.require("padding_answer_targets_counted_after_shift",
                   (shifted_labels != -100).sum().item() == answer_count,
                   counted_targets=(shifted_labels != -100).sum().item(), expected_targets=answer_count)
    report.require("padding_pad_and_prompt_labels_ignored",
                   torch.all(labels[inputs["attention_mask"] == 0] == -100).item()
                   and all(torch.all(labels[row, :-len(answer)] == -100).item()
                           for row, answer in enumerate(answers)))
    with torch.inference_mode(), forward_context("cpu", dtype):
        actual_logits = model(**inputs, use_cache=False).logits.float()
        reference_logits = reference(**inputs, use_cache=False).logits.float()
    actual_loss = torch.nn.functional.cross_entropy(
        actual_logits[:, :-1].reshape(-1, actual_logits.shape[-1]), shifted_labels.reshape(-1)
    )
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits[:, :-1].reshape(-1, reference_logits.shape[-1]), shifted_labels.reshape(-1)
    )
    # Independent explicit next-token indexing verifies the shifted loss target.
    individual_losses = []
    for row, answer in enumerate(answers):
        start = inputs["input_ids"].shape[1] - len(answer)
        for offset, target in enumerate(answer):
            individual_losses.append(-actual_logits[row, start + offset - 1].log_softmax(-1)[target])
    manual_loss = torch.stack(individual_losses).mean()
    report.require("padding_shifted_loss_matches_explicit_answer_tokens",
                   torch.allclose(actual_loss, manual_loss, rtol=1e-6, atol=1e-6),
                   loss=actual_loss.item(), explicit_loss=manual_loss.item())
    report.require("padding_native_reference_answer_loss_exact",
                   torch.equal(actual_loss, reference_loss), loss=actual_loss.item())


def run(output):
    torch.set_num_threads(2)
    summary = {"status": "running", "torch_version": torch.__version__, "device": "cpu", "models": {}}
    with tempfile.TemporaryDirectory(prefix="geora-generation-") as temporary:
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(17)
            configuration = Qwen2Config(
                vocab_size=64, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=64, attention_dropout=0.0,
                bos_token_id=1, pad_token_id=0, eos_token_id=63,
            )
            configuration._attn_implementation = "eager"
            original = Qwen2ForCausalLM(configuration).float().eval()
            original.requires_grad_(False)
            initialized = copy.deepcopy(original)
            manifest = install_geora(initialized, rank=2, alpha=4, rho=0.2)
            state = export_adapter_state(initialized)
            manifest.pop("forward_mode", None)
            model = copy.deepcopy(original)
            load_geora_state(model, state, manifest, forward_mode="difference")
            reference = copy.deepcopy(original)
            move_with_precision(model, "cpu", dtype)
            move_with_precision(reference, "cpu", dtype)
            report = CheckReport(Path(temporary) / str(dtype), "checks.json")
            check_generation_paths(
                model, reference, [[1, 4, 5], [1, 7, 8, 9, 10]],
                pad_token_id=0, eos_token_id=63, device="cpu", frozen_dtype=dtype,
                report=report, max_new_tokens=4,
            )
            check_supervised_padding(model, reference, report, dtype)
            report.finish()
            summary["models"][str(dtype)] = report.data
    summary.update(status="passed", scope="Tiny CPU forward/generation/label mechanics only; no optimization.")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print("Report:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
