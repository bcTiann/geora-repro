"""Validate a streamed temporary checkpoint copy using CPU only.

The copy record must contain source SHA256 digests calculated while copying.
Check destination bytes, then every FP32 loaded parameter against raw BF16
checkpoint values. Run before spending a GPU allocation on this copy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geora_check_utils import fresh_fp32_model, pinned_configuration


def run(directory):
    torch.set_num_threads(2)
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    configuration, source = pinned_configuration()
    path = directory / "staging_record.json"
    record = json.loads(path.read_text())
    if Path(record["source"]).resolve() != source.resolve():
        raise RuntimeError("Staging copy source differs from the pinned checkpoint")
    started = time.perf_counter()
    weight_bytes = None
    for name, information in record["files"].items():
        data = (directory / name).read_bytes()
        if len(data) != information["bytes"] or hashlib.sha256(data).hexdigest() != information["source_sha256"]:
            raise RuntimeError("Destination bytes differ from copied source: " + name)
        if name == "model.safetensors":
            weight_bytes = data
    if weight_bytes is None:
        raise RuntimeError("Missing staged model.safetensors")
    print("All copied-file checksums match", flush=True)
    model_started = time.perf_counter()
    model = fresh_fp32_model(directory, disable_mmap=True)
    load_seconds = time.perf_counter() - model_started
    raw_weights = load(weight_bytes)
    parameters = dict(model.named_parameters())
    if not all(torch.equal(parameters[name].detach(), value.float()) for name, value in raw_weights.items()):
        raise RuntimeError("Loaded FP32 values differ from source checkpoint")
    record.update(status="validated", model_revision=configuration["model_revision"],
                  model_load_seconds=load_seconds, source_tensor_count=len(raw_weights),
                  all_source_tensors_exact=True, validation_seconds=time.perf_counter()-started,
                  loader_options={"disable_mmap": True, "HF_DEACTIVATE_ASYNC_LOAD": "1"})
    path.write_text(json.dumps(record, indent=2) + "\n")
    print("CPU staging validated:", load_seconds, "seconds load;", len(raw_weights), "exact tensors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    run(parser.parse_args().checkpoint_dir)
