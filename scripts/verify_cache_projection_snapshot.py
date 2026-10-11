"""Independent CPU analysis of the exact BF16 tensors saved on MI250X."""
import hashlib
import json
import math
from pathlib import Path
import argparse

import numpy as np
import torch
from safetensors.torch import load_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
arguments = parser.parse_args()
report = json.loads(arguments.report.read_text())
case = report['case']
snapshot_path = arguments.snapshot
assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == case['snapshot']['sha256']
tensors = load_file(snapshot_path)
row = case['row']
prefix = case['prefix_width']
inputs = tensors['fixed_inputs'].double().numpy()
weight = tensors['weight'].double().numpy()
bias = tensors['bias'].double().numpy()
rechecks = []
for item in case['mismatches']:
    column, channel = item['padded_column'], item['channel']
    products = inputs[row, column] * weight[channel]
    dot_numpy = float(np.dot(inputs[row, column], weight[channel]))
    sum_fsum = math.fsum(map(float, products)) + float(bias[channel])
    oracle_delta = sum_fsum - item['fp64_dot_plus_bias']
    assert abs(oracle_delta) < 1e-12
    assert tensors['cached_outputs'][row, column, channel].item() == item['cached_V']
    assert tensors['fixed_outputs'][row, column, channel].item() == item['fixed_V']
    rechecks.append(dict(padded_column=column, channel=channel, numpy_dot_plus_bias=dot_numpy+float(bias[channel]),
                         fsum_dot_plus_bias=sum_fsum, delta_to_GPU_recorded_CPU_FP64=oracle_delta))
cached = tensors['cached_outputs']
retrieved = tensors['cache_retrieved_V']
heads, head_dim = retrieved.shape[1], retrieved.shape[-1]
expected = cached.view(cached.shape[0], cached.shape[1], heads, head_dim).transpose(1, 2)
assert torch.equal(expected, retrieved)
valid = tensors['attention_mask'][row, :prefix].bool()
unequal = torch.nonzero((cached[row] != tensors['fixed_outputs'][row, :prefix]) & valid[:, None]).tolist()
assert unequal == [[item['padded_column'], item['channel']] for item in case['mismatches']]
promoted_chunk = tensors['promoted_chunk_outputs'][row, valid].to(torch.bfloat16)
promoted_full = tensors['promoted_full_outputs'][row, :prefix][valid].to(torch.bfloat16)
result = dict(status='passed', snapshot_sha256=case['snapshot']['sha256'], coordinate_count=len(rechecks),
              cache_preserves_projection_exact=True, six_native_difference_coordinates_exact=True,
              independent_numpy_and_fsum_rechecks=rechecks,
              promoted_FP32_then_BF16_unequal_elements=(promoted_chunk != promoted_full).sum().item(),
              scope='Verify saved original tensors and six-coordinate analysis; no new model forward or GPU use')
arguments.output.parent.mkdir(parents=True, exist_ok=True)
arguments.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
