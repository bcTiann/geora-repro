"""Exercise the full validation logic on a small Qwen model, without downloads."""

import copy
from pathlib import Path
import sys

import tempfile
project_directory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_directory))
sys.path.insert(0, str(project_directory / 'scripts'))
temporary_results = tempfile.TemporaryDirectory(prefix='geora-validation-')
results_directory = Path(temporary_results.name)
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from geora_layers import install_geora
from geora_check_utils import CheckReport, move_with_precision
from check_geora_training import validate_model_and_update

torch.set_num_threads(2)
for dtype in (torch.float32, torch.bfloat16):
    torch.manual_seed(0)
    configuration = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    )
    configuration._attn_implementation = 'eager'
    original = Qwen2ForCausalLM(configuration).float().eval()
    original.requires_grad_(False)
    def fresh():
        return copy.deepcopy(original)
    reference = fresh()
    model = fresh()
    manifest = install_geora(model, rank=2, alpha=4, rho=0.2)
    move_with_precision(reference, 'cpu', dtype)
    move_with_precision(model, 'cpu', dtype)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    inputs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    labels = ids.clone()
    labels[:, :3] = -100
    directory = results_directory / str(dtype)
    report = CheckReport(directory, 'checks.json')
    validate_model_and_update(
        model, reference, manifest, fresh, inputs, labels, directory,
        report, 'cpu', dtype,
    )
    report.finish()
    print('END-TO-END SMALL MODEL PASSED:', dtype, len(report.data['checks']))

# The layout check must reject accidentally casting trainable adapters to BF16.
bad = fresh()
bad_manifest = install_geora(bad, rank=2, alpha=4, rho=0.2)
bad.to(dtype=torch.bfloat16)
bad_report = CheckReport(results_directory / 'bad_dtype', 'checks.json')
try:
    validate_model_and_update(
        bad, reference, bad_manifest, fresh, inputs, labels,
        bad_report.path.parent, bad_report, 'cpu', torch.bfloat16,
    )
except RuntimeError as error:
    assert 'parameter_precision' in str(error)
    assert bad_report.data['status'] == 'failed'
    print('EXPECTED BAD ADAPTER DTYPE REJECTED')
else:
    raise AssertionError('Bad adapter precision was not detected')

temporary_results.cleanup()
