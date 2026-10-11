"""Independently verify saved logits and vector identities with NumPy FP64."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/local_error_propagation/20261011.json')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/local_error_propagation/numpy_verification.json')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report['status'] == 'passed'
    result = dict(status='running', device='CPU', implementation='independent NumPy FP64', cases=[],
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for case in report['cases']:
        path = Path(case['snapshot']['path'])
        if not path.is_absolute():
            path = ROOT / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['snapshot']['sha256']
        tensors = load_file(path)
        record = dict(label=case['label'], snapshot_sha256=case['snapshot']['sha256'])
        for label, key in [('cached','cached'),('fixed','fixed_no_cache')]:
            logits = tensors[key + '.target.lm_head'].astype(np.float64)
            weights = np.exp(logits - logits.max())
            probability = float(weights[case['token_id']] / weights.sum())
            residual = abs(probability - case['endpoint'][label]['selected_probability'])
            assert residual < 1e-12
            record[label + '_probability'] = probability
            record[label + '_probability_absolute_residual'] = residual
        for item in case['interventions']:
            index = item['layer']
            cached = tensors[f'cached.target.layer.{index}.output'].astype(np.float64)
            fixed = tensors[f'fixed_no_cache.target.layer.{index}.output'].astype(np.float64)
            aligned = tensors[f'intervention.{index}.aligned_history_output'][case['target_prefix_column']].astype(np.float64)
            propagated = tensors[f'intervention.{index}.propagated_prior_difference']
            fresh = tensors[f'intervention.{index}.same_input_path_difference']
            assert np.array_equal(fixed-aligned, propagated)
            assert np.array_equal(aligned-cached, fresh)
            assert np.array_equal(fixed-cached, propagated+fresh)
        record['exact_target_vector_decompositions_checked'] = len(case['interventions'])
        for control in case.get('layer0_mlp_same_input_shape_controls', []):
            name = control['module'].split('.')[-1]
            assert np.array_equal(tensors[f'layer0_shape_control.{name}.one'], tensors['cached.target.'+control['module']])
            assert np.array_equal(tensors[f'layer0_shape_control.{name}.full'], tensors['fixed_no_cache.target.'+control['module']])
        record['layer0_control_native_recoveries_checked'] = len(case.get('layer0_mlp_same_input_shape_controls', []))
        result['cases'].append(record)
    result['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Independent NumPy verification passed; no model forward or GPU use.')


if __name__ == '__main__':
    main()
