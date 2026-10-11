"""Independently recompute all saved sampled-token probabilities with NumPy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=json.loads(args.report.read_text())
    assert source['status']=='passed'
    result=dict(status='running',device='CPU',implementation='NumPy FP64; no model forward',
        source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=[])
    for case in source['cases']:
        path=Path(case['snapshot']['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['snapshot']['sha256']
        values=load_file(path)
        residual=0.0
        ratio_residual=0.0
        max_logit_difference=0.0
        for item in case['sampled_token_probabilities']:
            key=f"row{item['row']}.step{item['completion_step']}"
            logps={}
            for label in ('cached','fixed'):
                logits=values[label+'.'+key].astype(np.float64)
                shifted=logits-logits.max()
                probability=float(np.exp(shifted[item['token_id']])/np.exp(shifted).sum())
                logp=float(shifted[item['token_id']]-np.log(np.exp(shifted).sum()))
                residual=max(residual,abs(probability-item[label+'_probability']))
                logps[label]=logp
            relative=math.expm1(logps['fixed']-logps['cached'])
            ratio_residual=max(ratio_residual,abs(relative-item['relative_probability_difference']))
            delta=values['fixed.'+key].astype(np.float64)-values['cached.'+key].astype(np.float64)
            max_logit_difference=max(max_logit_difference,float(np.abs(delta).max()))
        assert residual<1e-12 and ratio_residual<1e-12
        result['cases'].append(dict(label=case['label'],valid_token_count=len(case['sampled_token_probabilities']),
            full_vocab_probabilities_checked=2*len(case['sampled_token_probabilities']),
            snapshot_sha256=case['snapshot']['sha256'],max_probability_absolute_residual=residual,
            max_ratio_absolute_residual=ratio_residual,max_full_vocab_logit_difference=max_logit_difference))
    result['status']='passed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Independent saved-logits NumPy verification passed; no GPU or model forward.')


if __name__=='__main__':
    main()
