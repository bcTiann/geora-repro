"""Draw static scientific summaries from persisted measurements only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=ROOT/'reports/local_error_propagation/20261011.json')
parser.add_argument('--output', type=Path, default=ROOT/'figures/local_error_propagation/layer_propagation.png')
args = parser.parse_args()
report = json.loads(args.report.read_text())
assert report['status'] == 'passed'
bf16_this, bf16_end, fp32_this, fp32_end = report['cases']
colors = {'This':'#2563eb', 'End':'#dc6b19'}
fig, axes = plt.subplots(2,2, figsize=(12,8), layout='constrained')
for dtype, cases, ax in [('BF16',[bf16_this,bf16_end],axes[0,0]),('FP32',[fp32_this,fp32_end],axes[0,1])]:
    for label, case in zip(('This','End'),cases):
        x = [item['layer'] for item in case['layers']]
        y = [100*item['output']['relative_l2'] for item in case['layers']]
        ax.plot(x,y,label=label,color=colors[label],lw=2)
    ax.set_title(dtype + ': same layer-output vector (1536 dimensions)', fontsize=11)
    ax.set_ylabel('Relative L2 difference (%)')
    ax.set_xlabel('Decoder layer index (0-27)')
    ax.grid(alpha=.2)
    ax.legend()
    ax.set_xlim(0,27)
axes[0,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
items = bf16_this['interventions']
for key, label, color, style in (
    ('total_output_difference_l2','Total difference','#334155','-'),
    ('propagated_prior_difference_l2','Response to prior-state difference','#2563eb','--'),
    ('same_input_path_difference_l2','Remaining path difference at equal inputs','#16a34a','-')):
    axes[1,0].plot([x['layer'] for x in items],[x[key] for x in items],label=label,color=color,lw=2,ls=style)
axes[1,0].set_title('This: components of each layer output difference', fontsize=11)
axes[1,0].set_ylabel('L2 length of target-vector difference')
axes[1,0].set_xlabel('Layer whose input was aligned (0-27)')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(alpha=.2)
axes[1,0].set_xlim(0,27)
for label, case in [('This',bf16_this),('End',bf16_end)]:
    axes[1,1].plot([x['layer'] for x in case['interventions']],
                  [100*x['remaining_endpoint']['probability_relative_difference'] for x in case['interventions']],
                  label=label,color=colors[label],lw=2)
    axes[1,1].axhline(100*case['endpoint']['probability_relative_difference'],color=colors[label],ls=':',alpha=.7)
axes[1,1].axhline(0,color='#64748b',lw=.8)
axes[1,1].set_title('Final probability after input alignment\nIndependent control at each layer', fontsize=11)
axes[1,1].set_ylabel('(Fixed / cached token probability - 1) (%)')
axes[1,1].set_xlabel('Independently aligned layer (0-27)')
axes[1,1].legend()
axes[1,1].grid(alpha=.2)
axes[1,1].set_xlim(0,27)
fig.suptitle('Frozen Qwen2.5-1.5B on Metal: same saved tokens, no training',fontsize=14)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output,dpi=170)
plt.close(fig)

print('Figure saved:', args.output)
