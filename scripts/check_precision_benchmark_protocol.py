"""CPU-only independent review of the copied GeoRA benchmark protocol."""

import hashlib
import json
from pathlib import Path
import sys

REPO = Path('/Users/tianbaochen/geora-repro')
sys.path[:0] = [str(REPO), str(REPO/'scripts')]
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from geora_layers import GeoRALinear, iter_target_linears
from geora_check_utils import move_with_precision
from grpo_rollout import score_completion_logps
from grpo_math import grpo_loss
import precision_benchmark_helpers as helper

torch.set_num_threads(1)


def small_model():
    torch.manual_seed(20261011)
    config = Qwen2Config(
        vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, attention_dropout=0.0,
        tie_word_embeddings=False, pad_token_id=0, eos_token_id=2,
    )
    config._attn_implementation='eager'
    model = Qwen2ForCausalLM(config)
    model.requires_grad_(False)
    for name, projection in list(iter_target_linears(model)):
        parent_name, attr = name.rsplit('.',1)
        A0 = torch.randn(2,projection.in_features)*0.05
        B0 = torch.randn(projection.out_features,2)*0.05
        replacement = GeoRALinear(projection,A0,B0,scaling=2.0,forward_mode='difference')
        setattr(model.get_submodule(parent_name),attr,replacement)
    for p in model.parameters():
        if not p.requires_grad:
            p.data = p.data.bfloat16()
    return model


def digest_parameters(model, trainable):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad == trainable:
            h.update(name.encode())
            # Frozen values are mathematically identical under exact upcast.
            h.update(p.detach().float().numpy().tobytes())
    return h.hexdigest()


ids=torch.tensor([
    [0,0,5,6,7,8,9,2,0,0],
    [0,5,6,7,8,9,10,11,2,0],
    [5,6,7,8,9,10,11,12,13,2],
    [5,6,7,8,9,10,14,15,16,2],
])
mask=(ids!=0).long()
completion_mask=mask[:,6:].bool()
rollout=dict(input_ids=ids,attention_mask=mask,completion_mask=completion_mask,
             prompt_width=6,max_new_tokens=4,generated_steps=4,fixed_width=10,temperature=1.0)
advantages=torch.tensor([-1.,1.,-1.,1.])
checks=[]
cases=[]


def require(name,condition,**measurements):
    checks.append(dict(name=name,passed=bool(condition),**measurements))
    print('PASS' if condition else 'FAIL',name,flush=True)
    if not condition:
        raise AssertionError(name)


for label,dtype in [('BF16',torch.bfloat16),('FP32',torch.float32)]:
    model=small_model()
    move_with_precision(model,'cpu',dtype)
    model.train()
    model.config.use_cache=False
    initial={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    frozen_hash=digest_parameters(model,False)
    adapter_hash=digest_parameters(model,True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-6,betas=(.9,.999),eps=1e-8,
                                 weight_decay=0.,foreach=False)
    old=score_completion_logps(model,rollout,'cpu',dtype,gradient_enabled=False)
    reference=old.clone()
    trials=[]
    original_trial=helper.train_trial
    expected_parameters=None
    expected_optimizer=None

    def observe(*args,**kwargs):
        require(label+f'_trial_{len(trials)}_initial_AB',all(torch.equal(p,initial[name])
            for name,p in model.named_parameters() if p.requires_grad))
        require(label+f'_trial_{len(trials)}_grad_none',all(p.grad is None for p in parameters))
        require(label+f'_trial_{len(trials)}_zero_moments_and_step',all(
            s['step'].item()==0 and torch.count_nonzero(s['exp_avg']).item()==0
            and torch.count_nonzero(s['exp_avg_sq']).item()==0 for s in optimizer.state.values()))
        record=original_trial(*args,**kwargs)
        require(label+f'_trial_{len(trials)}_first_step',all(s['step'].item()==1
            for s in optimizer.state.values()))
        require(label+f'_trial_{len(trials)}_nonzero_gradient',record['gradient_norm_before_clipping']>0)
        require(label+f'_trial_{len(trials)}_timing_partition',abs(record['total_seconds']-
            record['forward_and_loss_seconds']-record['backward_clip_step_seconds'])<1e-10)
        trials.append({name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad})
        return record

    helper.train_trial=observe
    result=helper.benchmark(model,optimizer,score_completion_logps,grpo_loss,rollout,'cpu',dtype,
                            old,reference,advantages,warmup=2,repeats=3)
    helper.train_trial=original_trial
    require(label+'_five_trials_identical_updates',all(all(torch.equal(trial[name],trials[0][name])
        for name in trial) for trial in trials))
    require(label+'_adapter_values_changed',any(not torch.equal(trials[-1][name],initial[name]) for name in initial))
    require(label+'_frozen_values_unchanged',digest_parameters(model,False)==frozen_hash)
    require(label+'_measured_count',len(result['measurements'])==3)
    require(label+'_FP32_adapter_and_optimizer',all(p.dtype==torch.float32 for p in parameters)
        and all(s['exp_avg'].dtype==torch.float32 and s['exp_avg_sq'].dtype==torch.float32 for s in optimizer.state.values()))

    # A freshly allocated AdamW first step must produce the same update as the
    # warmed, in-place zeroed state used by the benchmark.
    fresh_model=small_model()
    move_with_precision(fresh_model,'cpu',dtype)
    fresh_model.train()
    fresh_parameters=[p for p in fresh_model.parameters() if p.requires_grad]
    fresh_optimizer=torch.optim.AdamW(fresh_parameters,lr=1e-6,betas=(.9,.999),eps=1e-8,
                                       weight_decay=0.,foreach=False)
    fresh_record=original_trial(fresh_model,fresh_optimizer,score_completion_logps,grpo_loss,
        rollout,'cpu',dtype,old,reference,advantages)
    require(label+'_warmed_reset_equals_fresh_AdamW_first_step',all(torch.equal(p,trials[-1][name])
        for name,p in fresh_model.named_parameters() if p.requires_grad))
    cases.append(dict(label=label,frozen_hash=frozen_hash,initial_adapter_hash=adapter_hash,
                      input_shape=list(ids.shape),advantages=advantages.tolist(),benchmark=result))

require('BF16_FP32_exact_same_frozen_numeric_values',cases[0]['frozen_hash']==cases[1]['frozen_hash'])
require('BF16_FP32_same_initial_adapter',cases[0]['initial_adapter_hash']==cases[1]['initial_adapter_hash'])
require('BF16_FP32_same_input_and_advantages',cases[0]['input_shape']==cases[1]['input_shape']
    and cases[0]['advantages']==cases[1]['advantages'])
output=Path('/tmp/precision_benchmark_protocol.json')
output.write_text(json.dumps(dict(status='passed',scope='CPU small Qwen protocol test; no GPU performance claim',
                                  checks=checks,cases=cases),indent=2)+'\n')
print('Report:',output,flush=True)
