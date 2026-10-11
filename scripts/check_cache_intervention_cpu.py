"""CPU prechecks for actual attention capture and target-only interventions."""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from cache_attention_intervention import AttentionCapture, inspect_interventions, product_controls
from cache_layer_probe import trace_path
from geora_check_utils import CheckReport


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    report=CheckReport(args.output_dir,'cache_intervention_cpu.json')
    report.data.update(stage='tiny_attention_intervention_precheck',cases=[],optimizer_steps=0)
    start=time.perf_counter()
    original=modeling_qwen2.eager_attention_forward
    try:
        for dtype in (torch.float32,torch.bfloat16):
            torch.manual_seed(18)
            model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                attention_dropout=0,eos_token_id=2,pad_token_id=0))
            model.config._attn_implementation='eager'
            model.requires_grad_(False).eval().to(dtype=dtype)
            before={n:p.clone() for n,p in model.named_parameters()}
            prompts=[[3,4,5]]*4+[[6,7,8,9,10]]*4
            case=dict(completion_ids=[[7,8,9,2,0],[7,2,0,0,0]]+[[7,8,9,10,11]]*6,
                      completion_lengths=[4,2]+[5]*6)
            inspect_interventions(model,prompts,case,0,2,0,[2],dtype,'cpu',report,str(dtype))
            report.require(str(dtype)+'_all_parameter_values_exact', all(
                torch.equal(before[n],p) for n,p in model.named_parameters()))
            report.require(str(dtype)+'_hooks_and_wrapper_restored',
                modeling_qwen2.eager_attention_forward is original and
                all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        # Independent hand-worked example: P=[1/4,3/4], deltaV at first key=[4,0].
        p=torch.tensor([[[[.25,.75]]]],dtype=torch.float64)
        v=torch.tensor([[[[2.,0.],[0.,4.]]]],dtype=torch.float64)
        dv=torch.tensor([[[[4.,0.],[0.,0.]]]],dtype=torch.float64)
        delta=(p@(v+dv)-p@v).reshape(-1)
        report.require('synthetic_P_deltaV_matches_hand_result',torch.equal(delta,torch.tensor([1.,0.])))
        # Exercise the control helper with that hand-worked nonzero V change.
        full_p=p.repeat(1,1,2,1)
        before_output=(full_p@v).transpose(1,2).contiguous()
        after_output=(p@(v+dv)).transpose(1,2).contiguous()
        fixed=SimpleNamespace(native={'P':full_p,'V':v,'output':before_output})
        cached=SimpleNamespace(native={'P':p,'V':v+dv,'output':after_output})
        control, products=product_controls(cached,fixed,0,1,2,torch.tensor([True,True]),
                                           torch.float64,'cpu')
        report.require('control_nonzero_V_swap_matches_hand_result',torch.equal(
            products['full_cached_V']-products['full_original'],torch.tensor([[1.,0.]])))
        report.require('control_fp64_linear_identity',control['fp64_P_deltaV_identity_residual']==0)
        report.require('control_combined_recovers_cached_output',
                       control['combined_difference_to_native_cached']['exact'])
        zero_cached=SimpleNamespace(native={'P':p,'V':v,'output':before_output[:,-1:]})
        zero, _=product_controls(zero_cached,fixed,0,1,2,torch.tensor([True,True]),torch.float64,'cpu')
        report.require('control_zero_V_change_zero_effect',zero['historical_V_effect']['exact'])
        # A deliberate exception cannot leave the process attention wrapper changed.
        try:
            with AttentionCapture(model,0,0,1,'fixed_no_cache').installed():
                raise RuntimeError('intentional restoration probe')
        except RuntimeError:
            pass
        report.require('exception_restores_hooks_and_wrapper',
                       modeling_qwen2.eager_attention_forward is original and
                       all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        report.data['elapsed_seconds']=time.perf_counter()-start
        report.finish()
    except Exception as error:
        report.data.update(status='failed',error=str(error));report.write();raise


if __name__=='__main__':main()
