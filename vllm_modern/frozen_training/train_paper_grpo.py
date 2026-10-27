"""LoRA / GeoRA 的单设备 GRPO 训练入口；训练循环由 TRL 管理。

先读 README.md。--dry-run 只打印配置，不加载模型或开始训练。
这里的 paper 表示对齐论文已公开的设置，不表示所有未公开细节已被复原。
"""

import argparse
import hashlib
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig
from answer_scoring import GSM8K_REWARD_VERSION, parse_numeric_answer

from geora_paper_utils import (
    ExperimentGRPOTrainer, GeoRAGRPOTrainer, format_math_example,
    geora_layers, grpo_boxed_reward, install_geora_layers,
    linear_target_names, load_geora_layers,
)


def write_json(path, value):
    """先写临时文件再替换，避免中断时留下只写了一半的 JSON。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def sha256_file(path):
    """分块计算文件摘要。相同摘要用于检查数据/代码是否被悄悄更换。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("lora", "geora_mixed"), required=True)
    parser.add_argument("--base-model", type=Path, required=True, help="本地 Qwen 基座目录")
    parser.add_argument("--train-data", type=Path, required=True, help="已划分好的 GSM8K train.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="本次方法/seed 独用的输出目录")
    parser.add_argument("--max-steps", type=int, required=True, help="目标总优化器步数；续训时不是额外步数")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="一次前向/反向处理的回答数")
    parser.add_argument("--generation-batch-size", type=int, default=8, help="一次生成的回答总数，须是 G 的倍数")
    parser.add_argument("--answers-per-update", type=int, default=128, help="每次更新的总回答数；论文 batch 单位未披露，此处显式按回答理解")
    parser.add_argument("--num-generations", type=int, default=8, help="G：同题回答数。正式对照保持 8")
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--hours", type=float, help="本次进程训练时长的软上限；完成当前更新再停，不替代公平比较的总步数")
    parser.add_argument("--resume", action="store_true", help="从此输出目录最近的完整 checkpoint 恢复")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda", help="ROCm 也写 cuda；cpu 仅供微型测试")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    positive = ("max_steps", "micro_batch_size", "generation_batch_size", "answers_per_update",
                "max_completion_length", "max_prompt_length", "rank", "alpha", "learning_rate", "save_steps")
    if any(getattr(options, key) <= 0 for key in positive) or options.num_generations < 2:
        parser.error("步数、长度、batch、rank、alpha、学习率应为正数，G 至少为 2。")
    if not 0 <= options.rho <= 1 or options.save_total_limit < 2:
        parser.error("rho 应在 [0, 1]，save-total-limit 至少为 2。")
    if options.hours is not None and options.hours <= 0:
        parser.error("hours 应为正数。")
    # 让生成组、微批次和优化器更新的边界对齐；不自行拆分同题的 reward 组。
    micro, generation, total = options.micro_batch_size, options.generation_batch_size, options.answers_per_update
    if generation % options.num_generations or generation % micro or total % generation:
        parser.error("generation-batch-size 须同时是 G 和 micro 的倍数；answers-per-update 须是 generation-batch-size 的倍数。")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("这版 GeoRA 存取适配限定单进程。请每个任务申请 1 个逻辑 GPU。")
    return options


def protocol_settings(options):
    """影响训练结果的约定。max_steps/时限/保存频率允许续训时改变。"""
    return {
        "variant": options.variant,
        "rank": options.rank, "alpha": options.alpha, "rho": options.rho,
        "target_modules": "all-linear-excluding-output-head",
        "fixed_dtype": "bfloat16", "adapter_dtype": "float32", "autocast": False,
        "svd": "exact_cpu_fp32", "reward": GSM8K_REWARD_VERSION,
        "numeric_comparison": "exact_fraction_decimal_no_tolerance",
        "answer_extraction": "last_boxed_no_rescue_after_malformed_command",
        "prompt": "step_by_step_numeric_boxed_v1",
        "learning_rate": options.learning_rate, "optimizer": "adamw_torch",
        "weight_decay": 0.0, "scheduler": "constant", "max_grad_norm": 1.0,
        "num_generations": options.num_generations,
        "answers_per_update": options.answers_per_update,
        "batch_128_unit_assumption": "responses_not_prompts",
        "micro_batch_size": options.micro_batch_size,
        "generation_batch_size": options.generation_batch_size,
        "max_prompt_length": options.max_prompt_length,
        "max_completion_length": options.max_completion_length,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0,
        "beta": 0.0, "num_iterations": 1, "loss_type": "grpo",
        "scale_rewards": "group", "epsilon": 0.2,
        "mask_truncated_completions": False,
        "gradient_checkpointing": True, "use_reentrant": False,
        "seed": options.seed, "data_seed": options.seed,
        "device_type": options.device,
    }


def prepare_dataset(path, tokenizer, max_prompt_length):
    """JSONL -> Dataset -> chat prompt；只保留实际 chat token 长度合格的题。

    load_dataset 的 train 是本次文件映射到的键名。
    map 返回 prompt/标准答案，remove_columns 删除旧 question/source 等字段。
    filter 只筛题，不裁掉题目正文。长度由当前 tokenizer 的 chat_template 决定。
    """
    raw = load_dataset("json", data_files={"train": str(path)})["train"]
    data = raw.map(format_math_example, remove_columns=raw.column_names)
    invalid_gold_ids = [row["id"] for row in data if parse_numeric_answer(row["correct_answer_text"]) is None]
    if invalid_gold_ids:
        raise ValueError(f"GSM8K 标准答案不符合冻结的数值语法：{invalid_gold_ids[:20]}")
    data = data.filter(lambda row: len(tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True,
    )) <= max_prompt_length)
    if len(data) == 0:
        raise ValueError("过滤后没有训练题。")
    return data, {"raw_questions": len(raw), "kept_questions": len(data)}


def build_training_args(options):
    """把回答总数换成框架的梯度累积步数，不手写 backward/update 循环。"""
    return GRPOConfig(
        output_dir=str(options.output_dir),
        per_device_train_batch_size=options.micro_batch_size,
        gradient_accumulation_steps=options.answers_per_update // options.micro_batch_size,
        generation_batch_size=options.generation_batch_size,
        num_generations=options.num_generations, num_iterations=1,
        per_device_eval_batch_size=16, num_generations_eval=1,
        max_completion_length=options.max_completion_length,
        temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
        generation_kwargs={"use_cache": True},
        learning_rate=options.learning_rate, optim="adamw_torch", weight_decay=0.0,
        lr_scheduler_type="constant", max_grad_norm=1.0,
        beta=0.0, loss_type="grpo", scale_rewards="group", epsilon=0.2,
        mask_truncated_completions=False,
        # BF16 权重已显式加载；此处关闭 autocast，保留 FP32 A/B 的矩阵计算。
        bf16=False, fp16=False, use_cpu=options.device == "cpu",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=options.device == "cuda", use_vllm=False,
        pad_to_multiple_of=64, torch_empty_cache_steps=None,
        max_steps=options.max_steps, eval_strategy="no", eval_on_start=False,
        save_strategy="steps", save_steps=options.save_steps,
        save_total_limit=options.save_total_limit, save_only_model=False,
        logging_steps=1, report_to="none", seed=options.seed, data_seed=options.seed,
    )


def latest_checkpoint(output_dir, variant, device):
    """由 Trainer 找最后 checkpoint，再检查优化器/调度器/RNG/权重是否齐全。

    不跳过损坏的最新版本偷偷恢复旧版；遇到不完整保存会给出明确错误。
    """
    latest = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    if latest is None:
        raise ValueError("--resume 指定的目录没有 checkpoint。")
    checkpoint = Path(latest)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_budget.json"]
    required += ["adapter_model.safetensors", "adapter_config.json"] if variant == "lora" else ["geora.safetensors", "geora_config.json"]
    if device == "mps":
        required.append("mps_rng_state.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"最新 checkpoint 不完整：{checkpoint}，缺少 {missing}")
    return checkpoint


def validate_parameters(model, target_names, rank):
    """核对可训练元素数量/精度；两种方法必须训练同样大小的 A/B。"""
    expected = 0
    for name in target_names:
        layer = model.get_submodule(name)
        if hasattr(layer, "base_layer"):
            layer = layer.base_layer
        shape = layer.W_res.shape if hasattr(layer, "W_res") else layer.weight.shape
        expected += rank * (shape[0] + shape[1])
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if sum(value.numel() for _, value in trainable) != expected:
        raise ValueError("可训练参数数量与目标层不一致。")
    if not all(value.dtype == torch.float32 for _, value in trainable):
        raise ValueError("本协议要求可训练 A/B 为 FP32。")
    return {"target_layer_count": len(target_names), "trainable_tensors": len(trainable),
            "trainable_elements": expected, "trainable_dtype": "float32"}


class RunMonitor(TrainerCallback):
    """只负责日志、梯度健康与停止请求；不实现采样、loss 或优化器更新。

    SLURM 提前发来的 USR1 / TERM 只设置标志，完成当前优化器步后才保存。
    如果系统来不及等待完整一步，只能从此前已经保存的 checkpoint 续训。
    """

    def __init__(self, options, model, previous_seconds=0.0):
        self.options, self.model = options, model
        self.previous_seconds = previous_seconds
        self.stop_signal = None
        self.gradient_checks = self.nonzero_steps = 0
        self.started = None
        self.last_handlers = {}
        # 只观察一层；统计比较的是本次进程开始后变化，不是从最初训练起累计。
        if options.variant == "lora":
            self.watched_name, layer = next((name, layer) for name, layer in model.named_modules() if hasattr(layer, "lora_A"))
            self.A, self.B = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        else:
            self.watched_name, layer = next(iter(geora_layers(model).items()))
            self.A, self.B = layer.A, layer.B
        self.initial_A, self.initial_B = self.A.detach().cpu().clone(), self.B.detach().cpu().clone()

    def request_stop(self, number, frame):
        self.stop_signal = signal.Signals(number).name

    def on_train_begin(self, args, state, control, **kwargs):
        self.started, self.start_step = time.monotonic(), state.global_step
        for number in (signal.SIGTERM, signal.SIGUSR1):
            self.last_handlers[number] = signal.signal(number, self.request_stop)
        if self.A.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def elapsed(self):
        return 0.0 if self.started is None else time.monotonic() - self.started

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
            raise RuntimeError("缺少梯度或出现 NaN/Inf；停止，保留此前完整 checkpoint。")
        self.gradient_checks += 1
        self.nonzero_steps += int(torch.stack([g.detach().abs().max() for g in gradients]).max().item() > 0)

    def on_step_end(self, args, state, control, **kwargs):
        timed_out = self.options.hours is not None and self.elapsed() >= self.options.hours * 3600
        if timed_out or self.stop_signal:
            print(f"\n将在完整更新后保存并停止：{self.stop_signal or '本次训练时限'}", flush=True)
            control.should_save = control.should_training_stop = True
        # 最后一步也保存；不依赖最后一步刚好是 save_steps 的倍数。
        if state.global_step >= args.max_steps:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        write_json(self.options.output_dir / f"checkpoint-{state.global_step}" / "training_budget.json",
                   {"training_seconds": self.previous_seconds + self.elapsed()})

    def on_log(self, args, state, control, logs=None, **kwargs):
        entry = {"step": state.global_step, "session_elapsed_seconds": self.elapsed(), **(logs or {})}
        if self.A.device.type == "cuda":
            entry["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
            entry["peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
        with (self.options.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def health(self, step):
        result = {
            "global_step": step, "session_start_step": self.start_step,
            "session_training_seconds": self.elapsed(),
            "total_training_seconds": self.previous_seconds + self.elapsed(),
            "gradient_checks": self.gradient_checks, "steps_with_nonzero_gradient": self.nonzero_steps,
            "watched_layer": self.watched_name,
            "A_max_change_since_session_start": (self.A.detach().cpu() - self.initial_A).abs().max().item(),
            "B_max_change_since_session_start": (self.B.detach().cpu() - self.initial_B).abs().max().item(),
            "stop_signal": self.stop_signal,
        }
        if self.A.device.type == "cuda":
            result.update(peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
        return result

    def restore_handlers(self):
        for number, handler in self.last_handlers.items():
            signal.signal(number, handler)


def main():
    options = parse_options()
    for key in ("base_model", "train_data", "output_dir"):
        setattr(options, key, getattr(options, key).expanduser().resolve())
    print(json.dumps({"settings": protocol_settings(options),
                      "gradient_accumulation_steps": options.answers_per_update // options.micro_batch_size,
                      "questions_per_update": options.answers_per_update // options.num_generations,
                      "target_total_steps": options.max_steps}, ensure_ascii=False, indent=2), flush=True)
    if options.dry_run:
        return
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA/ROCm GPU。先运行 check_setonix_environment.py。")
    if options.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("未检测到 MPS。")

    tokenizer = AutoTokenizer.from_pretrained(options.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data, data_counts = prepare_dataset(options.train_data, tokenizer, options.max_prompt_length)
    protocol = {
        "purpose": "independent_paper_aligned_reimplementation_v1",
        "settings": protocol_settings(options), "data_counts": data_counts,
        "train_file_sha256": sha256_file(options.train_data),
        "base_model_path": str(options.base_model),
        # 连同全部权重和 tokenizer 文件记录摘要，不只比较模型配置名称。
        # 每次启动/续训需顺序读取一次 checkpoint，是有意支付的完整性检查成本。
        "base_files_sha256": {
            path.name: sha256_file(path) for path in sorted(options.base_model.iterdir())
            if path.is_file() and (path.suffix in (".safetensors", ".bin", ".json", ".jinja", ".txt", ".model"))
        },
        "tokenizer_chat_template": tokenizer.chat_template,
        "versions": {name: version(name) for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "math-verify", "safetensors", "numpy", "pyarrow", "pandas", "sympy", "latex2sympy2_extended", "antlr4-python3-runtime")},
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in ("train_paper_grpo.py", "geora_paper_utils.py", "answer_scoring.py")},
    }
    protocol_path = options.output_dir / "run_protocol.json"
    checkpoint = None
    if options.resume:
        if not protocol_path.is_file() or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("续训协议/代码/数据/依赖与原运行不一致。请检查差异；不要覆盖原运行。")
        checkpoint = latest_checkpoint(options.output_dir, options.variant, options.device)
        saved_step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
        if saved_step >= options.max_steps:
            print(f"已有 step={saved_step}；目标已达到，未训练。")
            return
    elif options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise ValueError("输出目录非空；若续训请加 --resume，否则选新的输出目录。")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(protocol_path, protocol)
    write_json(options.output_dir / "invocation.json", vars(options))

    initialized_at = time.monotonic()
    set_seed(options.seed)
    model = AutoModelForCausalLM.from_pretrained(options.base_model, local_files_only=True, dtype=torch.bfloat16).to(options.device)
    model.config.use_cache = False
    target_names = linear_target_names(model)
    if checkpoint is not None:
        if options.variant == "lora":
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True, local_files_only=True)
        else:
            load_geora_layers(model, checkpoint, install=True, expected_base_model_path=options.base_model)
    elif options.variant == "lora":
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules="all-linear", r=options.rank,
            lora_alpha=options.alpha, lora_dropout=0.0, bias="none",
        ))
    else:
        install_geora_layers(model, options.rank, options.rho, options.alpha)
    # PEFT 名称多了包装前缀；通过后缀定位同一批原始层，GeoRA 则不增加前缀。
    if options.variant == "lora":
        wrapped_names = {name for name, layer in model.named_modules() if hasattr(layer, "lora_A")}
        target_names = [next(name for name in wrapped_names if name.endswith("." + target)) for target in target_names]
    parameters = validate_parameters(model, target_names, options.rank)
    print("可训练参数检查：", parameters, flush=True)
    initialization_seconds = time.monotonic() - initialized_at
    previous_seconds = 0.0 if checkpoint is None else json.loads((checkpoint / "training_budget.json").read_text())["training_seconds"]
    monitor = RunMonitor(options, model, previous_seconds)
    trainer_class = ExperimentGRPOTrainer if options.variant == "lora" else GeoRAGRPOTrainer
    extra = {} if options.variant == "lora" else {"base_model_path": options.base_model}
    # 两种方法的初始化消耗随机数不同。重新设种子，让新训练采样从同一 seed 开始。
    # 续训时 Trainer 会再恢复 checkpoint 中的 RNG、优化器、调度器和数据进度。
    set_seed(options.seed)
    reward_function = partial(grpo_boxed_reward, audit_path=options.output_dir / "reward_audit.jsonl")
    reward_function.__name__ = "gsm8k_exact_reward"
    trainer = trainer_class(model=model, processing_class=tokenizer, args=build_training_args(options),
                            reward_funcs=reward_function, train_dataset=train_data,
                            callbacks=[monitor], **extra)
    if str(trainer.accelerator.device).split(":")[0] != options.device:
        raise RuntimeError(f"框架设备 {trainer.accelerator.device} 与指定设备 {options.device} 不一致。")
    try:
        result = trainer.train(resume_from_checkpoint=None if checkpoint is None else str(checkpoint))
        trainer.save_metrics("train", result.metrics)
        health = monitor.health(trainer.state.global_step)
        write_json(options.output_dir / "health.json", health)
        write_json(options.output_dir / "run_summary.json", {
            "variant": options.variant, "parameters": parameters,
            "initialization_seconds": initialization_seconds, "health": health,
            "checkpoint": str(latest_checkpoint(options.output_dir, options.variant, options.device)),
            "train_metrics": result.metrics,
        })
        print("完成：", options.output_dir / "run_summary.json", flush=True)
    finally:
        monitor.restore_handlers()


if __name__ == "__main__":
    main()
