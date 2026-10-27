"""已经通过 notebook 检查的 GeoRA 层、严格 boxed 奖励及 TRL 存取适配。

训练循环、GRPO loss、梯度累积、优化器和数据进度均由 TRL 管理。
本模块只定义自定义层及其保存方式，不依赖其他 agent 的 GeoRA 实现。
"""

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from answer_scoring import correctness_reward, extract_final_boxed, score_gsm8k
from safetensors.torch import load_file, save_file
from trl import GRPOTrainer


def format_math_example(row):
    """将一行 GSM8K 数据转换为 conversational prompt 和标准答案。"""
    return {
        "id": row["id"],
        "prompt": [{
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "Put the final numeric answer, without units, inside \\boxed{}."
                "\nQuestion: " + row["question"]
            ),
        }],
        "correct_answer_text": str(row["target"]),
    }


def grpo_boxed_reward(completions, correct_answer_text, audit_path=None, **kwargs):
    """TRL 批量奖励回调：输入多份回答，按原顺序返回等长的奖励列表。"""
    records = [
        score_gsm8k(completion[-1]["content"], target)
        for completion, target in zip(completions, correct_answer_text, strict=True)
    ]
    log_extra = kwargs.get("log_extra")
    if log_extra is not None:
        log_extra("reward_status", [record["status"] for record in records])
        log_extra("extracted_answer", [record["extracted_answer"] for record in records])
    if audit_path is not None:
        state = kwargs.get("trainer_state")
        summary = {"optimizer_step_before_update": getattr(state, "global_step", None),
                   "responses": len(records),
                   "status_counts": {status: sum(record["status"] == status for record in records)
                                     for status in ("correct", "incorrect", "missing_or_incomplete_boxed", "invalid_prediction")}}
        with Path(audit_path).open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
    return [record["reward"] for record in records]


def initialize_geora_factors(weight, rank, rho):
    """FP32 [out, in] 权重 -> A0 [rank, in]、B0 [out, rank]。"""
    if weight.ndim != 2 or weight.dtype != torch.float32:
        raise ValueError("GeoRA 初始化需要二维 FP32 权重。")
    if not 1 <= rank <= min(weight.shape) or not 0 <= rho <= 1:
        raise ValueError("rank 或 rho 超出有效范围。")
    absolute_weight = weight.abs()
    euclidean_mask = absolute_weight <= torch.quantile(absolute_weight.flatten(), rho)
    U, singular_values, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank_weight = U[:, :rank] @ torch.diag(singular_values[:rank]) @ Vh[:rank, :]
    absolute_rank_weight = rank_weight.abs()
    spectral_mask = absolute_rank_weight <= torch.quantile(absolute_rank_weight.flatten(), rho)
    geo_weight = weight * (euclidean_mask | spectral_mask)
    U, singular_values, Vh = torch.linalg.svd(geo_weight, full_matrices=False)
    sqrt_singular_values = torch.diag(singular_values[:rank].sqrt())
    return sqrt_singular_values @ Vh[:rank, :], U[:, :rank] @ sqrt_singular_values


class GeoRALinear(nn.Module):
    """沿用 notebook 的残差实现；训练 A/B，冻结 W_res/bias。

    输入 [..., in] -> 输出 [..., out]。初始化用 FP32。
    两条分支分别按各自权重 dtype 计算，支持 BF16 残差 + FP32 A/B。
    相加后输出恢复为输入 dtype，与 PEFT LoRA 的分支精度处理一致。
    W_res = W - (alpha/rank) * B0 @ A0。
    """

    def __init__(self, weight, rank, rho, alpha, bias=None):
        super().__init__()
        self.rank, self.rho, self.alpha = rank, rho, alpha
        self.scaling = alpha / rank
        with torch.no_grad():
            A0, B0 = initialize_geora_factors(weight, rank, rho)
            residual = weight - self.scaling * (B0 @ A0)
        self.A = nn.Parameter(A0.detach().clone())
        self.B = nn.Parameter(B0.detach().clone())
        self.register_buffer("W_res", residual.detach())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_saved_tensors(cls, state, rank, rho, alpha):
        """从已保存的 A/B/残差重建层，避免恢复时重新做 SVD。

        直接注册已有张量即可；这里特意不调用执行 SVD 的 __init__。
        """
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.rank, layer.rho, layer.alpha = rank, rho, alpha
        layer.scaling = alpha / rank
        layer.A = nn.Parameter(state["A"].clone())
        layer.B = nn.Parameter(state["B"].clone())
        layer.register_buffer("W_res", state["W_res"].clone())
        layer.register_buffer("bias", state.get("bias"))
        return layer

    def forward(self, x):
        input_dtype = x.dtype
        # 固定分支与适配器分支可以使用不同精度，不把整个输入统一转 FP32。
        residual_input = x.to(dtype=self.W_res.dtype)
        residual_output = F.linear(residual_input, self.W_res, self.bias)
        adapter_input = x.to(dtype=self.A.dtype)
        compressed_input = F.linear(adapter_input, self.A)
        adapter_output = F.linear(compressed_input, self.B)
        return (residual_output + self.scaling * adapter_output).to(dtype=input_dtype)


def geora_layers(model):
    """返回当前模型的 {完整层名称: GeoRALinear 对象}。"""
    return {name: layer for name, layer in model.named_modules() if isinstance(layer, GeoRALinear)}


def linear_target_names(model):
    """返回与本次 PEFT all-linear 对齐的 Linear 名称，排除输出 lm_head。

    对 Qwen2.5-1.5B 是 28 层 × 每层 q/k/v/o/gate/up/down = 196。
    只保留名称，避免替换过程中一直持有全部旧权重的引用。
    """
    output = model.get_output_embeddings()
    return [name for name, layer in model.named_modules()
            if isinstance(layer, nn.Linear) and layer is not output]


def install_geora_layers(model, rank=16, rho=0.2, alpha=32):
    """冻结原模型，以 FP32 CPU SVD 初始化所有目标层，返回层名称列表。

    固定 W_res/bias 使用 BF16，训练 A/B 使用 FP32。
    SVD 的临时 FP32 矩阵在 CPU，逐层释放；这不是把整个模型转 FP32。
    """
    model.requires_grad_(False)
    targets = linear_target_names(model)
    if not targets:
        raise ValueError("没有找到尚未替换的线性层。")
    for index, name in enumerate(targets, start=1):
        original = model.get_submodule(name)
        print(f"初始化 GeoRA {index}/{len(targets)}：{name}", flush=True)
        weight = original.weight.detach().to(device="cpu", dtype=torch.float32)
        bias = None if original.bias is None else original.bias.detach().to(device="cpu", dtype=torch.float32)
        layer = GeoRALinear(weight, rank, rho, alpha, bias)
        layer.W_res = layer.W_res.to(dtype=torch.bfloat16)
        if layer.bias is not None:
            layer.bias = layer.bias.to(dtype=torch.bfloat16)
        layer = layer.to(device=original.weight.device)
        model.set_submodule(name, layer, strict=True)
        del original, weight, bias, layer
    return targets


def save_geora_layers(model, destination, base_model_path):
    """保存 A/B、固定残差和 bias，加上原始 checkpoint 路径；不重复保存其余基座权重。"""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    layers = geora_layers(model)
    if not layers:
        raise ValueError("模型中没有 GeoRA 层。")
    tensors, layer_configs = {}, {}
    for name, layer in layers.items():
        layer_configs[name] = {"rank": layer.rank, "rho": layer.rho, "alpha": layer.alpha}
        for key, tensor in layer.state_dict().items():
            tensors[f"{name}.{key}"] = tensor.detach().cpu().contiguous()
    save_file(tensors, str(destination / "geora.safetensors"))
    manifest = {"format_version": 1, "base_model_path": str(Path(base_model_path).resolve()), "layers": layer_configs}
    (destination / "geora_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def load_geora_layers(model, checkpoint, install=False, expected_base_model_path=None):
    """加载 GeoRA 状态；install=True 用于新基座，其余情况原地复制，保留优化器引用。"""
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "geora_config.json").read_text())
    if manifest["format_version"] != 1:
        raise ValueError("不支持的 GeoRA checkpoint 格式。")
    if expected_base_model_path is not None and Path(manifest["base_model_path"]).resolve() != Path(expected_base_model_path).resolve():
        raise ValueError("GeoRA checkpoint 对应的基座路径不一致。")
    tensors = load_file(str(checkpoint / "geora.safetensors"), device="cpu")
    expected_keys = set()
    if install:
        model.requires_grad_(False)
    elif set(geora_layers(model)) != set(manifest["layers"]):
        raise ValueError("GeoRA 层名称与 checkpoint 不一致。")

    for name, config in manifest["layers"].items():
        prefix = name + "."
        state = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
        expected_keys.update(prefix + key for key in state)
        original = model.get_submodule(name)
        if install:
            if not isinstance(original, nn.Linear) or tuple(original.weight.shape) != tuple(state["W_res"].shape):
                raise ValueError(f"原始层结构不匹配：{name}")
            layer = GeoRALinear.from_saved_tensors(state, **config).to(device=original.weight.device)
            model.set_submodule(name, layer, strict=True)
        else:
            if any(getattr(original, key) != value for key, value in config.items()):
                raise ValueError(f"GeoRA 初始化设置不匹配：{name}")
            original.load_state_dict(state, strict=True)
    if expected_keys != set(tensors):
        raise ValueError("checkpoint 包含未识别的张量。")
    return manifest


class ExperimentGRPOTrainer(GRPOTrainer):
    """三组共用的贪心评估和 MPS RNG 适配；LoRA 的存取直接沿用 PEFT/Trainer。

    本轮限定单设备、beta=0、num_iterations=1；不实现分布式 checkpoint 或 KL reference。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("当前实验适配仅支持单设备。")
        if self.beta != 0 or self.num_iterations != 1:
            raise ValueError("本轮保持 beta=0、num_iterations=1。")

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self._save_mps_rng(Path(output_dir or self.args.output_dir))

    def _save_mps_rng(self, destination):
        # 当前安装版本的 Trainer 没有保存 MPS RNG，补上这一项。
        if self.accelerator.device.type == "mps":
            torch.save(torch.mps.get_rng_state(), destination / "mps_rng_state.pt")

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        if checkpoint is not None and self.accelerator.device.type == "mps":
            state = torch.load(Path(checkpoint) / "mps_rng_state.pt", map_location="cpu", weights_only=True)
            torch.mps.set_rng_state(state)

    def evaluation_loop(self, *args, **kwargs):
        # 训练仍随机采样；仅在评估期间切到每题一份贪心回答，结束后恢复。
        training_config, training_kwargs = self.generation_config, self.generation_kwargs
        self.generation_config = copy.deepcopy(training_config)
        self.generation_config.do_sample = False
        self.generation_kwargs = dict(training_kwargs, do_sample=False)
        for key in ("temperature", "top_p", "top_k", "min_p"):
            setattr(self.generation_config, key, None)
            self.generation_kwargs[key] = None
        try:
            return super().evaluation_loop(*args, **kwargs)
        finally:
            self.generation_config, self.generation_kwargs = training_config, training_kwargs


class GeoRAGRPOTrainer(ExperimentGRPOTrainer):
    """只补充自定义 GeoRA 层的存取；其余与 LoRA 的训练流程一致。"""

    def __init__(self, *args, base_model_path, **kwargs):
        self.base_model_path = str(Path(base_model_path).resolve())
        super().__init__(*args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        destination = Path(output_dir or self.args.output_dir)
        save_geora_layers(self.accelerator.unwrap_model(self.model), destination, self.base_model_path)
        self.processing_class.save_pretrained(destination)
        torch.save(self.args, destination / "training_args.bin")
        self._save_mps_rng(destination)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_geora_layers(
            self.accelerator.unwrap_model(self.model if model is None else model),
            resume_from_checkpoint,
            expected_base_model_path=self.base_model_path,
        )
