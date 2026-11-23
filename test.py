import copy
import torch
import torch.nn as nn
import wandb
from model import get_cifar10_mobilenetv2
from compress import CustomQuantConv2d
from memory_profiler import measure_model_weight_memory, measure_activation_memory

def apply_compression(model: nn.Module, w_bits: int, a_bits: int, skip_first_last: bool = True) -> nn.Module:
    comp_model = copy.deepcopy(model)
    conv_names = [n for n, m in comp_model.named_modules() if isinstance(m, nn.Conv2d)]

    for name, module in list(comp_model.named_modules()):
        if isinstance(module, nn.Conv2d):
            # Optional: keep first conv layer in higher precision (8-bit) to preserve accuracy
            bits_w = 8 if (skip_first_last and name == conv_names[0]) else w_bits
            bits_a = 8 if (skip_first_last and name == conv_names[0]) else a_bits

            parent_name, attr_name = name.rsplit(".", 1)
            parent = comp_model.get_submodule(parent_name)
            setattr(parent, attr_name, CustomQuantConv2d(module, w_bits=bits_w, a_bits=bits_a))

    return comp_model

# Example usage inside your sweep loop:
# 1. Measure Baseline
sample_x = torch.randn(1, 3, 32, 32).cuda()
base_w = measure_model_weight_memory(baseline_model)
base_a = measure_activation_memory(baseline_model, sample_x)
base_total_mb = base_w["total_weight_mb"] + base_a["total_activation_mb"]

# 2. Measure Compressed Model
comp_model = apply_compression(baseline_model, w_bits=4, a_bits=8).cuda()
comp_w = measure_model_weight_memory(comp_model)
comp_a = measure_activation_memory(comp_model, sample_x)
comp_total_mb = comp_w["total_weight_mb"] + comp_a["total_activation_mb"]

weight_cr = base_w["total_weight_bytes"] / comp_w["total_weight_bytes"]
act_cr = base_a["total_activation_bytes"] / comp_a["total_activation_bytes"]
overall_cr = base_total_mb / comp_total_mb

# 3. Log exact metrics for the W&B Parallel Coordinates plot
wandb.log({
    "activation_quant_bits": 8,
    "weight_quant_bits": 4,
    "compression_ratio": overall_cr,
    "weight_compression_ratio": weight_cr,
    "activation_compression_ratio": act_cr,
    "model_size_mb": comp_w["total_weight_mb"],
    "metadata_overhead_mb": comp_w["metadata_overhead_mb"],
    "quantized_acc": eval_top1_accuracy(comp_model, test_loader),
})