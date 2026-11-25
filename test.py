%%writefile test.py
import argparse
import copy
import os
import torch
import torch.nn as nn
import wandb

from model import get_cifar10_mobilenetv2
from dataset import get_cifar10_dataloaders
from compress import CustomQuantConv2d
from memory_profiler import measure_model_weight_memory, measure_activation_memory
from utils import set_seed, evaluate_top1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Compressed MobileNet-v2 on CIFAR-10")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/baseline_mobilenetv2.pth", help="Path to trained baseline checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="CIFAR-10 dataset directory")
    parser.add_argument("--weight_quant_bits", type=int, default=8, help="Bit-width for weight quantization (e.g., 2, 3, 4, 8)")
    parser.add_argument("--activation_quant_bits", type=int, default=8, help="Bit-width for activation quantization (e.g., 2, 3, 4, 8)")
    parser.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="cs6886w-mobilenetv2-compression", help="W&B project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging for local offline tests")
    return parser.parse_args()


def apply_compression(model: nn.Module, w_bits: int, a_bits: int, skip_first_conv: bool = True) -> nn.Module:
    comp_model = copy.deepcopy(model)
    conv_names = [n for n, m in comp_model.named_modules() if isinstance(m, nn.Conv2d)]

    for name, module in list(comp_model.named_modules()):
        if isinstance(module, nn.Conv2d):
            bits_w = 8 if (skip_first_conv and name == conv_names[0]) else w_bits
            bits_a = 8 if (skip_first_conv and name == conv_names[0]) else a_bits

            parent_name, attr_name = name.rsplit(".", 1)
            parent = comp_model.get_submodule(parent_name)
            setattr(parent, attr_name, CustomQuantConv2d(module, w_bits=bits_w, a_bits=bits_a))

    return comp_model


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load CIFAR-10 test set
    _, test_loader = get_cifar10_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed)

    # 2. Instantiate and load the trained baseline model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Pass --checkpoint <path>.")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("config", {})
    baseline_model = get_cifar10_mobilenetv2(
        num_classes=10,
        width_mult=ckpt_cfg.get("width_mult", 1.0),
        dropout=ckpt_cfg.get("dropout", 0.2),
    ).to(device)
    baseline_model.load_state_dict(ckpt["model_state_dict"])
    baseline_model.eval()

    # 3. Profile baseline memory and accuracy
    sample_x = torch.randn(1, 3, 32, 32, device=device)
    base_w = measure_model_weight_memory(baseline_model)
    base_a = measure_activation_memory(baseline_model, sample_x)
    base_total_bytes = base_w["total_weight_bytes"] + base_a["total_activation_bytes"]
    _, base_acc = evaluate_top1(baseline_model, test_loader, device)

    # 4. Compress model and profile compressed memory & accuracy
    comp_model = apply_compression(
        baseline_model,
        w_bits=args.weight_quant_bits,
        a_bits=args.activation_quant_bits,
    ).to(device)
    comp_model.eval()

    comp_w = measure_model_weight_memory(comp_model)
    comp_a = measure_activation_memory(comp_model, sample_x)
    comp_total_bytes = comp_w["total_weight_bytes"] + comp_a["total_activation_bytes"]
    _, quantized_acc = evaluate_top1(comp_model, test_loader, device)

    weight_cr = base_w["total_weight_bytes"] / comp_w["total_weight_bytes"]
    act_cr = base_a["total_activation_bytes"] / comp_a["total_activation_bytes"]
    overall_cr = base_total_bytes / comp_total_bytes

    print(f"=== W{args.weight_quant_bits} / A{args.activation_quant_bits} Results ===")
    print(f"Baseline Acc:   {base_acc:.2f}% | Compressed Acc: {quantized_acc:.2f}%")
    print(f"Weight Size:    {base_w['total_weight_mb']:.3f} MB -> {comp_w['total_weight_mb']:.3f} MB (CR: {weight_cr:.2f}x)")
    print(f"Metadata OvHd:  {comp_w['metadata_overhead_mb']:.4f} MB")
    print(f"Act Size (Tot): {base_a['total_activation_mb']:.3f} MB -> {comp_a['total_activation_mb']:.3f} MB (CR: {act_cr:.2f}x)")
    print(f"Overall CR:     {overall_cr:.2f}x")

    # 5. Log metrics to Weights & Biases
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=f"w{args.weight_quant_bits}_a{args.activation_quant_bits}",
            config=vars(args),
        )
        wandb.log({
            "activation_quant_bits": args.activation_quant_bits,
            "weight_quant_bits": args.weight_quant_bits,
            "compression_ratio": overall_cr,
            "weight_compression_ratio": weight_cr,
            "activation_compression_ratio": act_cr,
            "model_size_mb": comp_w["total_weight_mb"],
            "metadata_overhead_mb": comp_w["metadata_overhead_mb"],
            "quantized_acc": quantized_acc,
            "baseline_acc": base_acc,
        })
        run.finish()


if __name__ == "__main__":
    main()
