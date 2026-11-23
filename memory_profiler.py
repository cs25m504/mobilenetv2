import torch
import torch.nn as nn

def measure_model_weight_memory(model: nn.Module) -> dict:
    """
    Computes exact in-memory size (bytes and MB) of all model weights, biases,
    BatchNorm stats, and quantization metadata (scales, zero-points).
    """
    payload_bytes = 0
    metadata_bytes = 0

    # 1. Uncompressed / trainable parameters (e.g., remaining float32 layers, BN weights)
    for name, param in model.named_parameters():
        payload_bytes += param.element_size() * param.nelement()

    # 2. Registered buffers (packed weights + scale/zero_point + BN running_mean/var)
    for name, buf in model.named_buffers():
        nbytes = buf.element_size() * buf.nelement()
        if any(k in name for k in ["scale", "zp", "zero_point"]):
            metadata_bytes += nbytes
        else:
            payload_bytes += nbytes

    total_bytes = payload_bytes + metadata_bytes
    return {
        "weight_payload_mb": payload_bytes / (1024 ** 2),
        "metadata_overhead_mb": metadata_bytes / (1024 ** 2),
        "total_weight_mb": total_bytes / (1024 ** 2),
        "total_weight_bytes": total_bytes,
    }


def measure_activation_memory(model: nn.Module, sample_input: torch.Tensor) -> dict:
    """
    Runs a single forward pass to measure total and peak activation memory across layers.
    """
    model.eval()
    layer_act_bytes = {}
    hooks = []

    def hook_fn(name):
        def _hook(module, inp, out):
            # If custom quantized layer tracked its quantized input activation bytes:
            if hasattr(module, "last_act_bytes") and module.last_act_bytes > 0:
                layer_act_bytes[name] = module.last_act_bytes
            elif isinstance(out, torch.Tensor):
                layer_act_bytes[name] = out.element_size() * out.nelement()
        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)) or hasattr(mod, "last_act_bytes"):
            hooks.append(mod.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        _ = model(sample_input)

    for h in hooks:
        h.remove()

    total_act_bytes = sum(layer_act_bytes.values())
    peak_act_bytes = max(layer_act_bytes.values()) if layer_act_bytes else 0

    return {
        "total_activation_mb": total_act_bytes / (1024 ** 2),
        "peak_activation_mb": peak_act_bytes / (1024 ** 2),
        "total_activation_bytes": total_act_bytes,
        "peak_activation_bytes": peak_act_bytes,
    }