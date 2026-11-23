import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def pack_low_bit_tensor(q_tensor: torch.Tensor, num_bits: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Packs an integer tensor with values in [0, 2^num_bits - 1] into a 1D uint8 tensor
    in memory so RAM/VRAM usage reflects true bit-width compression.
    """
    orig_shape = tuple(q_tensor.shape)
    flat = q_tensor.flatten().to(torch.uint8)

    if num_bits == 8:
        return flat, orig_shape

    # Number of values packed into each 8-bit byte (e.g., two 4-bit values per byte)
    vals_per_byte = 8 // num_bits
    pad_len = (vals_per_byte - (flat.numel() % vals_per_byte)) % vals_per_byte
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0)

    reshaped = flat.view(-1, vals_per_byte)
    packed = torch.zeros(reshaped.size(0), dtype=torch.uint8, device=q_tensor.device)
    for i in range(vals_per_byte):
        packed |= (reshaped[:, i] & ((1 << num_bits) - 1)) << (i * num_bits)

    return packed, orig_shape


def unpack_low_bit_tensor(packed: torch.Tensor, orig_shape: tuple[int, ...], num_bits: int) -> torch.Tensor:
    """Unpacks a 1D uint8 tensor back to integer values on the fly during forward pass."""
    if num_bits == 8:
        return packed.view(orig_shape)

    vals_per_byte = 8 // num_bits
    total_elements = math.prod(orig_shape)
    mask = (1 << num_bits) - 1

    unpacked_cols = [((packed >> (i * num_bits)) & mask) for i in range(vals_per_byte)]
    unpacked = torch.stack(unpacked_cols, dim=1).flatten()[:total_elements]
    return unpacked.view(orig_shape)


def quantize_tensor(x: torch.Tensor, num_bits: int, per_channel: bool = False):
    """Uniform asymmetric quantization returning integer codes, scale, and zero_point."""
    qmin, qmax = 0, (1 << num_bits) - 1
    if per_channel and x.ndim >= 2:
        dims = tuple(range(1, x.ndim))
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()

    scale = (x_max - x_min).clamp(min=1e-8) / float(qmax - qmin)
    zero_point = torch.round(-x_min / scale).clamp(qmin, qmax).to(torch.uint8)
    q_x = torch.round(x / scale + zero_point.float()).clamp(qmin, qmax).to(torch.uint8)
    # Store scale in float16 to minimize metadata overhead
    return q_x, scale.to(torch.float16), zero_point


class CustomQuantConv2d(nn.Module):
    """
    Replaces nn.Conv2d with bit-packed weights + metadata buffers and dynamic activation quantization.
    """
    def __init__(self, conv: nn.Conv2d, w_bits: int = 8, a_bits: int = 8):
        super().__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        with torch.no_grad():
            q_w, w_scale, w_zp = quantize_tensor(conv.weight.data, w_bits, per_channel=True)
            packed_w, orig_shape = pack_low_bit_tensor(q_w, w_bits)

        self.orig_weight_shape = orig_shape
        # Register as buffers so float32 parameters are freed from memory
        self.register_buffer("packed_weight", packed_w)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("w_zp", w_zp)

        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.data.to(torch.float16))
        else:
            self.bias = None

        # Track last quantized activation size (in bytes) for profiling
        self.last_act_bytes = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize input activations to a_bits (simulated packed memory footprint)
        if self.a_bits < 32:
            q_x, a_scale, a_zp = quantize_tensor(x, self.a_bits, per_channel=False)
            # Actual memory required if stored at a_bits + scale/zp metadata
            act_payload_bytes = math.ceil((q_x.numel() * self.a_bits) / 8.0)
            act_meta_bytes = a_scale.element_size() * a_scale.nelement() + a_zp.element_size() * a_zp.nelement()
            self.last_act_bytes = act_payload_bytes + act_meta_bytes
            x_dequant = (q_x.float() - a_zp.float()) * a_scale.float()
        else:
            self.last_act_bytes = x.element_size() * x.nelement()
            x_dequant = x

        # 2. Unpack weights from uint8 buffer and dequantize for compute
        q_w = unpack_low_bit_tensor(self.packed_weight, self.orig_weight_shape, self.w_bits)
        w_dequant = (q_w.float() - self.w_zp.float()) * self.w_scale.float()
        bias = self.bias.float() if self.bias is not None else None

        return F.conv2d(x_dequant, w_dequant, bias, self.stride, self.padding, self.dilation, self.groups)