import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer
import utils.hadamard_utils as hadamard_utils

class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        self.online_full_had=False
        self.use_temporary_parameter=False



    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had: # Full Hadamard in FP32
                input = hadamard_utils.matmul_hadU_cuda(input.float(), self.had_K, self.K).to(input_dtype)
            else: # Full Hadamard in FP16
                input = hadamard_utils.matmul_hadU_cuda(input, self.had_K, self.K)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias

        if self.use_weight_quant and self.wbits < 16:
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and self.input_bits < 16:
            input = self.input_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        # Apply inverse Hadamard for Q/K projections (after linear transformation)
        if hasattr(self, 'online_qk_had') and self.online_qk_had:
            # Reshape for per-head inverse Hadamard application
            bsz, seq_len, total_dim = out.shape

            # Reshape to (bsz, seq_len, num_heads, head_dim)
            out = out.view(bsz, seq_len, self.qk_num_heads, self.qk_head_dim)

            # Apply inverse Hadamard per head using transposed hadK
            if self.qk_fp32_had:  # Full Hadamard in FP32
                out = hadamard_utils.matmul_hadU_cuda(out.float(), self.qk_had_K.T, self.qk_K).to(out.dtype)
            else:  # Full Hadamard in current precision
                out = hadamard_utils.matmul_hadU_cuda(out, self.qk_had_K.T, self.qk_K)

            # Reshape back to (bsz, seq_len, total_dim)
            out = out.view(bsz, seq_len, total_dim)

        if self.use_act_quant and self.output_bits < 16:
            out = self.output_quantizer(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
