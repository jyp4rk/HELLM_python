"""
Fixed-Point Arithmetic Implementation for LLaMA Evaluation

Key differences from integer quantization (int_linear_fake):
1. Fixed-point maintains constant decimal position (Q-format)
2. Direct arithmetic operations without dequantization
3. Overflow/underflow handling through saturation
4. No dynamic scale factors - predetermined precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FixedPointConfig:
    """Configuration for fixed-point arithmetic formats"""

    # Common Q-formats (integer_bits.fractional_bits)
    Q8_8 = (8, 8)    # 16-bit: [-128, 127.996], precision 0.00390625
    Q16_16 = (16, 16)  # 32-bit: [-32768, 32767.999], precision 0.0000152587
    Q4_12 = (4, 12)   # 16-bit: [-8, 7.9997], precision 0.000244140625
    Q24_8 = (24, 8)   # 32-bit: Large range, lower precision

    def __init__(self, integer_bits: int, fractional_bits: int):
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits
        self.total_bits = integer_bits + fractional_bits
        self.scale = 2 ** fractional_bits

        # Compute bounds
        self.max_int = 2 ** (self.total_bits - 1) - 1
        self.min_int = -(2 ** (self.total_bits - 1))
        self.max_value = self.max_int / self.scale
        self.min_value = self.min_int / self.scale
        self.precision = 1.0 / self.scale


class FixedPointTensor:
    """
    Fixed-point tensor wrapper that maintains values in Q-format.
    Unlike int_linear_fake which does: float→int→float,
    Fixed-point stays in integer representation throughout computation.
    """

    def __init__(self,
                 data: torch.Tensor,
                 config: FixedPointConfig,
                 is_fixed: bool = False):
        """
        Args:
            data: Float tensor to convert OR already fixed-point tensor
            config: Q-format configuration
            is_fixed: True if data is already in fixed-point format
        """
        self.config = config

        if is_fixed:
            self.data = data.to(torch.int32)
        else:
            # Convert float to fixed-point
            scaled = data * config.scale
            # Saturating conversion (clamp before rounding)
            clamped = scaled.clamp(config.min_int, config.max_int)
            self.data = clamped.round().to(torch.int32)

    def to_float(self) -> torch.Tensor:
        """Convert back to floating-point (only for final output)"""
        return self.data.float() / self.config.scale

    def add(self, other: 'FixedPointTensor') -> 'FixedPointTensor':
        """Fixed-point addition with saturation"""
        assert self.config == other.config, "Q-formats must match"

        # Addition in fixed-point: direct integer add
        result = self.data.to(torch.int64) + other.data.to(torch.int64)

        # Saturate to prevent overflow
        result = result.clamp(self.config.min_int, self.config.max_int)

        return FixedPointTensor(result.to(torch.int32), self.config, is_fixed=True)

    def multiply(self, other: 'FixedPointTensor') -> 'FixedPointTensor':
        """
        Fixed-point multiplication: (a * 2^n) * (b * 2^n) = (a*b) * 2^(2n)
        Need to shift right by n to maintain Q-format
        """
        assert self.config == other.config, "Q-formats must match"

        # Multiply in higher precision to avoid overflow
        result = self.data.to(torch.int64) * other.data.to(torch.int64)

        # Shift right to maintain Q-format (divide by scale)
        result = result >> self.config.fractional_bits

        # Saturate
        result = result.clamp(self.config.min_int, self.config.max_int)

        return FixedPointTensor(result.to(torch.int32), self.config, is_fixed=True)

    def matmul(self, other: 'FixedPointTensor') -> 'FixedPointTensor':
        """
        Matrix multiplication in fixed-point.
        More efficient than element-wise multiply for large matrices.
        """
        # Convert to int64 for accumulation
        a_int64 = self.data.to(torch.int64)
        b_int64 = other.data.to(torch.int64)

        # Perform integer matrix multiplication
        result = torch.matmul(a_int64, b_int64)

        # Scale correction (divide by 2^fractional_bits)
        result = result >> self.config.fractional_bits

        # Saturate
        result = result.clamp(self.config.min_int, self.config.max_int)

        return FixedPointTensor(result.to(torch.int32), self.config, is_fixed=True)


class FixedPointLinear(nn.Module):
    """
    Fixed-point linear layer for LLaMA evaluation.

    Key difference from int_linear_fake:
    - int_linear_fake: Quantize → Compute in float → Dequantize
    - FixedPointLinear: Convert to fixed → Compute in fixed → Stay in fixed
    """

    def __init__(self,
                 linear_module: nn.Linear,
                 weight_config: FixedPointConfig = FixedPointConfig.Q8_8,
                 activation_config: FixedPointConfig = FixedPointConfig.Q8_8):
        super().__init__()

        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.weight_config = weight_config
        self.activation_config = activation_config

        # Pre-convert weights to fixed-point and store
        weight_fp = FixedPointTensor(linear_module.weight, weight_config)
        self.register_buffer('weight_fixed', weight_fp.data)

        if linear_module.bias is not None:
            bias_fp = FixedPointTensor(linear_module.bias, activation_config)
            self.register_buffer('bias_fixed', bias_fp.data)
        else:
            self.bias_fixed = None

        # Store original weights for comparison
        self.register_buffer('weight_float', linear_module.weight.clone())

    def forward(self, x: torch.Tensor, return_fixed: bool = False) -> torch.Tensor:
        """
        Forward pass using fixed-point arithmetic.

        Args:
            x: Input tensor (float)
            return_fixed: If True, return FixedPointTensor instead of float

        Returns:
            Output tensor (float by default, or FixedPointTensor if requested)
        """
        # Convert input to fixed-point
        x_fp = FixedPointTensor(x, self.activation_config)

        # Create weight fixed-point tensor (already converted)
        w_fp = FixedPointTensor(self.weight_fixed, self.weight_config, is_fixed=True)

        # Matrix multiplication in fixed-point
        out_fp = x_fp.matmul(w_fp.data.T)  # Transpose for linear layer

        # Add bias if present (in fixed-point)
        if self.bias_fixed is not None:
            bias_fp = FixedPointTensor(self.bias_fixed, self.activation_config, is_fixed=True)
            out_fp = out_fp.add(bias_fp)

        if return_fixed:
            return out_fp
        else:
            return out_fp.to_float()

    def compare_with_float(self, x: torch.Tensor) -> dict:
        """Compare fixed-point output with float computation"""
        # Fixed-point computation
        out_fixed = self.forward(x)

        # Float computation
        out_float = F.linear(x, self.weight_float,
                           self.bias_fixed.float() / self.activation_config.scale
                           if self.bias_fixed is not None else None)

        # Compute error metrics
        abs_error = (out_fixed - out_float).abs()
        rel_error = abs_error / (out_float.abs() + 1e-8)

        return {
            'fixed_output': out_fixed,
            'float_output': out_float,
            'max_abs_error': abs_error.max().item(),
            'mean_abs_error': abs_error.mean().item(),
            'max_rel_error': rel_error.max().item(),
            'mean_rel_error': rel_error.mean().item(),
            'snr_db': 20 * torch.log10(out_float.norm() / abs_error.norm()).item()
        }


def simulate_accumulation_effects(config: FixedPointConfig, num_accumulations: int):
    """
    Demonstrate accumulation behavior differences between fixed-point and int quantization.

    Fixed-point: Accumulates in same format, may saturate
    Int quant: Each operation requantizes, accumulates rounding errors
    """
    torch.manual_seed(42)

    # Create random values
    values = torch.randn(num_accumulations) * 0.1

    # Fixed-point accumulation
    acc_fp = FixedPointTensor(torch.zeros(1), config)
    for val in values:
        val_fp = FixedPointTensor(val.unsqueeze(0), config)
        acc_fp = acc_fp.add(val_fp)

    # Simulated int-quant accumulation (requantize each step)
    scale = 0.01  # Typical quantization scale
    acc_int = torch.zeros(1)
    for val in values:
        # Quantize-dequantize each addition
        acc_int = torch.round(acc_int / scale) * scale
        val_quant = torch.round(val / scale) * scale
        acc_int = acc_int + val_quant

    # True float accumulation
    acc_true = values.sum()

    print(f"Accumulation comparison ({num_accumulations} operations):")
    print(f"True value:      {acc_true.item():.6f}")
    print(f"Fixed-point:     {acc_fp.to_float().item():.6f}")
    print(f"Int-quant style: {acc_int.item():.6f}")
    print(f"Fixed-point error: {abs(acc_fp.to_float().item() - acc_true.item()):.6f}")
    print(f"Int-quant error:   {abs(acc_int.item() - acc_true.item()):.6f}")
