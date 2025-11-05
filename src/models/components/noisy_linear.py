# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor
##import Optional
from typing import Optional

def quantize_to_fixed_point(x: Tensor, scale: float, inverse: bool = False) -> Tensor:
    """
    Quantize tensor to fixed-point representation.

    Args:
        x: Input tensor
        scale: Scale factor (2^fractional_bits)
        inverse: If True, dequantize from fixed-point

    Returns:
        Quantized tensor
    """
    if inverse:
        # Dequantize: divide by scale
        return x / scale
    else:
        # Quantize: multiply by scale and round
        return torch.round(x * scale)


class NoisyLinear(nn.Linear):
    def __init__(self, *args, noise_config, **kwargs):
        super().__init__(*args, **kwargs)
        # Preprocessing state for weight splitting optimization
        self._preprocessed_weights = None
        self._preprocessed = False
        self.noise_config = noise_config

    def setup_noise(self):
        """
        Preprocess weights with rotation and remove original weights for memory optimization.
        This must be called before forward passes.

        Args:
            noise_config: Noise configuration for CKKS operations
            R1: First rotation matrix (optional)
            R2: Second rotation matrix (optional)
            transpose: Whether to transpose during rotation
        """
        dtype = self.weight.dtype
        weight = self.weight.float()
        # Split weight into integer and fractional parts for CKKS precision simulation
        weight_int = torch.trunc(weight)
        weight_frac = weight - weight_int

        # Get fractional bitwidth and compute scale
        fractional_bitwidth = self.noise_config.fractional_bitwidth

        # Use truncation approach for large bitwidths
        scale = 2 ** fractional_bitwidth
        self.scale = scale

        # Truncate small values instead of scaling large
        weight_frac = torch.where(
            torch.abs(weight_frac) < 1/self.scale,
            torch.zeros_like(weight_frac),
            weight_frac
        )

        # Quantize fractional part
        weight_frac_truncated = torch.round(weight_frac * scale) / scale
        weight = weight_int + weight_frac_truncated

        # Update the weight parameter data in-place to preserve parameter status
        with torch.no_grad():
            self.weight.data = weight.to(dtype)

        # Register as buffers so they move with the model during device dispatch
        # self.register_buffer('weight_int', weight_int, persistent=False)
        # self.register_buffer('weight_frac_truncated', weight_frac_truncated, persistent=False)
        # self.register_buffer('scale', torch.tensor(scale), persistent=False)

        # self._preprocessed_weights = {
        #     'weight_int': self.weight_int,
        #     'weight_frac_truncated': self.weight_frac_truncated,
        #     'scale': scale
        # }

        self._preprocessed = True

        # Store only the split weights to save memory
        # del self.weight
        # self.weight = None  # Set to None to indicate preprocessing is done

    def forward(
        self,
        input: Tensor,
        R1: Optional[Tensor] = None,
        R2: Optional[Tensor] = None,
        transpose: Optional[bool] = False,
        noise_config=None,
    ) -> Tensor:
        # Simple approach: weights must be preprocessed before forward
        if not self._preprocessed:
            raise RuntimeError("NoisyLinear weights must be preprocessed before forward pass. Call setup_noise() first.")
        # Use preprocessed weights (precise 4-term computation for fixed-point accuracy)
        # Use the registered buffers directly - they will be on the correct device
        # weight_int = self.weight_int
        # weight_frac_scaled = self.weight_frac_scaled
        # scale = self.scale
        dtype = input.dtype
        input = input.float()
        input_int = torch.trunc(input)
        input_frac = input - input_int

        input_frac = torch.where(
            torch.abs(input_frac) < 1/self.scale,
            torch.zeros_like(input_frac),
            input_frac
        )

        input_frac_truncated = torch.round(input_frac * self.scale) / self.scale

        input = input_int + input_frac_truncated
        input = input.to(dtype)

        output = nn.functional.linear(input, self.weight)

        # # Precise 4-term computation for fixed-point arithmetic
        # # Term 1: int * int (exact)
        # term1 = nn.functional.linear(input_int, weight_int)

        # # Term 2: int * frac_scaled / scale
        # term2_scaled = nn.functional.linear(input_int, weight_frac_scaled)
        # term2 = quantize_to_fixed_point(term2_scaled, scale, inverse=True)

        # # Term 3: frac_scaled * int / scale
        # term3_scaled = nn.functional.linear(input_frac_scaled, weight_int)
        # term3 = quantize_to_fixed_point(term3_scaled, scale, inverse=True)

        # # Term 4: frac_scaled * frac_scaled / scale^2
        # term4_scaled = nn.functional.linear(input_frac_scaled, weight_frac_scaled)
        # # Two-step dequantization for term4
        # term4 = quantize_to_fixed_point(term4_scaled, scale, inverse=True)
        # term4 = quantize_to_fixed_point(term4, scale, inverse=True)

        # Combine terms with overflow protection
        # output = term1 + term2 + term3 + term4

        # Check for NaN/Inf and replace with zeros if found
        # output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

        output = output.to(input.dtype)

        # if self.bias is not None:
        #     output = output + self.bias

        # if noise_config is not None:
        #     # Calculate delta from delta_bitwidth if needed
        #     delta = noise_config.delta_bitwidth
        #     rescale_error = torch.randn_like(output) * noise_config.sqrt_Nh / delta

        #     ## currently we ignore keyswitch_error
        #     keyswitch_error = torch.zeros_like(output)
        #     output = output + rescale_error + keyswitch_error

        # Add bias if present (was commented out in original)
        if self.bias is not None:
            output = output + self.bias

        return output
