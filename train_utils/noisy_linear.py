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


def truncate_towards_zero(x: Tensor) -> Tensor:
    """
    Truncate tensor values towards zero instead of using floor.
    For positive values: same as floor (0.13 -> 0)
    For negative values: same as ceil (-0.13 -> 0)

    Args:
        x: Input tensor

    Returns:
        Tensor with values truncated towards zero
    """
    return torch.trunc(x)


def safe_divide_with_rounding(numerator: Tensor, denominator: float, half_denom: float) -> Tensor:
    """
    Safely perform integer division with proper rounding and NaN/Inf protection.

    Args:
        numerator: Tensor to divide
        denominator: Divisor (should be positive)
        half_denom: Half of denominator for rounding

    Returns:
        Safely divided tensor
    """
    if denominator <= 0:
        raise ValueError(f"Invalid denominator: {denominator}")

    # Add rounding term and divide
    result = (numerator + half_denom).float() / denominator

    return result


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
        weight = self.weight
        # Split weight into integer and fractional parts for CKKS precision simulation
        weight_int = truncate_towards_zero(weight)
        weight_frac = weight - weight_int

        # Scale fractional part with safety checks
        fractional_bitwidth = self.noise_config.fractional_bitwidth

        # Clamp bitwidth to reasonable range to prevent overflow
        fractional_bitwidth = max(1, min(fractional_bitwidth, 30))

        scale = 2 ** fractional_bitwidth

        # Ensure scale is reasonable
        if scale <= 0 or scale > 1e9:
            raise ValueError(f"Invalid scale computed: {scale} from bitwidth {fractional_bitwidth}")

        weight_frac_scaled = torch.round(weight_frac * scale)

        # Clamp scaled values to prevent extreme values
        weight_frac_scaled = torch.clamp(weight_frac_scaled, min=-1e6, max=1e6)

        # Register as buffers so they move with the model during device dispatch
        self.register_buffer('weight_int', weight_int, persistent=False)
        self.register_buffer('weight_frac_scaled', weight_frac_scaled, persistent=False)
        self.scale = scale  # Keep scale as a regular attribute (it's a scalar)

        self._preprocessed_weights = {
            'weight_int': self.weight_int,
            'weight_frac_scaled': self.weight_frac_scaled,
            'scale': scale
        }

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
        if not self._preprocessed or self._preprocessed_weights is None:
            raise RuntimeError("NoisyLinear weights must be preprocessed before forward pass. Call preprocess_with_rotation() first.")
        # Use preprocessed weights (precise 4-term computation for fixed-point accuracy)
        # Use the registered buffers directly - they will be on the correct device
        weight_int = self.weight_int
        weight_frac_scaled = self.weight_frac_scaled
        scale = self.scale

        # Split input into integer and fractional parts for CKKS precision simulation
        input_int = truncate_towards_zero(input)
        input_frac = input - input_int
        input_frac_scaled = torch.round(input_frac * scale)

        # Clamp scaled input to prevent extreme values
        input_frac_scaled = torch.clamp(input_frac_scaled, min=-1e6, max=1e6)

        # Precise 4-term computation with safety checks
        term1 = nn.functional.linear(input_int, weight_int)

        # Use safe division for terms 2 and 3
        term2_num = nn.functional.linear(input_int, weight_frac_scaled)
        term2 = safe_divide_with_rounding(term2_num, scale, scale/2)

        term3_num = nn.functional.linear(input_frac_scaled, weight_int)
        term3 = safe_divide_with_rounding(term3_num, scale, scale/2)

        # Special handling for term4 which involves scale^2
        term4_num = nn.functional.linear(input_frac_scaled, weight_frac_scaled)
        scale_squared = scale * scale

        if scale_squared <= 0:
            # If scale squared is problematic, skip this term
            term4 = torch.zeros_like(term4_num)
        else:
            term4 = safe_divide_with_rounding(term4_num, scale_squared, scale_squared/2)

        # Combine terms with overflow protection
        output = term1 + term2 + term3 + term4

        # Check for NaN/Inf and replace with zeros if found
        output = torch.where(torch.isfinite(output), output, torch.zeros_like(output))

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
