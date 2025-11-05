# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor
from typing import Optional
import warnings


def truncate_towards_zero(x: Tensor) -> Tensor:
    """
    Truncate tensor values towards zero (not floor).
    For positive values: same as floor (0.13 -> 0)
    For negative values: same as ceil (-0.13 -> 0)
    """
    return torch.trunc(x)


class NoisyLinear(nn.Linear):
    """
    Fixed-point linear layer simulating CKKS homomorphic encryption behavior.
    Splits weights and inputs into integer and fractional parts for precision.
    """
    
    def __init__(self, *args, noise_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_config = noise_config
        self._preprocessed = False
        
    def setup_noise(self):
        """
        Preprocess weights for fixed-point arithmetic simulation.
        Must be called before forward passes.
        """
        with torch.no_grad():
            weight = self.weight.data
            
            # Split weight into integer and fractional parts
            weight_int = truncate_towards_zero(weight)
            weight_frac = weight - weight_int
            
            # Get scale from noise config with safety bounds
            fractional_bitwidth = self.noise_config.fractional_bitwidth
            fractional_bitwidth = max(1, min(fractional_bitwidth, 20))  # More conservative limit
            
            scale = 2 ** fractional_bitwidth
            
            # Scale and quantize fractional part
            weight_frac_scaled = torch.round(weight_frac * scale)
            
            # Register as buffers for proper device handling
            self.register_buffer('weight_int', weight_int, persistent=False)
            self.register_buffer('weight_frac_scaled', weight_frac_scaled, persistent=False)
            self.register_buffer('scale_buffer', torch.tensor(scale), persistent=False)
            
            self._preprocessed = True
            
            # Optionally free original weights to save memory
            if hasattr(self.noise_config, 'free_weights') and self.noise_config.free_weights:
                del self.weight
                self.weight = None
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with fixed-point arithmetic simulation.
        """
        if not self._preprocessed:
            raise RuntimeError("NoisyLinear weights must be preprocessed. Call setup_noise() first.")
        
        # Get preprocessed weights and scale
        weight_int = self.weight_int
        weight_frac_scaled = self.weight_frac_scaled
        scale = self.scale_buffer.item()  # Convert to scalar
        
        # Split input into integer and fractional parts
        input_int = truncate_towards_zero(input)
        input_frac = input - input_int
        
        # Scale fractional part
        input_frac_scaled = torch.round(input_frac * scale)
        
        # Four-term computation for fixed-point arithmetic
        # Term 1: int * int (exact)
        term1 = nn.functional.linear(input_int, weight_int, bias=None)
        
        # Term 2: int * frac (scaled)
        term2_raw = nn.functional.linear(input_int, weight_frac_scaled, bias=None)
        # Proper fixed-point division with rounding
        term2 = torch.floor((term2_raw + scale/2) / scale)
        
        # Term 3: frac * int (scaled)
        term3_raw = nn.functional.linear(input_frac_scaled, weight_int, bias=None)
        term3 = torch.floor((term3_raw + scale/2) / scale)
        
        # Term 4: frac * frac (double scaled)
        term4_raw = nn.functional.linear(input_frac_scaled, weight_frac_scaled, bias=None)
        scale_squared = scale * scale
        term4 = torch.floor((term4_raw + scale_squared/2) / scale_squared)
        
        # Combine all terms
        output = term1 + term2 + term3 + term4
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Add CKKS noise if configured
        if hasattr(self.noise_config, 'sqrt_Nh') and self.noise_config.sqrt_Nh > 0:
            # Rescaling noise (from CKKS operations)
            delta = 2 ** self.noise_config.delta_bitwidth
            rescale_noise = torch.randn_like(output) * (self.noise_config.sqrt_Nh / delta)
            output = output + rescale_noise
            
            # Optional: Add key-switching noise
            if hasattr(self.noise_config, 'keyswitch_noise_std'):
                ks_noise = torch.randn_like(output) * self.noise_config.keyswitch_noise_std
                output = output + ks_noise
        
        # Ensure output dtype matches input
        output = output.to(input.dtype)
        
        # Safety check for NaN/Inf
        if not torch.isfinite(output).all():
            warnings.warn("NaN or Inf detected in NoisyLinear output")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, preprocessed={self._preprocessed}'