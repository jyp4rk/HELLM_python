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


def quantize_to_precision(x: Tensor, precision_bits: int) -> Tensor:
    """
    Quantize tensor to fixed-point precision by truncating values below threshold.
    Instead of scaling up, we truncate to simulate limited precision.
    
    Args:
        x: Input tensor
        precision_bits: Number of fractional bits to preserve
        
    Returns:
        Quantized tensor with limited precision
    """
    # Calculate the precision threshold (smallest representable value)
    precision = 2 ** (-precision_bits)
    
    # Round to the nearest multiple of precision
    quantized = torch.round(x / precision) * precision
    
    # Optional: Zero out values smaller than precision to avoid numerical issues
    mask = torch.abs(quantized) < precision
    quantized = torch.where(mask, torch.zeros_like(quantized), quantized)
    
    return quantized


class NoisyLinear(nn.Linear):
    """
    Fixed-point linear layer simulating CKKS homomorphic encryption behavior.
    Uses truncation instead of scaling to maintain numerical stability.
    """
    
    def __init__(self, *args, noise_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_config = noise_config
        self._preprocessed = False
        
    def setup_noise(self):
        """
        Preprocess weights for fixed-point arithmetic simulation.
        Uses truncation approach instead of scaling.
        """
        with torch.no_grad():
            weight = self.weight.data
            
            # Get precision bits from config
            precision_bits = int(self.noise_config.fractional_bitwidth)
            precision_bits = max(4, min(precision_bits, 16))  # Reasonable range
            
            # Split weight into integer and fractional parts
            weight_int = truncate_towards_zero(weight)
            weight_frac = weight - weight_int
            
            # Quantize fractional part to limited precision (truncate small values)
            weight_frac_quantized = quantize_to_precision(weight_frac, precision_bits)
            
            # Register as buffers for proper device handling
            self.register_buffer('weight_int', weight_int, persistent=False)
            self.register_buffer('weight_frac', weight_frac_quantized, persistent=False)
            self.register_buffer('precision_bits', torch.tensor(precision_bits), persistent=False)
            
            self._preprocessed = True
            
            # Store precision for forward pass
            self.precision = 2 ** (-precision_bits)
            
            # Optionally free original weights to save memory
            if hasattr(self.noise_config, 'free_weights') and self.noise_config.free_weights:
                del self.weight
                self.weight = None
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass with fixed-point arithmetic simulation using truncation.
        """
        if not self._preprocessed:
            raise RuntimeError("NoisyLinear weights must be preprocessed. Call setup_noise() first.")
        
        # Get preprocessed weights
        weight_int = self.weight_int
        weight_frac = self.weight_frac
        precision_bits = self.precision_bits.item()
        
        # Split input into integer and fractional parts
        input_int = truncate_towards_zero(input)
        input_frac = input - input_int
        
        # Quantize input fractional part to limited precision
        input_frac = quantize_to_precision(input_frac, precision_bits)
        
        # Four-term computation for fixed-point arithmetic
        # All operations maintain the precision level without scaling
        
        # Term 1: int * int (exact)
        term1 = nn.functional.linear(input_int, weight_int, bias=None)
        
        # Term 2: int * frac (limited precision)
        term2 = nn.functional.linear(input_int, weight_frac, bias=None)
        term2 = quantize_to_precision(term2, precision_bits)
        
        # Term 3: frac * int (limited precision)
        term3 = nn.functional.linear(input_frac, weight_int, bias=None)
        term3 = quantize_to_precision(term3, precision_bits)
        
        # Term 4: frac * frac (double precision loss)
        term4 = nn.functional.linear(input_frac, weight_frac, bias=None)
        # This term has double precision loss, so we apply more aggressive truncation
        term4 = quantize_to_precision(term4, precision_bits // 2)
        
        # Combine all terms
        output = term1 + term2 + term3 + term4
        
        # Final quantization to maintain precision
        output = quantize_to_precision(output, precision_bits)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        # Add CKKS noise if configured
        if hasattr(self.noise_config, 'sqrt_Nh') and self.noise_config.sqrt_Nh > 0:
            # Rescaling noise (from CKKS operations)
            # Noise is added at the precision level
            noise_std = self.noise_config.sqrt_Nh * self.precision
            rescale_noise = torch.randn_like(output) * noise_std
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
        precision_bits = self.precision_bits.item() if hasattr(self, 'precision_bits') else 'N/A'
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, precision_bits={precision_bits}'