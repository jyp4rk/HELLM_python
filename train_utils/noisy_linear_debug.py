# coding=utf-8
# Debug version of NoisyLinear to identify NaN sources

import torch
import torch.nn as nn
from torch._tensor import Tensor
from typing import Optional
import warnings


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


def check_for_nans_infs(tensor, name, warn=True):
    """Check tensor for NaN/Inf values and optionally warn"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        if warn:
            print(f"⚠️  {name}: NaN={has_nan}, Inf={has_inf}, shape={tensor.shape}")
            print(f"   Min={tensor.min():.6f}, Max={tensor.max():.6f}")
            if has_nan:
                nan_count = torch.isnan(tensor).sum()
                print(f"   NaN count: {nan_count}")
            if has_inf:
                inf_count = torch.isinf(tensor).sum()
                print(f"   Inf count: {inf_count}")
        return True
    return False


class NoisyLinear(nn.Linear):
    def __init__(self, *args, noise_config, **kwargs):
        super().__init__(*args, **kwargs)
        # Preprocessing state for weight splitting optimization
        self._preprocessed_weights = None
        self._preprocessed = False
        self.noise_config = noise_config
        self.debug_enabled = True

    def setup_noise(self):
        """
        Preprocess weights with rotation and remove original weights for memory optimization.
        This must be called before forward passes.
        """
        weight = self.weight
        if self.debug_enabled:
            print(f"🔧 Setting up noise for layer {self.__class__.__name__}")
            check_for_nans_infs(weight, "Original weight")

        # Split weight into integer and fractional parts for CKKS precision simulation
        weight_int = truncate_towards_zero(weight)
        weight_frac = weight - weight_int

        if self.debug_enabled:
            check_for_nans_infs(weight_int, "Weight integer part")
            check_for_nans_infs(weight_frac, "Weight fractional part")
            print(f"   Weight frac range: [{weight_frac.min():.6f}, {weight_frac.max():.6f}]")

        # Scale fractional part
        try:
            scale = 2 ** self.noise_config.fractional_bitwidth
            if self.debug_enabled:
                print(f"   Fractional bitwidth: {self.noise_config.fractional_bitwidth}")
                print(f"   Scale: {scale}")

            if scale <= 0:
                raise ValueError(f"Invalid scale: {scale}")

            weight_frac_scaled = torch.round(weight_frac * scale)

            if self.debug_enabled:
                check_for_nans_infs(weight_frac_scaled, "Weight frac scaled")
                print(f"   Weight frac scaled range: [{weight_frac_scaled.min():.6f}, {weight_frac_scaled.max():.6f}]")

        except Exception as e:
            print(f"❌ Error in weight scaling: {e}")
            raise

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
            raise RuntimeError("NoisyLinear weights must be preprocessed before forward pass. Call setup_noise() first.")

        if self.debug_enabled:
            check_for_nans_infs(input, "Input")

        # Use preprocessed weights (precise 4-term computation for fixed-point accuracy)
        # Use the registered buffers directly - they will be on the correct device
        weight_int = self.weight_int
        weight_frac_scaled = self.weight_frac_scaled
        scale = self.scale

        if self.debug_enabled:
            print(f"🔍 Forward pass - Scale: {scale}")
            check_for_nans_infs(weight_int, "Weight int (from buffer)")
            check_for_nans_infs(weight_frac_scaled, "Weight frac scaled (from buffer)")

        # Split input into integer and fractional parts for CKKS precision simulation
        try:
            input_int = truncate_towards_zero(input)
            input_frac = input - input_int

            if self.debug_enabled:
                check_for_nans_infs(input_int, "Input integer part")
                check_for_nans_infs(input_frac, "Input fractional part")
                print(f"   Input frac range: [{input_frac.min():.6f}, {input_frac.max():.6f}]")

            input_frac_scaled = torch.round(input_frac * scale)

            if self.debug_enabled:
                check_for_nans_infs(input_frac_scaled, "Input frac scaled")

        except Exception as e:
            print(f"❌ Error in input processing: {e}")
            raise

        # Precise 4-term computation maintaining int64 precision where possible
        try:
            if self.debug_enabled:
                print("🧮 Computing 4 terms...")

            term1 = nn.functional.linear(input_int, weight_int)
            if self.debug_enabled:
                check_for_nans_infs(term1, "Term1 (int*int)")

            # Check for division by zero
            if scale == 0:
                raise ValueError("Scale is zero - cannot perform division")

            term2_num = nn.functional.linear(input_int, weight_frac_scaled) + scale//2
            if self.debug_enabled:
                check_for_nans_infs(term2_num, "Term2 numerator")
            term2 = term2_num // scale
            if self.debug_enabled:
                check_for_nans_infs(term2, "Term2 (int*frac)")

            term3_num = nn.functional.linear(input_frac_scaled, weight_int) + scale//2
            if self.debug_enabled:
                check_for_nans_infs(term3_num, "Term3 numerator")
            term3 = term3_num // scale
            if self.debug_enabled:
                check_for_nans_infs(term3, "Term3 (frac*int)")

            # Most critical: check scale*scale
            scale_squared = scale * scale
            if scale_squared == 0:
                raise ValueError("Scale squared is zero - cannot perform division")

            term4_num = nn.functional.linear(input_frac_scaled, weight_frac_scaled) + scale_squared//2
            if self.debug_enabled:
                check_for_nans_infs(term4_num, "Term4 numerator")
                print(f"   Scale squared: {scale_squared}")
            term4 = term4_num // scale_squared
            if self.debug_enabled:
                check_for_nans_infs(term4, "Term4 (frac*frac)")

            # Combine terms
            output = (term1 + term2 + term3 + term4).to(input.dtype)

            if self.debug_enabled:
                check_for_nans_infs(output, "Final output")
                print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")

        except Exception as e:
            print(f"❌ Error in 4-term computation: {e}")
            print(f"   Scale: {scale}")
            print(f"   Input shape: {input.shape}")
            print(f"   Weight shapes: int={weight_int.shape}, frac_scaled={weight_frac_scaled.shape}")
            raise

        return output
