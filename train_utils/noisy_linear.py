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


class NoisyLinear(nn.Linear):
    def __init__(self, *args, rotation_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Preprocessing state for weight splitting optimization
        self._preprocessed_weights = None
        self._preprocessed = False
        self.rotation_config = rotation_config

    def _apply_rotation(self, R1, R2):
        """Apply rotation matrices to weight."""
        dtype = self.weight.dtype
        device = self.weight.device

        if R1 is not None:
            R1 = R1.to(device)
        if R2 is not None:
            R2 = R2.to(device)
        if self.rotation_config == 'r1w':
            R1_transpose = False
            R2 = None
        elif self.rotation_config == 'wr1t':
            R1_transpose = True
            R2 = None
        elif self.rotation_config == 'r1tr2':
            R1_transpose = False
            R2_transpose = True
        elif self.rotation_config == 'r2wr1t':
            R1_transpose = True
            R2_transpose = False
        else:
            raise ValueError(f"Unknown rotation_config: {self.rotation_config}")

        if R1 is not None:
            if R1_transpose:
                # For 'r1tw': R1.T @ weight (e.g., (4096, 4096) @ (4096, hidden_size))
                weight = (R1.T.to(torch.float64) @ self.weight.to(torch.float64)).to(dtype)
            else:
                # For 'wr1': weight @ R1 (e.g., (4096, 11008) @ (4096, 4096) = (4096, 11008))
                weight = (self.weight.to(torch.float64) @ R1.to(torch.float64)).to(dtype)

            if R2 is not None:
                # Each head dim = 128 for Llama model
                had_dim = R2.shape[0]
                if not R2_transpose:
                    W_ = weight
                    init_shape = W_.shape
                    temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(init_shape)
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(transposed_shape).t()
                weight = weight.to(dtype)

        return weight

    def preprocess_with_rotation(self, noise_config, R1=None, R2=None):
        """
        Preprocess weights with rotation and remove original weights for memory optimization.
        This must be called before forward passes.

        Args:
            noise_config: Noise configuration for CKKS operations
            R1: First rotation matrix (optional)
            R2: Second rotation matrix (optional)
            transpose: Whether to transpose during rotation
        """
        # Apply rotation if provided
        if R1 is not None:
            weight = self._apply_rotation(R1, R2)
        else:
            weight = self.weight

        # Split weight into integer and fractional parts for CKKS precision simulation
        weight_int = weight.floor()
        weight_frac = weight - weight_int

        # Scale fractional part
        scale = 2 ** noise_config.get("fractional_bitwidth", 27)
        weight_frac_scaled = torch.round(weight_frac * scale)

        self._preprocessed_weights = {
            'weight_int': weight_int,
            'weight_frac_scaled': weight_frac_scaled,
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
        weight_int = self._preprocessed_weights['weight_int']
        weight_frac_scaled = self._preprocessed_weights['weight_frac_scaled']
        scale = self._preprocessed_weights['scale']

        # Split input into integer and fractional parts for CKKS precision simulation
        input_int = input.floor()
        input_frac = input - input_int
        input_frac_scaled = torch.round(input_frac * scale)

        # Precise 4-term computation maintaining int64 precision where possible
        term1 = nn.functional.linear(input_int, weight_int)
        term2 = (nn.functional.linear(input_int, weight_frac_scaled) + scale//2) // scale
        term3 = (nn.functional.linear(input_frac_scaled, weight_int) + scale//2) // scale
        term4 = (nn.functional.linear(input_frac_scaled, weight_frac_scaled) + scale*scale//2) // (scale * scale)

        output = (term1 + term2 + term3 + term4).to(input.dtype)

        if self.bias is not None:
            output = output + self.bias

        # ## dim is collapsed dimension when multipying input with weight
        # dim = input.shape[-1]

        # if noise_config is not None:
        #     # Calculate delta from delta_bitwidth if needed
        #     delta = noise_config.get("delta", 2 ** noise_config.get("delta_bitwidth", 42))
        #     rescale_error = torch.randn_like(output) * noise_config.get("sqrt_Nh") / delta

        #     ## currently we ignore keyswitch_error
        #     keyswitch_error = torch.zeros_like(output)
        #     output = output + rescale_error + keyswitch_error

        return output
