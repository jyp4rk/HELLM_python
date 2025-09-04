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
    def forward(
        self,
        input: Tensor,
        R1: Optional[Tensor] = None,
        R2: Optional[Tensor] = None,
        transpose: bool = False,
        frac_bitwidth: int = 20,
        noise_config=None,
    ) -> Tensor:
        # quantize weight
        if R1 is not None:
            dtype = self.weight.dtype
            if not transpose:
                weight = (self.weight.to(torch.float64) @ R1.to(torch.float64)).to(
                    dtype
                )
            else:
                weight = (R1.T.to(torch.float64) @ self.weight.to(torch.float64)).to(
                    dtype
                )
            if R2 is not None:
                # Each head dim = 128 for Llama model
                had_dim = R2.shape[0]
                dtype = weight.dtype
                if transpose:
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
        else:
            weight = self.weight
        if hasattr(self, "quantizer"):
            ## throw error
            raise NotImplementedError("Quantization is not supported")
            dtype = weight.dtype
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(dtype)

        ## divide weight and inputs into integer and fractional parts
        weight_int = weight.floor()
        weight_frac = weight - weight_int

        input_int = input.floor()
        input_frac = input - input_int

        scale = 2 ** frac_bitwidth
        weight_frac_scaled = torch.round(weight_frac * scale)
        input_frac_scaled = torch.round(input_frac * scale)

        term1 = nn.functional.linear(input_int, weight_int, bias=None)
        term2 = nn.functional.linear(input_int, weight_frac_scaled, bias=None) / scale
        term3 = nn.functional.linear(input_frac_scaled, weight_int, bias=None) / scale
        term4 = nn.functional.linear(input_frac_scaled, weight_frac_scaled, bias=None) / (scale * scale)

        output = term1 + term2 + term3 + term4

        if self.bias is not None:
            output = output + self.bias

        ## dim is collapsed dimension when multipying input with weight
        dim = input.shape[-1]

        if noise_config is not None:
            rescale_error = torch.randn_like(output) * noise_config.get("sqrt_Nh") / noise_config.get("delta")

            ## currently we ignore keyswitch_error
            keyswitch_error = torch.zeros_like(output)
            output = output + rescale_error + keyswitch_error

        return output
