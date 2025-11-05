# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from src.utils import monkeypatch, quant_utils, utils
from src.utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from src.utils.utils import HadamardTransform
from src.utils.hadamard_utils import (
    get_hadK,
    matmul_hadU_cuda,
    matmul_hadU_cuda_invertible_inverse,
)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(model, args):
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    else:  ## throw error if no rotation path is provided
        raise ValueError("No rotation path provided")
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)


class QKRotationWrapperV2(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for Hadamard transform!"

        self.func = func

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward_v2(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    This is a version without quantization, only applying Hadamard transform.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapperV2, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)


class InverseHadamardWrapper(torch.nn.Module):
    """Wrapper to apply inverse Hadamard transform to the output of a linear layer."""

    def __init__(self, linear_layer):
        super().__init__()
        self.linear_layer = linear_layer
        # Scale to make the transform orthonormal (determinant = 1)
        self.scale = 1.0 / math.sqrt(linear_layer.out_features)

    def forward(self, x):
        # Get the output from the linear layer
        out = self.linear_layer(x)

        # Apply inverse Hadamard transform
        # H_norm = H / sqrt(n) is orthonormal (H_norm^T @ H_norm = I)
        # So H_norm^(-1) = H_norm^T = H / sqrt(n)
        transformed_out = HadamardTransform.apply(out.float()) * self.scale

        return transformed_out.to(out.dtype)


class HadamardOutputWrapper(torch.nn.Module):
    """Wrapper that contains a linear layer with Hadamard-transformed weights.
    This exposes the Hadamard-transformed output for statistics collection."""

    def __init__(self, linear_layer):
        super().__init__()
        self.linear_layer = linear_layer

    def forward(self, x):
        # This is the output with Hadamard-transformed weights
        return self.linear_layer(x)


class InverseHadamardTransform(torch.nn.Module):
    """Module to apply inverse Hadamard transform to restore the original output."""

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        # Get the same hadamard matrix that apply_exact_had_to_linear uses
        from utils.hadamard_utils import get_hadK

        self.had_K, self.K = get_hadK(out_features)

    def forward(self, x):
        # Apply the same inverse Hadamard transform that apply_exact_had_to_linear uses
        dtype = x.dtype
        device = x.device

        if self.had_K is not None:
            # Use the same matmul_hadU_cuda as apply_exact_had_to_linear
            from utils.hadamard_utils import matmul_hadU_cuda

            # Handle 3D tensors (batch, seq_len, features) by reshaping to 2D
            original_shape = x.shape
            if len(original_shape) == 3:
                # Reshape to (batch*seq_len, features)
                x_2d = x.view(-1, original_shape[-1])
            else:
                x_2d = x

            x_cuda = x_2d.float().cuda()
            # For activations, we need to apply the transform to the last dimension (features)
            # The last dimension s
            # hould be divisible by K (e.g., 11008 is divisible by 172)
            transformed_out = matmul_hadU_cuda(x_cuda, self.had_K, self.K)

            # Reshape back to original shape
            if len(original_shape) == 3:
                transformed_out = transformed_out.view(original_shape)

            return transformed_out.to(device=device, dtype=dtype)
        else:
            # Fallback to original method if dimension not supported
            scale = 1.0 / math.sqrt(self.out_features)
            # Move to CUDA for HadamardTransform if not already there
            x_for_transform = x.float().cuda() if not x.is_cuda else x.float()
            transformed_out = HadamardTransform.apply(x_for_transform) * scale
            return transformed_out.to(device=device, dtype=dtype)


class DCTInverseTransform(torch.nn.Module):
    """Module to apply DCT-based inverse transform for gate_proj and up_proj."""

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

        # Generate DCT matrix for the given dimension
        import numpy as np
        import scipy.fftpack

        # Generate DCT-II matrix (orthogonal by construction)
        dct_matrix = scipy.fftpack.dct(np.eye(out_features), axis=0, norm='ortho')
        self.register_buffer('dct_matrix', torch.tensor(dct_matrix, dtype=torch.float32))

    def forward(self, x):
        """Apply DCT inverse transform: y_final = H @ y_hadamard"""
        original_shape = x.shape
        original_device = x.device
        original_dtype = x.dtype

        # Reshape to 2D for matrix multiplication
        x_2d = x.view(-1, self.out_features)

        # Apply DCT inverse transform
        # (H @ x_2d^T)^T = x_2d @ H^T
        result_2d = x_2d @ self.dct_matrix.T.to(x_2d.device).to(x_2d.dtype)

        # Reshape back to original shape
        result = result_2d.view(original_shape)

        return result.to(device=original_device, dtype=original_dtype)


def apply_dct_to_linear(module, out_features):
    """Apply DCT transform to linear layer weights: W' = H^T @ W"""
    import numpy as np
    import scipy.fftpack

    # Generate DCT matrix
    dct_matrix = scipy.fftpack.dct(np.eye(out_features), axis=0, norm='ortho')
    H = torch.tensor(dct_matrix, dtype=torch.float32)

    # Transform weights: W' = H^T @ W
    original_device = module.weight.device
    original_dtype = module.weight.dtype

    # Move to CPU for computation if needed
    weights = module.weight.data.float()
    H_T = H.T.to(weights.device)

    # Apply transform
    module.weight.data = (H_T @ weights).to(device=original_device, dtype=original_dtype)


class ActQuantWrapperWithInverseHadamard(torch.nn.Module):
    """Wrapper that preserves all attributes of the original module but adds inverse Hadamard transform."""

    def __init__(self, original_module, out_features):
        super().__init__()
        self.original_module = original_module
        self.inverse_hadamard = InverseHadamardTransform(out_features)

    def forward(self, x):
        # Forward through original module (e.g., ActQuantWrapper with Hadamard-transformed weights)
        out = self.original_module(x)
        # Apply inverse Hadamard transform
        return self.inverse_hadamard(out)

    def __getattr__(self, name):
        # Delegate all attribute access to the original module
        # This ensures .weight, .module, etc. work correctly
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_module, name)

    def __setattr__(self, name, value):
        # Handle special attributes for this wrapper
        if name in ["original_module", "inverse_hadamard"]:
            super().__setattr__(name, value)
        else:
            # Delegate attribute setting to original module
            if hasattr(self, "original_module"):
                setattr(self.original_module, name, value)
            else:
                super().__setattr__(name, value)


def apply_hadamard_transform_to_input_projections(model):
    """
    Apply orthogonal transforms to q_proj, k_proj, gate_proj, and up_proj weights.

    - q_proj, k_proj: Use structured Hadamard transform with InverseHadamardTransform
    - gate_proj, up_proj: Use DCT transform with DCTInverseTransform (supports 11008 dim)

    For a linear layer y = Wx:
    - We transform weights: W' = H^T @ W where H is orthogonal
    - Keep the existing wrapper (e.g., ActQuantWrapper) with transformed weights
    - Append inverse transform to restore: y = H @ y_transformed

    All transforms are orthonormal (determinant = 1).
    Statistics collectors can hook into both the original wrapper and inverse transform.
    """

    def get_linear_module(module):
        """Extract the underlying Linear module from potentially wrapped modules."""
        if isinstance(module, torch.nn.Linear):
            return module
        elif hasattr(module, "module") and isinstance(module.module, torch.nn.Linear):
            # This handles ActQuantWrapper which has the Linear layer in .module
            return module.module
        elif hasattr(module, "linear_layer"):
            return module.linear_layer
        elif hasattr(module, "layer"):
            return module.layer
        else:
            # Search for any Linear module in the attributes
            for attr_name in dir(module):
                if not attr_name.startswith("_"):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, torch.nn.Linear):
                        return attr
            raise ValueError(f"Could not find Linear module in {type(module)}")

    for layer_idx, layer in enumerate(model.model.layers):
        # Apply Hadamard to q_proj weights
        if hasattr(layer.self_attn, "q_proj"):
            # Get the underlying Linear module (inside ActQuantWrapper)
            q_linear = get_linear_module(layer.self_attn.q_proj)
            # Store original weights for debugging
            original_weight = q_linear.weight.data.clone()
            # Apply Hadamard transform to weights
            apply_exact_had_to_linear(q_linear, had_dim=-1, output=True)
            # Use nn.Sequential with HadamardOutputWrapper and InverseHadamardTransform
            layer.self_attn.q_proj = torch.nn.Sequential(
                HadamardOutputWrapper(layer.self_attn.q_proj),
                InverseHadamardTransform(q_linear.out_features),
            )

        # Apply Hadamard to k_proj weights
        if hasattr(layer.self_attn, "k_proj"):
            k_linear = get_linear_module(layer.self_attn.k_proj)
            # Store original weights for debugging
            original_weight = k_linear.weight.data.clone()
            apply_exact_had_to_linear(k_linear, had_dim=-1, output=True)
            # Use nn.Sequential with HadamardOutputWrapper and InverseHadamardTransform
            layer.self_attn.k_proj = torch.nn.Sequential(
                HadamardOutputWrapper(layer.self_attn.k_proj),
                InverseHadamardTransform(k_linear.out_features),
            )

        # # Apply DCT to gate_proj weights
        # if hasattr(layer.mlp, 'gate_proj'):
        #     gate_linear = get_linear_module(layer.mlp.gate_proj)
        #     ## check gate_linear has bias
        #     if gate_linear.bias is not None:
        #         raise RuntimeError(
        #             "DCT transform does not support bias in gate_proj. "
        #             "Please remove the bias before applying DCT transform."
        #         )
        #     # Apply DCT transform to weights
        #     apply_dct_to_linear(gate_linear, gate_linear.out_features)
        #     layer.mlp.gate_proj = torch.nn.Sequential(
        #         HadamardOutputWrapper(layer.mlp.gate_proj),
        #         DCTInverseTransform(gate_linear.out_features)
        #     )

        # # Apply DCT to up_proj weights
        # if hasattr(layer.mlp, "up_proj"):
        #     up_linear = get_linear_module(layer.mlp.up_proj)
        #     ## check up_linear has bias
        #     if up_linear.bias is not None:
        #         raise RuntimeError(
        #             "DCT transform does not support bias in up_proj. "
        #             "Please remove the bias before applying DCT transform."
        #         )
        #     # Apply DCT transform to weights
        #     apply_dct_to_linear(up_linear, up_linear.out_features)
        #     layer.mlp.up_proj = torch.nn.Sequential(
        #         HadamardOutputWrapper(layer.mlp.up_proj),
        #         DCTInverseTransform(up_linear.out_features),
        #     )

    return model
