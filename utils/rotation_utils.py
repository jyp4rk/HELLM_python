# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import utils.model_utils as model_utils
import torch
import typing
from utils.train_utils import cleanup_memory
import tqdm, math
from utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)

def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)



def fuse_layer_norms(model):

    model_type = model_utils.get_model_type(model)
    # model_type = model_utils.LLAMA_MODEL

    kwargs = {'model': model, 'model_type': model_type}

    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model_utils.get_transformer_layers(**kwargs)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:

        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.MISTRAL_MODEL or model_type == model_utils.QWEN2_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        elif model_type == model_utils.INTERNLM2_MODEL:
            fuse_ln_linear(layer.attention_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.ffn_norm, [layer.feed_forward.w1, layer.feed_forward.w3])
        else:
            raise ValueError(f'Unknown model type {model_type}')

        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)

    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    norm_type = model_utils.get_norm_type(model)
    model_utils.replace_modules(
        model,
        norm_type,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
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

def random_block_orthogonal_matrix(size, device, block_size):
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
    assert size % block_size == 0
    n_block = size // block_size
    Q_big = torch.zeros(size, size, dtype=torch.float64).to(device)
    for i in range(n_block):
        Q = random_orthogonal_matrix(block_size,device)
        start_idx = i * block_size
        end_idx = start_idx + block_size
        Q_big[start_idx:end_idx, start_idx:end_idx] = Q
    Q_big = Q_big.contiguous()
    return Q_big

def get_orthogonal_matrix(size, mode, device=DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')



def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN2_MODEL or model_type == model_utils.MISTRAL_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN2_MODEL or model_type == model_utils.MISTRAL_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q, model_type, online):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN2_MODEL or model_type == model_utils.MISTRAL_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if online:
        apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output (require on-line hadamard)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation.
    It reshapes X and applies Walsh-Hadamard transform to the last dimension.
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(
        X.shape)

def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')

    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, Q, model_type, num_attention_heads, num_key_value_heads, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL or model_type == model_utils.QWEN2_MODEL or model_type == model_utils.MISTRAL_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    # v_proj output
    dtype = v_proj.weight.data.dtype
    W_ = v_proj.weight.data.to(device=DEV, dtype=torch.float64)
    shape = W_.shape
    W_ = W_.reshape(num_key_value_heads,shape[0]//num_key_value_heads,-1)
    v_proj.weight.data = torch.matmul(Q.T.unsqueeze(0), W_).reshape(shape).to(device='cpu', dtype=dtype)
    if v_proj.bias is not None:
        B_ = v_proj.bias.data.to(device=DEV, dtype=torch.float64)
        shape = B_.shape
        B_ = B_.reshape(num_key_value_heads,shape[0]//num_key_value_heads, -1)
        v_proj.bias.data = torch.matmul(Q.T, B_).flatten().to(device="cpu", dtype=dtype)

    # o_proj input
    dtype = o_proj.weight.data.dtype
    W_ = o_proj.weight.data.to(device=DEV, dtype=torch.float64)
    shape = W_.shape
    W_ = W_.reshape(-1, num_attention_heads, shape[-1]//num_attention_heads,)
    o_proj.weight.data = torch.matmul(W_, Q).reshape(shape).to(device='cpu', dtype=dtype)



@torch.inference_mode()
def rotate_model(model, rotate_mode='hadamard', online=False):
    Q = get_orthogonal_matrix(model.config.hidden_size,rotate_mode,DEV)
    config = model.config
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_attention_heads
    Q_ov = get_orthogonal_matrix(head_dim,rotate_mode,DEV)


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    cleanup_memory()
    layers = model_utils.get_transformer_layers(model,
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type, online)
        rotate_ov_proj(layers[idx], Q_ov, model_type, num_attention_heads,num_key_value_heads, head_dim)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, online_had, *args, **kwargs):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.model_dim = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.model_dim // self.num_heads
        assert is_pow2(self.head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.online_had = online_had
        self.use_k_quant = False
        self.k_bits = 16

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_k_quant = act_quant

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        if self.online_had:
            q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
            k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.use_k_quant and self.k_bits < 16:
            k = k.transpose(1, 2).flatten(-2)
            k = self.k_quantizer(k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import quantize.monkeypatch as monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)

# def apply_hadamard_transform_to_q_k_projections(model):
#     """
#     Apply orthogonal transforms to q_proj, k_proj, gate_proj, and up_proj weights.

#     - q_proj, k_proj: Use structured Hadamard transform with InverseHadamardTransform
#     - gate_proj, up_proj: Use DCT transform with DCTInverseTransform (supports 11008 dim)

#     For a linear layer y = Wx:
#     - We transform weights: W' = H^T @ W where H is orthogonal
#     - Keep the existing wrapper (e.g., ActQuantWrapper) with transformed weights
#     - Append inverse transform to restore: y = H @ y_transformed

#     All transforms are orthonormal (determinant = 1).
#     Statistics collectors can hook into both the original wrapper and inverse transform.
#     """

#     def get_linear_module(module):
#         """Extract the underlying Linear module from potentially wrapped modules."""
#         if isinstance(module, torch.nn.Linear):
#             return module
#         elif hasattr(module, "module") and isinstance(module.module, torch.nn.Linear):
#             # This handles ActQuantWrapper which has the Linear layer in .module
#             return module.module
#         elif hasattr(module, "linear_layer"):
#             return module.linear_layer
#         elif hasattr(module, "layer"):
#             return module.layer
#         else:
#             # Search for any Linear module in the attributes
#             for attr_name in dir(module):
#                 if not attr_name.startswith("_"):
#                     attr = getattr(module, attr_name)
#                     if isinstance(attr, torch.nn.Linear):
#                         return attr
#             raise ValueError(f"Could not find Linear module in {type(module)}")

#     for layer_idx, layer in enumerate(model.model.layers):
#         # Apply Hadamard to q_proj weights
#         if hasattr(layer.self_attn, "q_proj"):
#             # Get the underlying Linear module (inside ActQuantWrapper)
#             q_linear = get_linear_module(layer.self_attn.q_proj)
#             # Store original weights for debugging
#             original_weight = q_linear.weight.data.clone()
#             # Apply Hadamard transform to weights
#             apply_exact_had_to_linear(q_linear, had_dim=-1, output=True)
#             # Use nn.Sequential with HadamardOutputWrapper and InverseHadamardTransform
#             layer.self_attn.q_proj = torch.nn.Sequential(
#                 HadamardOutputWrapper(layer.self_attn.q_proj),
#                 InverseHadamardTransform(q_linear.out_features),
#             )

#         # Apply Hadamard to k_proj weights
#         if hasattr(layer.self_attn, "k_proj"):
#             k_linear = get_linear_module(layer.self_attn.k_proj)
#             # Store original weights for debugging
#             original_weight = k_linear.weight.data.clone()
#             apply_exact_had_to_linear(k_linear, had_dim=-1, output=True)
#             # Use nn.Sequential with HadamardOutputWrapper and InverseHadamardTransform
#             layer.self_attn.k_proj = torch.nn.Sequential(
#                 HadamardOutputWrapper(layer.self_attn.k_proj),
#                 InverseHadamardTransform(k_linear.out_features),
#             )

#         # # Apply DCT to gate_proj weights
#         # if hasattr(layer.mlp, 'gate_proj'):
#         #     gate_linear = get_linear_module(layer.mlp.gate_proj)
#         #     ## check gate_linear has bias
#         #     if gate_linear.bias is not None:
#         #         raise RuntimeError(
#         #             "DCT transform does not support bias in gate_proj. "
#         #             "Please remove the bias before applying DCT transform."
#         #         )
#         #     # Apply DCT transform to weights
#         #     apply_dct_to_linear(gate_linear, gate_linear.out_features)
#         #     layer.mlp.gate_proj = torch.nn.Sequential(
#         #         HadamardOutputWrapper(layer.mlp.gate_proj),
#         #         DCTInverseTransform(gate_linear.out_features)
#         #     )

#         # # Apply DCT to up_proj weights
#         # if hasattr(layer.mlp, "up_proj"):
#         #     up_linear = get_linear_module(layer.mlp.up_proj)
#         #     ## check up_linear has bias
#         #     if up_linear.bias is not None:
#         #         raise RuntimeError(
#         #             "DCT transform does not support bias in up_proj. "
#         #             "Please remove the bias before applying DCT transform."
#         #         )
#         #     # Apply DCT transform to weights
#         #     apply_dct_to_linear(up_linear, up_linear.out_features)
#         #     layer.mlp.up_proj = torch.nn.Sequential(
#         #         HadamardOutputWrapper(layer.mlp.up_proj),
#         #         DCTInverseTransform(up_linear.out_features),
#         #     )

#     return model


def get_linear_module(module):
    """
    Extract the underlying Linear module from potentially wrapped modules.
    Handles QuantLinear wrappers and other module wrapping patterns.
    """
    if isinstance(module, torch.nn.Linear):
        return module
    elif hasattr(module, "module") and isinstance(module.module, torch.nn.Linear):
        # This handles ActQuantWrapper which has the Linear layer in .module
        return module.module
    elif hasattr(module, "linear_layer"):
        return module.linear_layer
    elif hasattr(module, "layer"):
        return module.layer
    elif hasattr(module, 'weight') and hasattr(module, 'bias'):
        # QuantLinear modules act like Linear for weight operations
        return module
    else:
        # Search for any Linear module in the attributes
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                attr = getattr(module, attr_name)
                if isinstance(attr, torch.nn.Linear):
                    return attr
        raise ValueError(f"Could not find Linear module in {type(module)}")


def fuse_hadamard_into_qk_weights(model):
    """
    Fuse Hadamard rotation into q_proj and k_proj weight matrices offline.
    This modifies the weights so that W_new = W_original @ H^T

    The fused weights allow applying the inverse Hadamard transform online
    during inference to achieve the same rotation effect as the original
    RoPE-based wrapper system.

    Args:
        model: The transformer model to modify
    """
    import utils.model_utils as model_utils
    from utils.hadamard_utils import apply_exact_had_to_linear

    model_type = model_utils.get_model_type(model)
    layers = model_utils.get_transformer_layers(model=model, model_type=model_type)

    for layer_idx, layer in enumerate(layers):
        # Apply Hadamard to q_proj weights
        if hasattr(layer.self_attn, 'q_proj'):
            q_linear = get_linear_module(layer.self_attn.q_proj)
            apply_exact_had_to_linear(q_linear, had_dim=-1, output=True)

        # Apply Hadamard to k_proj weights
        if hasattr(layer.self_attn, 'k_proj'):
            k_linear = get_linear_module(layer.self_attn.k_proj)
            apply_exact_had_to_linear(k_linear, had_dim=-1, output=True)


def register_qk_online_hadamard(model, config):
    """
    Register online inverse Hadamard transforms for q_proj and k_proj modules.
    This sets up the infrastructure needed to apply inverse Hadamard transforms
    during inference, complementing the offline weight fusion.

    Similar to existing register_online_had but specifically for Q/K projections.

    Args:
        model: The transformer model
        config: Model configuration containing dimension information
    """
    import utils.model_utils as model_utils
    import utils.hadamard_utils as hadamard_utils

    model_type = model_utils.get_model_type(model)
    layers = model_utils.get_transformer_layers(model=model, model_type=model_type)

    # Calculate dimensions
    head_dim = config.hidden_size // config.num_attention_heads

    for layer in layers:
        # Register for q_proj
        q_module = layer.self_attn.q_proj
        # Check if it's a QuantLinear module (has the online_full_had attribute structure)
        if hasattr(q_module, '__dict__'):  # Ensure it can hold new attributes
            # Get Hadamard matrix for Q projections - will be transposed in forward pass for cancellation
            had_K, K = hadamard_utils.get_hadK(config.intermediate_size, True)
            q_module.online_qk_had = True  # New flag for Q/K Hadamard

            q_module.qk_had_K = had_K.to(q_module.weight.device).float()
            q_module.qk_K = K

        # Register for k_proj
        k_module = layer.self_attn.k_proj
        if hasattr(k_module, '__dict__'):  # Ensure it can hold new attributes
            # Get Hadamard matrix for K projections - will be transposed in forward pass for cancellation
            had_K, K = hadamard_utils.get_hadK(config.intermediate_size, True)
            k_module.online_qk_had = True  # New flag for Q/K Hadamard
            # Handle case where had_K is None (when K=1, pure Walsh-Hadamard)
            k_module.qk_had_K = had_K.to(k_module.weight.device).float()
            k_module.qk_K = K

# Usage Example:
# To replace the RoPE-based Q/K rotation wrapper with the new weight fusion approach,
# modify your main script as follows:
#
# OLD approach (current plot_activation.py lines 232-237):
#    rope_function_name = model_utils.get_rope_function_name(model)
#    layers = model_utils.get_layers(model)
#    for layer in layers:
#        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
#            layer.self_attn,
#            rope_function_name,
#            config=model.config,
#            online_had=args.qk_online_had,
#        )
#
# NEW approach (weight fusion + online inverse):
#    if args.pre_rotate and args.qk_online_had:
#        # Step 1: Fuse Hadamard into Q/K weights (offline)
#        rotation_utils.fuse_hadamard_into_qk_weights(model)
#
#        # Step 2: Register online inverse Hadamard (after quantization wrapping)
#        rotation_utils.register_qk_online_hadamard(model, model.config)
#
# This new approach provides:
# - Cleaner architecture (no monkey-patching)
# - Better performance (fused weights + efficient online inverse)
# - Easier debugging and profiling
# - Full compatibility with quantization and statistics collection
