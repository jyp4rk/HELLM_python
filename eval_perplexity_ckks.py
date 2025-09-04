import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, LlamaConfig
from train_utils.modeling_llama_CKKS import LlamaForCausalLM
from utils.data_utils import test_ppl, get_loaders
from utils.stat_utils import get_prefixed_tokens
from utils.train_utils import create_logger
from utils.attribute_noise_injector import NoiseConfig, NoiseType

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", default="./log/prefix_ppl", type=str)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--outlier_threshold", type=float, default=5.0)
    parser.add_argument("--activation_type", type=str, default="hidden_state")
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--rotation_mode", type=str, default="identity", choices=["identity", "hadamard"],
                       help="Rotation matrix type for R1/R2 (identity or hadamard)")
    parser.add_argument("--rmsnorm_noise_std", type=float, default=None,
                       help="Noise std for RMSNorm layers")
    parser.add_argument("--softmax_noise_std", type=float, default=None,
                       help="Noise std for attention softmax")
    parser.add_argument("--activation_noise_std", type=float, default=None,
                       help="Noise std for SiLU/Swish activations")
    # parser.add_argument("--linear_noise_enabled", action="store_true",
    #                    help="Enable noise injection for NoisyLinear layers")
    parser.add_argument("--N_bitwidth", type=float, default=16,
                       help="Polynomial degree for NoisyLinear noise")
    parser.add_argument("--hamming_weight", type=float, default=192,
                       help="Hamming_weight for NoisyLinear noise")
    parser.add_argument("--delta_bitwidth", type=float, default=42,
                       help="Delta bitwidth for NoisyLinear noise")


    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setup logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger(Path(args.output_dir))

    # Load model and tokenizer
    logger.info(f"Loading CKKS model from {args.model_path}")
    config = LlamaConfig.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Initialize rotation matrices
    logger.info(f"Initializing rotation matrices with mode: {args.rotation_mode}")
    if args.rotation_mode == "hadamard":
        import utils.rotation_utils as rotation_utils
        Q = rotation_utils.get_orthogonal_matrix(config.hidden_size, "hadamard", torch.device("cuda"))
        model.R1 = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        model.R1.weight.data = Q.clone()
        model.R1.requires_grad_(False)
        for layer in model.model.layers:
            layer.self_attn.R2 = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            layer.self_attn.R2.weight.data = Q.clone()
            layer.self_attn.R2.requires_grad_(False)
    else:  # identity
        model.R1 = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        model.R1.weight.data = torch.eye(config.hidden_size)
        model.R1.requires_grad_(False)
        for layer in model.model.layers:
            layer.self_attn.R2 = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            layer.self_attn.R2.weight.data = torch.eye(config.hidden_size)
            layer.self_attn.R2.requires_grad_(False)

    # Setup noise injection if requested
    noise_configs = {}

    # RMSNorm noise
    if args.rmsnorm_noise_std is None:
        rmsnorm_std = 0.0  # Default to no noise if not specified
    else:
        rmsnorm_std = args.rmsnorm_noise_std

    if rmsnorm_std > 0:
        noise_configs['rmsnorm'] = NoiseConfig(
            noise_type=NoiseType.GAUSSIAN,
            std=rmsnorm_std,
            enabled=True
        )
        logger.info(f"RMSNorm noise injection enabled with std: {rmsnorm_std}")

    # Softmax noise
    if args.softmax_noise_std is None:
        softmax_std = 0.0  # Default to no noise if not specified
    else:
        softmax_std = args.softmax_noise_std

    if softmax_std > 0:
        noise_configs['softmax'] = NoiseConfig(
            noise_type=NoiseType.GAUSSIAN,
            std=softmax_std,
            enabled=True
        )
        logger.info(f"Softmax noise injection enabled with std: {softmax_std}")

    # Activation (SiLU/Swish) noise
    if args.activation_noise_std is None:
        activation_std = 0.0  # Default to no noise if not specified
    else:
        activation_std = args.activation_noise_std

    if activation_std > 0:
        noise_configs['activation'] = NoiseConfig(
            noise_type=NoiseType.GAUSSIAN,
            std=activation_std,
            enabled=True
        )
        logger.info(f"Activation noise injection enabled with std: {activation_std}")

    # Configure NoisyLinear noise if requested
    sqrt_Nh = np.sqrt(args.hamming_weight * args.N_bitwidth)
    linear_noise_config = {
        "sqrt_Nh": sqrt_Nh,
        "delta_bitwidth": args.delta_bitwidth,
        "fractional_bitwidth": args.delta_bitwidth-15
    }
    logger.info(f"NoisyLinear noise will be enabled with sqrt_Nh={sqrt_Nh:.2f}, delta_bitwidth={args.delta_bitwidth}, fractional_bitwidth={args.delta_bitwidth-15}")

    # Apply all noise configurations to model
    if noise_configs or linear_noise_config:
        model.setup_noise_injection(
            rmsnorm_config=noise_configs.get('rmsnorm') if noise_configs else None,
            softmax_config=noise_configs.get('softmax') if noise_configs else None,
            activation_config=noise_configs.get('activation') if noise_configs else None,
            linear_noise_config=linear_noise_config
        )

        enabled_noise_types = []
        if noise_configs:
            enabled_noise_types.extend(list(noise_configs.keys()))
        if linear_noise_config:
            enabled_noise_types.append('linear')

        logger.info(f"Noise injection configured for: {enabled_noise_types}")
    else:
        logger.info("No noise injection enabled")

    model.eval()

    # Get calibration data for finding prefix tokens
    logger.info("Loading calibration data...")
    calib_loader, _ = get_loaders(
        "wikitext2", tokenizer,
        train_size=args.calib_samples,
        val_size=0,
        seed=args.seed,
        seqlen=args.ppl_seqlen
    )

    # Get prefix tokens using existing function
    logger.info("Finding prefix tokens...")
    from utils.prefix_cache import get_cached_prefixed_tokens
    prefixed_tokens = get_cached_prefixed_tokens(
        calib_loader, model, tokenizer,
        model_name="llama-2-7b",  # Add the required model_name parameter
        outlier_threshold=int(args.outlier_threshold),  # Ensure integer type
        activation_type=args.activation_type,  # Use the correct activation type
        cache_file=f'./cache/123.json'
    )
    logger.info(f"outlier_threshold: {args.outlier_threshold}")
    logger.info(f"Found {len(prefixed_tokens)} prefix tokens: {prefixed_tokens}")
    logger.info(f"Decoded: {tokenizer.decode(prefixed_tokens)}")

    # Generate KV cache from prefix tokens
    logger.info("Generating KV cache from prefix tokens...")
    prefix_input = torch.tensor([prefixed_tokens]).cuda()
    with torch.no_grad():
        outputs = model(prefix_input, use_cache=True)
        prefixed_key_values = outputs.past_key_values

    # Test perplexity with prefix KV cache
    logger.info("Testing perplexity with prefix conditioning...")
    datasets = ["wikitext2"]  # Only use wikitext2 to avoid C4 loading issues

    # # Test WITHOUT prefix (baseline)
    # logger.info("\n=== Baseline (no prefix) ===")
    # baseline_ppl = test_ppl(args, model, tokenizer, None, datasets)
    # for dataset in baseline_ppl:
    #     logger.info(f'{dataset} perplexity (baseline): {baseline_ppl[dataset]:.2f}')

    # Test WITH prefix
    logger.info("\n=== With prefix tokens ===")
    prefix_ppl = test_ppl(args, model, tokenizer, prefixed_key_values, datasets)
    for dataset in prefix_ppl:
        logger.info(f'{dataset} perplexity (with prefix): {prefix_ppl[dataset]:.2f}')

    # # Show difference
    # logger.info("\n=== Comparison ===")
    # for dataset in datasets:
    #     diff = prefix_ppl[dataset] - baseline_ppl[dataset]
    #     pct_change = (diff / baseline_ppl[dataset]) * 100
    #     logger.info(f'{dataset}: {baseline_ppl[dataset]:.2f} → {prefix_ppl[dataset]:.2f} (Δ={diff:.2f}, {pct_change:+.1f}%)')

if __name__ == "__main__":
    main()
