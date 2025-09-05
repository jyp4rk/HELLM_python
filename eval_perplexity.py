import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_utils import test_ppl, get_loaders
from utils.stat_utils import get_prefixed_tokens
from utils.train_utils import create_logger

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", default="./log/prefix_ppl", type=str)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--outlier_threshold", type=float, default=5.0)
    parser.add_argument("--calib_samples", type=int, default=64)

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
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
        activation_type="hidden_state",  # Use the correct activation type
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
