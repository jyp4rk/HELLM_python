import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_utils import test_ppl,get_loaders
from utils.stat_utils import get_prefixed_tokens
from utils.train_utils import create_logger
from utils.sink_utils import run_complete_sink_analysis
from eval_utils.modeling_llama import LlamaForCausalLM


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
        type=str,
    )
    parser.add_argument("--output_dir", default="./log/softmax_sink", type=str)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--outlier_threshold", type=float, default=64.0)
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument(
        "--no_prefix", action="store_true", help="Run analysis without prefix tokens"
    )

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
    # logger.info(f"Loading model from {args.model_path}")
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.float16,
    #     device_map="auto"
    # )
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    # Get calibration data for finding prefix tokens
    logger.info("Loading calibration data...")
    calib_loader, _ = get_loaders(
        "wikitext2",
        tokenizer,
        train_size=args.calib_samples,
        val_size=0,
        seed=args.seed,
        seqlen=args.ppl_seqlen,
    )

    if args.no_prefix:
        # No prefix mode - analyze without any prefix tokens
        logger.info(
            "Running in NO PREFIX mode - analyzing sequences without prefix conditioning"
        )
        prefixed_tokens = []
        prefixed_key_values = None
        prefix_length = 0
    else:
        # Get prefix tokens using existing function
        logger.info("Finding prefix tokens...")
        from utils.prefix_cache import get_cached_prefixed_tokens

        prefixed_tokens = get_cached_prefixed_tokens(
            calib_loader,
            model,
            tokenizer,
            model_name="llama-2-7b",  # Add the required model_name parameter
            outlier_threshold=int(args.outlier_threshold),  # Ensure integer type
            activation_type="down_proj",  # Use the correct activation type
            cache_file=f"./cache/123.json",
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
        prefix_length = len(prefixed_tokens)

    # # Test perplexity with prefix KV cache
    # logger.info("Testing perplexity with prefix conditioning...")
    # datasets = ["wikitext2"]  # Only use wikitext2 to avoid C4 loading issues

    # # Test WITH prefix
    # logger.info("\n=== With prefix tokens ===")
    # prefix_ppl = test_ppl(args, model, tokenizer, prefixed_key_values, datasets)
    # for dataset in prefix_ppl:
    #     logger.info(f'{dataset} perplexity (with prefix): {prefix_ppl[dataset]:.2f}')

    # Run attention sink analysis``
    logger.info("\n=== Attention Sink Analysis ===")
    logger.info("Analyzing attention sink patterns in prefix vs sequence positions...")

    # Get test data for sink analysis
    test_loader, _ = get_loaders(
        "wikitext2",
        tokenizer,
        train_size=0,
        val_size=50,  # Use more samples for better analysis
        seed=args.seed,
        seqlen=args.ppl_seqlen,
    )

    # Get a sample batch for analysis
    test_batch = next(iter(test_loader))
    input_ids = test_batch[0][:1]  # Use first 1 sample for more robust analysis

    # Move input_ids to same device as model
    input_ids = input_ids.cuda()

    logger.info(f"Running sink analysis on sequence length: {input_ids.shape[1]}")
    logger.info(f"Prefix length: {prefix_length}")
    logger.info(f"Input device: {input_ids.device}")

    if args.no_prefix:
        logger.info(
            "Analyzing WITHOUT prefix conditioning - will track global maximum and minimum softmax inputs"
        )

    # Run complete sink analysis with prefixed KV cache
    _ = run_complete_sink_analysis(
        model=model,
        input_ids=input_ids,
        prefix_length=prefix_length,
        output_dir=Path(args.output_dir),
        past_key_values=prefixed_key_values,  # Pass the prefix KV cache (None if no prefix)
        use_pre_softmax=True,  # Use pre-softmax attention weights for better sink analysis
    )

    logger.info("Attention sink analysis completed!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
