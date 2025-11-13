"""
RMSNorm Input Range Analysis

This script analyzes the input range of RMSNorm layers across all positions in the model.
Similar to eval_softmax_sink.py but for RMSNorm inputs.

RMSNorm appears in two positions per decoder layer:
1. input_layernorm: Before attention (pre-attention)
2. post_attention_layernorm: After attention, before MLP (post-attention)
"""

import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer
from utils.data_utils import get_loaders
from utils.train_utils import create_logger
from eval_utils.modeling_llama import LlamaForCausalLM


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
        type=str,
    )
    parser.add_argument("--output_dir", default="./log/rmsnorm_range", type=str)
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3)
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
    logger.info(f"Loading model from {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    # Get calibration data
    logger.info("Loading calibration data...")
    test_loader, _ = get_loaders(
        "wikitext2",
        tokenizer,
        train_size=0,
        val_size=50,
        seed=args.seed,
        seqlen=args.ppl_seqlen,
    )

    # Get a sample batch for analysis
    test_batch = next(iter(test_loader))
    input_ids = test_batch[0][:1]  # Use first sample
    input_ids = input_ids.cuda()

    logger.info(f"Running RMSNorm range analysis on sequence length: {input_ids.shape[1]}")
    logger.info(f"Input device: {input_ids.device}")

    # Run RMSNorm range analysis
    from utils.rmsnorm_utils import run_complete_rmsnorm_analysis

    _ = run_complete_rmsnorm_analysis(
        model=model,
        input_ids=input_ids,
        output_dir=Path(args.output_dir),
    )

    logger.info("RMSNorm range analysis completed!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
