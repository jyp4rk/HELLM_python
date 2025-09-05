#!/bin/bash

# Quick test script for eval_perplexity_ckks.py
# Runs minimal configurations for rapid validation

MODEL_PATH="/data/models/llama-2-7b-hf"  # Update this path to your model
OUTPUT_DIR="./log/ckks_quick_test"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "CKKS Perplexity Quick Test"
echo "=========================================="

# Quick test with reduced samples for faster execution
echo ""
echo "Running quick test (reduced samples)..."
echo "------------------------------------------"

# Test with identity rotation and no noise (baseline)
echo "Configuration: Identity rotation, no noise"
python eval_perplexity_ckks.py \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/identity_baseline \
    --rotation_mode identity \
    --noise_std 0.0 \
    --ppl_seqlen 512 \
    --seed 42 \
    --outlier_threshold 5.0 \
    --calib_samples 8

# Test with hadamard rotation and small noise  
echo ""
echo "Configuration: Hadamard rotation, small noise (0.01)"
python eval_perplexity_ckks.py \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/hadamard_noise \
    --rotation_mode hadamard \
    --noise_std 0.01 \
    --ppl_seqlen 512 \
    --seed 42 \
    --outlier_threshold 5.0 \
    --calib_samples 8

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="