#!/bin/bash

# Test script for eval_perplexity_ckks.py with various configurations
# Tests different combinations of rotation modes and noise levels

MODEL_PATH="/data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"  # Update this path to your model
OUTPUT_DIR="./log/ckks_perplexity_tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "CKKS Perplexity Evaluation Test Suite"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Test 1: (Baseline) Hadamard rotation, no noise
python eval_perplexity_ckks.py \
    --model_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR}/test1_hadamard_no_noise_${TIMESTAMP} \
    --ppl_seqlen 2048 \
    --seed 42 \
    --outlier_threshold 9.0 \
    --activation_type hidden_state \
    --calib_samples 16 \
    --rotation_mode identity \
    --rmsnorm_noise_std 0.0 \
    --softmax_noise_std 0.0 \
    --activation_noise_std 0.0 \
    --N_bitwidth 16 \
    --hamming_weight 192 \
    --delta_bitwidth 36 \

# # Test 5: Hadamard rotation with small noise
# echo ""
# echo "Test 5: Hadamard rotation with small noise (std=0.01)"
# echo "------------------------------------------"
# python eval_perplexity_ckks.py \
#     --model_path ${MODEL_PATH} \
#     --output_dir ${OUTPUT_DIR}/test5_hadamard_noise_0.01_${TIMESTAMP} \
#     --rotation_mode hadamard \
#     --rmsnorm_noise_std 0.01 \
#     --softmax_noise_std 0.01 \
#     --activation_noise_std 0.01 \
#     --ppl_seqlen 2048 \
#     --seed 42 \
#     --outlier_threshold 9.0 \
#     --activation_type hidden_state \
#     --calib_samples 16 \
#     --N_bitwidth 16 \
#     --hamming_weight 192 \
#     --delta_bitwidth 42

# # Test 6: Hadamard rotation with medium noise
# echo ""
# echo "Test 6: Hadamard rotation with medium noise (std=0.05)"
# echo "------------------------------------------"
# python eval_perplexity_ckks.py \
#     --model_path ${MODEL_PATH} \
#     --output_dir ${OUTPUT_DIR}/test6_hadamard_noise_0.05_${TIMESTAMP} \
#     --rotation_mode hadamard \
#     --rmsnorm_noise_std 0.05 \
#     --softmax_noise_std 0.05 \
#     --activation_noise_std 0.05 \
#     --ppl_seqlen 2048 \
#     --seed 42 \
#     --outlier_threshold 9.0 \
#     --activation_type hidden_state \
#     --calib_samples 16 \
#     --N_bitwidth 16 \
#     --hamming_weight 192 \
#     --delta_bitwidth 42

# echo ""
# echo "=========================================="
# echo "All tests completed!"
# echo "Results saved to: ${OUTPUT_DIR}"
# echo "=========================================="

# # Optional: Generate summary report
# echo ""
# echo "Generating summary report..."
# echo ""
# echo "Test Summary (${TIMESTAMP})" > ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt
# echo "================================" >> ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt

# for test_dir in ${OUTPUT_DIR}/test*_${TIMESTAMP}; do
#     if [ -d "$test_dir" ]; then
#         test_name=$(basename $test_dir)
#         echo "" >> ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt
#         echo "Test: $test_name" >> ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt
#         # Extract perplexity results from log files if they exist
#         if [ -f "$test_dir/main.log" ]; then
#             grep -E "perplexity|ppl" "$test_dir/main.log" | tail -n 5 >> ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt
#         fi
#     fi
# done

echo "Summary report saved to: ${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
