#!/bin/bash

# Test perplexity with prefix tokens on Llama-2-7b
MODEL_PATH="/data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

echo "Testing Llama-2-7b perplexity with prefix tokens..."

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 4.0 \
    --calib_samples 1024

    # Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 5.0 \
    --calib_samples 1024

    # Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 6.0 \
    --calib_samples 1024

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 7.0 \
    --calib_samples 1024

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 8.0 \
    --calib_samples 1024

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 9.0 \
    --calib_samples 1024

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 10.0 \
    --calib_samples 1024

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 11.0 \
    --calib_samples 1024

# Run the evaluation
CUDA_VISIBLE_DEVICES=0,1 python eval_prefix.py \
    --model_path $MODEL_PATH \
    --output_dir ./log/prefix_ppl \
    --ppl_seqlen 2048 \
    --seed 2 \
    --outlier_threshold 12.0 \
    --calib_samples 1024
echo "Evaluation complete! Check ./log/prefix_ppl for results."

# OUTLIER_THRESHOLD=64

# # plot layer-wise magnitude of linear outputs with hadamard rotation and prefixed tokens
# CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
# --model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
# --model_name llama-2-7b \
# --pre_rotate \
# --outlier_threshold $OUTLIER_THRESHOLD \
# --set_prefixed_tokens \
# --save_dir ./figures/plot_linear_output_off_rotate_prefix_$OUTLIER_THRESHOLD \
# --plot_linear_output

# # plot the 3D images of linear input
# CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
# --model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
# --model_name llama-2-7b \
# --pre_rotate \
# --outlier_threshold $OUTLIER_THRESHOLD \
# --set_prefixed_tokens \
# --save_dir ./figures/plot_layer_input_3d_off_rotate_prefix_$OUTLIER_THRESHOLD \
# --plot_layer_input_3d
