#!/bin/bash
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Enhanced SpinQuant rotation optimization using CVaR outlier loss
# CVaR (Conditional Value-at-Risk) focuses on the worst α% of activations
# providing more efficient gradient allocation compared to zone-based penalties

# Usage: ./11_optimize_rotation_cvar.sh <model> <w_bits> <a_bits> <k_bits> [v_bits] [preset] [alpha]
# Example: ./11_optimize_rotation_cvar.sh meta-llama/Llama-2-7b-hf 16 16 16 16 default 0.99
# Presets: default, aggressive, conservative, fast
# - default: LR=0.01, balanced convergence
# - aggressive: enhanced CVaR parameters for faster threshold learning
# - fast: LR=0.02, fastest convergence (use with caution)

MODEL=${1:-"meta-llama/Llama-2-7b-hf"}
W_BITS=${2:-16}
A_BITS=${3:-4}
K_BITS=${4:-4}
V_BITS=${5:-$4}
PRESET=${6:-"default"}
ALPHA=${7:-"0.99"}

# Support for fast learning rate preset
if [ "$PRESET" = "fast" ]; then
    LEARNING_RATE="0.02"   # 2x faster base learning rate
    WARMUP_RATIO="0.02"    # Very fast warmup 
    echo "Using FAST preset: LR=0.02, warmup=0.02"
else
    LEARNING_RATE="0.01"   # Standard improved learning rate
    WARMUP_RATIO="0.05"    # Standard improved warmup
fi

echo "======================================================"
echo "SpinQuant CVaR Outlier Loss Rotation Optimization"
echo "======================================================"
echo "Model: $MODEL"
echo "Bits: W=$W_BITS, A=$A_BITS, K=$K_BITS, V=$V_BITS"
echo "CVaR Config: preset=$PRESET, alpha=$ALPHA"
echo "Learning rate: 0.01 (10x increase), warmup_ratio: 0.05 (faster warmup)"
echo "======================================================"

# Set CVaR loss configuration via environment variables
export OUTLIER_LOSS_TYPE="cvar"
export OUTLIER_CONFIG_PRESET="$PRESET"
export OUTLIER_ALPHA="$ALPHA"

# Loss scaling configuration (make CVaR loss more significant)
export OUTLIER_LOSS_SCALE="0.1"  # 10x larger than previous 0.01 scaling
export CVAR_LOSS_SCALE="0.1"     # CVaR-specific scaling

# Adaptive threshold based on quantization precision
# For 16-bit quantization, activations are typically smaller, so reduce threshold
if [ "$W_BITS" = "16" ] && [ "$A_BITS" = "16" ]; then
    export OUTLIER_THRESHOLD="2.0"  # Lower threshold for 16-bit
    echo "Using adapted threshold=2.0 for 16-bit quantization"
else
    export OUTLIER_THRESHOLD="5.0"  # Standard threshold for lower precision
    echo "Using standard threshold=5.0 for mixed precision"
fi

# Enhanced CVaR configuration for faster convergence
export CVAR_THRESHOLD_LR_SCALE="20.0"  # 2x faster threshold learning
export CVAR_EMA_DECAY="0.9"            # Faster statistics adaptation
if [ "$PRESET" = "aggressive" ]; then
    export CVAR_THRESHOLD_LR_SCALE="50.0"  # Very fast threshold learning
    export CVAR_EMA_DECAY="0.8"            # Very fast statistics adaptation
    echo "Using aggressive CVaR parameters for faster convergence"
fi

# Set GPU configuration for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1  # Disable P2P for stability

# Run optimization with torchrun for distributed training
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29501 optimize_rotation.py \
--input_model "$MODEL" \
--output_rotation_path "./outputs/cvar_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}" \
--output_dir "./outputs/cvar_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/" \
--logging_dir "./logs/cvar_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate $LEARNING_RATE \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_safetensors False \
--max_steps 500 \
--max_grad_norm 1.0 \
--gradient_accumulation_steps 1 \
--warmup_ratio $WARMUP_RATIO \
--dataloader_num_workers 0 \
--w_bits $W_BITS \
--a_bits $A_BITS \
--k_bits $K_BITS \
--v_bits $V_BITS \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128

echo "======================================================"
echo "CVaR rotation optimization completed!"
echo "Results saved to: ./outputs/cvar_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/"
echo "Logs saved to: ./logs/cvar_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/"
echo "======================================================"