#!/bin/bash
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Enhanced SpinQuant rotation optimization using Hinged-Max outlier loss
# Hinged-Max provides direct threshold control with zero penalty inside [-T, T]
# Supports both hard hinge and soft L∞ (log-sum-exp) variants

# Usage: ./12_optimize_rotation_hinged.sh <model> <w_bits> <a_bits> <k_bits> [v_bits] [preset] [threshold]
# Example: ./12_optimize_rotation_hinged.sh meta-llama/Llama-2-7b-hf 16 16 16 16 default 5.0
# Presets: default, aggressive, conservative, soft_lse

MODEL=${1:-"meta-llama/Llama-2-7b-hf"}
W_BITS=${2:-16}
A_BITS=${3:-4}
K_BITS=${4:-4}
V_BITS=${5:-$4}
PRESET=${6:-"default"}
THRESHOLD=${7:-"5.0"}

echo "======================================================"
echo "SpinQuant Hinged-Max Outlier Loss Rotation Optimization"
echo "======================================================"
echo "Model: $MODEL"
echo "Bits: W=$W_BITS, A=$A_BITS, K=$K_BITS, V=$V_BITS"
echo "Hinged Config: preset=$PRESET, target_threshold=$THRESHOLD"
echo "======================================================"

# Set Hinged-Max loss configuration via environment variables
export OUTLIER_LOSS_TYPE="hinged"
export OUTLIER_CONFIG_PRESET="$PRESET"
export OUTLIER_THRESHOLD="$THRESHOLD"

# Loss scaling configuration (make Hinged loss more significant)
export OUTLIER_LOSS_SCALE="0.1"      # 10x larger than previous 0.01 scaling
export HINGED_LOSS_SCALE="0.1"       # Hinged-specific scaling

# Adaptive threshold based on quantization precision
# For 16-bit quantization, activations are typically smaller, so reduce threshold
if [ "$W_BITS" = "16" ] && [ "$A_BITS" = "16" ]; then
    THRESHOLD="2.0"  # Lower threshold for 16-bit
    echo "Using adapted threshold=2.0 for 16-bit quantization"
else
    THRESHOLD="${THRESHOLD:-5.0}"  # Use provided or default threshold for lower precision
    echo "Using threshold=$THRESHOLD for mixed precision"
fi
export OUTLIER_THRESHOLD="$THRESHOLD"

# Additional Hinged-Max parameters
if [ "$PRESET" = "soft_lse" ]; then
    echo "Using soft L∞ (log-sum-exp) variant"
    export OUTLIER_LSE_TAU_INITIAL="0.1"
    export OUTLIER_LSE_TAU_FINAL="10.0"
else
    echo "Using hard hinge variant"
fi

# Set GPU configuration for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1  # Disable P2P for stability

# Run optimization with torchrun for distributed training
torchrun --nnodes=1 --nproc_per_node=2 --master_port=29502 optimize_rotation.py \
--input_model "$MODEL" \
--output_rotation_path "./outputs/hinged_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}" \
--output_dir "./outputs/hinged_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/" \
--logging_dir "./logs/hinged_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 0.01 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_safetensors False \
--max_steps 500 \
--max_grad_norm 1.0 \
--gradient_accumulation_steps 1 \
--warmup_ratio 0.05 \
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
echo "Hinged-Max rotation optimization completed!"
echo "Results saved to: ./outputs/hinged_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/"
echo "Logs saved to: ./logs/hinged_${PRESET}_${W_BITS}_${A_BITS}_${K_BITS}/"
echo "======================================================"