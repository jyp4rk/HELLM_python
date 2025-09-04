#!/bin/bash
OUTLIER_OBJECT=all
OUTLIER_THRESHOLD=8

# plot layer-wise magnitude of linear outputs
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
--model_name llama-2-7b \
--set_prefixed_tokens \
--outlier_object $OUTLIER_OBJECT \
--outlier_threshold $OUTLIER_THRESHOLD \
--save_dir ./figures/output_prefix_${OUTLIER_THRESHOLD}_${OUTLIER_OBJECT} \
--plot_linear_output

# # plot the 3D images of linear input
# CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
# --model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
# --model_name llama-2-7b \
# --pre_rotate --down_online_had --qk_online_had \
# --save_dir ./figures/plot_layer_input_3d_prefix \
# --plot_layer_input_3d
