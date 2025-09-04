#!/bin/bash
OUTLIER_OBJECT=hidden_state
OUTLIER_THRESHOLD=64

# plot layer-wise magnitude of linear outputs with hadamard rotation and prefixed tokens
CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
--model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
--model_name llama-2-7b \
--pre_rotate \
--outlier_object $OUTLIER_OBJECT \
--outlier_threshold $OUTLIER_THRESHOLD \
--set_prefixed_tokens \
--save_dir ./figures/output_onoff_rotate_prefix_notestqkdirecthad_${OUTLIER_THRESHOLD}_${OUTLIER_OBJECT} \
--plot_linear_output

# --pre_rotate --qk_direct_had \
# # plot the 3D images of linear input
# CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
# --model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
# --model_name llama-2-7b \
# --pre_rotate --down_online_had --qk_online_had \
# --outlier_threshold $OUTLIER_THRESHOLD \
# --set_prefixed_tokens \
# --save_dir ./figures/plot_layer_input_3d_rotate_prefix_$OUTLIER_THRESHOLD \
# --plot_layer_input_3d
