#!/bin/bash
OUTLIER_THRESHOLD=64
OUTLIER_OBJECT=hidden_state

# plot layer-wise magnitude of linear outputs with hadamard rotation and prefixed tokens
CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
--model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
--model_name llama-2-7b \
--pre_rotate \
--outlier_threshold $OUTLIER_THRESHOLD \
--outlier_object $OUTLIER_OBJECT \
--set_prefixed_tokens \
--save_dir ./figures/plot_linear_output_off_rotate_prefix_${OUTLIER_THRESHOLD}_${OUTLIER_OBJECT} \
--plot_linear_output

# plot the 3D images of linear input
CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
--model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
--model_name llama-2-7b \
--pre_rotate \
--outlier_threshold $OUTLIER_THRESHOLD \
--outlier_object $OUTLIER_OBJECT \
--set_prefixed_tokens \
--save_dir ./figures/input_3d_off_rotate_prefix_${OUTLIER_THRESHOLD}_${OUTLIER_OBJECT} \
--plot_layer_input_3d
