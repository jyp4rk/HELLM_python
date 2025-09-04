CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
--model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
--model_name llama-2-7b \
--set_prefixed_tokens \
--save_dir ./figures/input_down_proj_prefix \
--plot_linear_input

# CUDA_VISIBLE_DEVICES=0,1 python plot_activation.py \
# --model_path /data/jypark/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
# --model_name llama-2-7b \
# --pre_rotate --down_online_had --qk_online_had \
# --set_prefixed_tokens \
# --save_dir ./figures/input_down_proj_rotate_prefix \
# --plot_linear_input
