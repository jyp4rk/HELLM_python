# PrefixQuant
Official PyTorch implement for [PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization](https://arxiv.org/abs/2410.05265).

## News
[2025/05] 🔥 **We explore the [Scaling Law for Quantization-Aware Training](https://export.arxiv.org/abs/2505.14302), which offers insights and instruction for LLMs QAT.**

[2025/1] Support the learnable activation cliping for dynamic quantization.

[2024/10] We release PrefixQuant, the first work to let static activation quantization outperforms dynamic ones in LLM. We only open the fake quantization code now, and the inference kernels will be released later.

## 📋 Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training & Quantization](#training--quantization)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
- [Testing](#testing)
- [Migration Guide](#migration-guide)
- [Citation](#citation)

## 📁 Project Structure

The codebase has been reorganized for better maintainability:

```
HELLM_python/
├── src/                      # Core source code
│   ├── models/              # Model implementations
│   │   ├── llama.py                    # Base LLaMA model
│   │   ├── llama_ckks.py               # CKKS noise injection variant
│   │   ├── llama_quant.py              # Quantized variant
│   │   └── components/                 # Reusable components
│   │       └── noisy_linear.py         # Noisy linear layers
│   ├── quantization/        # Quantization module
│   │   ├── core/                       # Core quantization logic
│   │   │   ├── quantizer.py           # Quantizer implementations
│   │   │   ├── block_ap.py            # Block-wise quantization
│   │   │   └── fixed_point.py         # Fixed-point quantization
│   │   ├── linear/                     # Linear layer quantization
│   │   │   ├── int_linear_fake.py     # Fake quantization
│   │   │   └── int_linear_real.py     # Real quantization
│   │   ├── layers/                     # Quantized layers
│   │   │   ├── quant_norm.py          # Quantized normalization
│   │   │   └── noisy_swish.py         # Noisy activation
│   │   ├── losses/                     # Loss functions
│   │   └── triton_utils/               # Triton kernels
│   ├── training/            # Training utilities
│   │   ├── fsdp_trainer.py             # FSDP trainer
│   │   ├── optimizer.py                # Custom optimizers
│   │   └── quant_linear.py             # Training quantized layers
│   └── utils/               # Utility functions
│       ├── data_utils.py               # Data loading/processing
│       ├── model_utils.py              # Model utilities
│       ├── rotation_utils.py           # Hadamard rotation
│       ├── hadamard_utils.py           # Hadamard transforms
│       ├── quant_utils.py              # Quantization helpers
│       └── ckks_utils.py               # CKKS noise injection
├── scripts/                 # Entry point scripts
│   └── train.py                        # Main training script
├── benchmarks/              # Evaluation scripts
│   ├── eval.py                         # Main evaluation
│   ├── eval_perplexity.py              # Perplexity evaluation
│   └── eval_utils/                     # Evaluation utilities
├── tests/                   # Test suite
│   ├── test_q_k_up_gate.py
│   ├── test_ckks_noise_injection.py
│   └── test_softmax_configurations.py
├── experiments/             # Research experiments
│   ├── optimize_rotation.py            # Rotation optimization
│   ├── plot_activation.py              # Activation visualization
│   └── stat_activation.py              # Activation statistics
└── examples/                # Example scripts
```


## Installation

### 1. Create Environment
```bash
# Create conda environment
conda create -n prefixquant python==3.9

# Activate environment
conda activate prefixquant

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
# Check Python version
python --version  # Should be 3.9.x

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# List installed packages
pip list | grep -E "torch|transformers"
```

## 🚀 Quick Start

### Basic Quantization (No Fine-tuning)
```bash
# Quantize Llama-3-8B to W4A4KV4 (4-bit weights, activations, and KV cache)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-w4a4kv4 \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --v_bits 4 \
    --k_bits 4 \
    --kv_group_size 128 \
    --kv_mode static \
    --mse_init \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --save_quant_dir ./pre_quantized_models/llama-3-8b-w4a4kv4
```

### Quick Evaluation
```bash
# Evaluate a quantized model
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval.py \
    --quant_model ./pre_quantized_models/llama-3-8b-w4a4kv4 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## 📚 Training & Quantization

### 1. W4A4KV4 Quantization (4-bit everything)
```bash
# Basic quantization
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-w4a4kv4 \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --v_bits 4 \
    --k_bits 4 \
    --kv_group_size 128 \
    --kv_mode static \
    --mse_init \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --save_quant_dir ./pre_quantized_models/llama-3-8b-w4a4kv4
```

### 2. W4A4KV4 with Fine-tuning (Recommended)
```bash
# Add fine-tuning for better accuracy
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-w4a4kv4-ft \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --v_bits 4 \
    --k_bits 4 \
    --kv_group_size 128 \
    --kv_mode static \
    --mse_init \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --epochs 20 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --save_quant_dir ./pre_quantized_models/llama-3-8b-w4a4kv4-ft
```

### 3. W4A8KV4 Quantization (Higher accuracy)
```bash
# 4-bit weights, 8-bit activations, 4-bit KV cache
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-w4a8kv4 \
    --wbits 4 \
    --input_bits 8 \
    --input_mode static \
    --v_bits 4 \
    --k_bits 4 \
    --kv_group_size 128 \
    --kv_mode static \
    --mse_init \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --epochs 10 \
    --eval_ppl \
    --save_quant_dir ./pre_quantized_models/llama-3-8b-w4a8kv4
```

### 4. Dynamic Quantization
```bash
# Use dynamic quantization with activation clipping
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-dynamic \
    --wbits 4 \
    --input_bits 4 \
    --input_mode dynamic \
    --activation_clipping \
    --kv_mode dynamic \
    --eval_ppl \
    --save_quant_dir ./pre_quantized_models/llama-3-8b-dynamic
```

### 5. Large Model Quantization (Llama-3-70B)
```bash
# Quantize 70B model with adjusted learning rates
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
    --model_path /path/to/llama-3-70B \
    --model_name Llama-3-70b \
    --output_dir ./log/llama-3-70b-w4a4kv4 \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --v_bits 4 \
    --k_bits 4 \
    --quant_lr 2e-5 \
    --weight_lr 2e-6 \
    --epochs 20 \
    --eval_ppl \
    --save_quant_dir ./pre_quantized_models/llama-3-70b-w4a4kv4
```

### 6. Llama-2-70B (with stability fix)
```bash
# Llama-2-70B requires skip_mse loss for stability
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
    --model_path /path/to/llama-2-70B \
    --model_name Llama-2-70b \
    --output_dir ./log/llama-2-70b-w4a4kv4 \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --loss_type skip_mse \
    --epochs 20 \
    --eval_ppl \
    --save_quant_dir ./pre_quantized_models/llama-2-70b-w4a4kv4
```

## 🔍 Evaluation

### 1. Perplexity Evaluation
```bash
# Evaluate perplexity on WikiText2 and C4
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval_perplexity.py \
    --quant_model ./pre_quantized_models/llama-3-8b-w4a4kv4 \
    --eval_ppl
```

### 2. Downstream Task Evaluation
```bash
# Evaluate on multiple downstream tasks
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval.py \
    --quant_model ./pre_quantized_models/llama-3-8b-w4a4kv4 \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

### 3. CKKS Noise Injection Evaluation
```bash
# Evaluate with CKKS noise injection (for secure inference)
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval_perplexity_ckks.py \
    --quant_model ./pre_quantized_models/llama-3-8b-w4a4kv4 \
    --eval_ppl \
    --noise_scale 0.01
```

### 4. Softmax Sink Analysis
```bash
# Analyze softmax sink tokens
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval_softmax_sink.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --analyze_sinks
```

## 📊 Visualization

### 1. Plot Linear Input Activations
```bash
# Visualize token-wise maximum values for linear inputs
CUDA_VISIBLE_DEVICES=0 python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_linear_input \
    --output_dir ./plots/linear_input
```

### 2. Plot with Hadamard Rotation
```bash
# Plot activations with Hadamard rotation applied
CUDA_VISIBLE_DEVICES=0 python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_linear_input \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --output_dir ./plots/with_rotation
```

### 3. Plot Different Activation Types
```bash
# Plot linear outputs (Q/K/V projections)
python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_linear_output

# Plot outlier token positions
python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_outlier_token_position

# Plot outlier token content
python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_outlier_token

# Plot layer-wise outlier token numbers
python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_layer_wise_outlier_token_number

# Plot 3D layer inputs
python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_layer_input_3d

# Plot 3D block outputs
python experiments/plot_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --plot_block_output_3d
```

### 4. Generate Activation Statistics
```bash
# Collect activation statistics across layers
CUDA_VISIBLE_DEVICES=0 python experiments/stat_activation.py \
    --model_path /path/to/llama-2-7b \
    --model_name llama-2-7b \
    --output_dir ./stats/activations
```

### 5. Optimize Rotation Matrices
```bash
# Optimize Hadamard rotation matrices
CUDA_VISIBLE_DEVICES=0 python experiments/optimize_rotation.py \
    --model_path /path/to/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/optimize_rotation \
    --epochs 10
```

## 🧪 Advanced Usage

### Custom Model Configuration
```bash
# Fine-grained control over quantization
python scripts/train.py \
    --model_path /path/to/model \
    --model_name custom-model \
    --output_dir ./log/custom \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --input_group_size 128 \
    --v_bits 4 \
    --k_bits 4 \
    --kv_group_size 128 \
    --kv_mode static \
    --mse_init \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --quant_lr 1e-4 \
    --weight_lr 1e-5 \
    --epochs 20 \
    --batch_size 1 \
    --max_memory 40 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy \
    --save_quant_dir ./models/custom
```

### Multi-GPU Training
```bash
# Use multiple GPUs with FSDP
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
    --model_path /path/to/large-model \
    --model_name large-model \
    --output_dir ./log/multi_gpu \
    --wbits 4 \
    --input_bits 4 \
    --epochs 20 \
    --max_memory 40 \
    --save_quant_dir ./models/multi_gpu
```

### Export Quantized Model
```bash
# After quantization, the model is automatically saved to save_quant_dir
# You can load it later with:
python benchmarks/eval.py \
    --quant_model ./pre_quantized_models/your-model \
    --eval_ppl
```

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_q_k_up_gate.py -v

# Run CKKS noise injection test
python -m pytest tests/test_ckks_noise_injection.py -v

# Run softmax configuration test
python -m pytest tests/test_softmax_configurations.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📖 Migration Guide

**⚠️ IMPORTANT:** The codebase structure has been reorganized. If you have existing scripts or notebooks:

### Old Import Paths → New Import Paths

```python
# OLD (deprecated)
from utils.data_utils import get_loaders
from quantize.block_ap import block_ap
from train_utils.optimizer import custom_optimizer
from eval_utils.rotation_utils import rotate_model

# NEW (current)
from src.utils.data_utils import get_loaders
from src.quantization.core.block_ap import block_ap
from src.training.optimizer import custom_optimizer
from benchmarks.eval_utils.rotation_utils import rotate_model
```

### Old Script Paths → New Script Paths

```bash
# OLD
python main.py ...
python eval.py ...
python plot_activation.py ...

# NEW
python scripts/train.py ...
python benchmarks/eval.py ...
python experiments/plot_activation.py ...
```

### Complete Import Mapping

| Old Module | New Module |
|-----------|-----------|
| `quantize.*` | `src.quantization.*` |
| `quantize.quantizer` | `src.quantization.core.quantizer` |
| `quantize.block_ap` | `src.quantization.core.block_ap` |
| `quantize.int_linear_fake` | `src.quantization.linear.int_linear_fake` |
| `quantize.int_linear_real` | `src.quantization.linear.int_linear_real` |
| `quantize.recon_loss` | `src.quantization.losses.recon_loss` |
| `utils.*` | `src.utils.*` |
| `train_utils.*` | `src.training.*` |
| `train_utils.noisy_linear` | `src.models.components.noisy_linear` |
| `eval_utils.*` | `benchmarks.eval_utils.*` |

More examples can be found in `./examples/`.


## Citation
If you use our PrefixQuant approach in your research, please cite our paper:
```
@article{prefixquant,
  title={PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization},
  author={Chen, Mengzhao and  Liu, Yi and Wang, Jiahao and Bin, Yi and Shao, Wenqi and Luo, Ping},
  journal={arXiv preprint arXiv:2410.05265},
  year={2024}
}
```
