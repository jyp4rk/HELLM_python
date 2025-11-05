# PrefixQuant Usage Guide

Complete guide with practical examples for using PrefixQuant.

## Table of Contents
- [Getting Started](#getting-started)
- [Common Use Cases](#common-use-cases)
- [Parameter Reference](#parameter-reference)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/HELLM_python.git
cd HELLM_python

# Create conda environment
conda create -n prefixquant python=3.9
conda activate prefixquant

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### 2. Download Model

```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli login  # Enter your HF token

# Download Llama-3-8B
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./models/llama-3-8B

# Or use git
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B ./models/llama-3-8B
```

### 3. Quick Test

```bash
# Test basic quantization (no fine-tuning, ~30 min on A100)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path ./models/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/test \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --eval_ppl \
    --save_quant_dir ./models/test-quantized
```

## Common Use Cases

### Use Case 1: Standard Quantization (Recommended)

**Goal:** Quantize Llama-3-8B with best accuracy/efficiency trade-off

```bash
# W4A4KV4 with fine-tuning (20 epochs)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path ./models/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-standard \
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
    --save_quant_dir ./models/llama-3-8b-w4a4kv4

# Expected time: ~4 hours on A100
# Expected perplexity: ~7.5 on WikiText2
```

### Use Case 2: Fast Quantization (No Fine-tuning)

**Goal:** Quick quantization for testing or demo

```bash
# W4A4KV4 without fine-tuning
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path ./models/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-fast \
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
    --save_quant_dir ./models/llama-3-8b-fast

# Expected time: ~30 min on A100
# Expected perplexity: ~8.0 on WikiText2 (slightly worse than fine-tuned)
```

### Use Case 3: High Accuracy Mode

**Goal:** Best possible accuracy with 8-bit activations

```bash
# W4A8KV4 (8-bit activations for better accuracy)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path ./models/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-high-acc \
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
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --save_quant_dir ./models/llama-3-8b-w4a8kv4

# Expected time: ~2 hours on A100
# Expected perplexity: ~6.8 on WikiText2 (closer to FP16)
```

### Use Case 4: Dynamic Quantization

**Goal:** Runtime-adaptive quantization

```bash
# Dynamic quantization with activation clipping
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --model_path ./models/llama-3-8B \
    --model_name Llama-3-8b \
    --output_dir ./log/llama-3-8b-dynamic \
    --wbits 4 \
    --input_bits 4 \
    --input_mode dynamic \
    --activation_clipping \
    --kv_mode dynamic \
    --eval_ppl \
    --save_quant_dir ./models/llama-3-8b-dynamic

# Expected time: ~1 hour on A100
# Note: Dynamic quantization is more flexible but may be slower at inference
```

### Use Case 5: Large Model (70B)

**Goal:** Quantize 70B parameter model across multiple GPUs

```bash
# Multi-GPU quantization for Llama-3-70B
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train.py \
    --model_path ./models/llama-3-70B \
    --model_name Llama-3-70b \
    --output_dir ./log/llama-3-70b-w4a4kv4 \
    --wbits 4 \
    --input_bits 4 \
    --input_mode static \
    --v_bits 4 \
    --k_bits 4 \
    --kv_group_size 128 \
    --kv_mode static \
    --quant_lr 2e-5 \
    --weight_lr 2e-6 \
    --epochs 20 \
    --max_memory 40 \
    --eval_ppl \
    --save_quant_dir ./models/llama-3-70b-w4a4kv4

# Expected time: ~16 hours on 4x A100 (80GB)
# Required memory: ~160 GB total (40GB per GPU)
```

### Use Case 6: Model Evaluation

**Goal:** Evaluate a quantized model on multiple benchmarks

```bash
# 1. Perplexity evaluation
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval_perplexity.py \
    --quant_model ./models/llama-3-8b-w4a4kv4 \
    --eval_ppl

# Output example:
# WikiText2 PPL: 7.52
# C4 PPL: 9.18

# 2. Downstream tasks
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval.py \
    --quant_model ./models/llama-3-8b-w4a4kv4 \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande

# Output example:
# PIQA: 78.5%
# ARC-Easy: 72.3%
# ARC-Challenge: 45.2%
# HellaSwag: 68.9%
# WinoGrande: 71.4%

# 3. All evaluations in one command
CUDA_VISIBLE_DEVICES=0 python benchmarks/eval.py \
    --quant_model ./models/llama-3-8b-w4a4kv4 \
    --eval_ppl \
    --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

### Use Case 7: Activation Analysis

**Goal:** Visualize and analyze model activations

```bash
# 1. Plot activation distributions
CUDA_VISIBLE_DEVICES=0 python experiments/plot_activation.py \
    --model_path ./models/llama-2-7b \
    --model_name llama-2-7b \
    --plot_linear_input \
    --output_dir ./plots/activations

# 2. Compare with and without rotation
# Without rotation
python experiments/plot_activation.py \
    --model_path ./models/llama-2-7b \
    --model_name llama-2-7b \
    --plot_linear_input \
    --output_dir ./plots/no_rotation

# With rotation
python experiments/plot_activation.py \
    --model_path ./models/llama-2-7b \
    --model_name llama-2-7b \
    --plot_linear_input \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens \
    --output_dir ./plots/with_rotation

# 3. Generate statistics
python experiments/stat_activation.py \
    --model_path ./models/llama-2-7b \
    --model_name llama-2-7b \
    --output_dir ./stats

# Output: activation_stats.json with layer-wise statistics
```

### Use Case 8: Custom Integration

**Goal:** Use PrefixQuant in your own Python code

```python
# custom_script.py
from src.utils.data_utils import get_loaders
from src.quantization.core.block_ap import block_ap
from src.utils.quant_utils import wrap_to_quant_model
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/llama-3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Wrap model for quantization
model = wrap_to_quant_model(
    model,
    wbits=4,
    input_bits=4,
    input_mode='static'
)

# Get calibration data
train_loader, test_loader = get_loaders(
    dataset_name='wikitext2',
    nsamples=128,
    seed=0,
    seqlen=2048
)

# Quantize model layer by layer
for block_idx, block in enumerate(model.model.layers):
    print(f"Quantizing block {block_idx}...")
    block_ap(
        block=block,
        train_loader=train_loader,
        dev=torch.device('cuda:0')
    )

# Save quantized model
torch.save(model.state_dict(), './models/custom_quantized.pth')
```

Run your custom script:
```bash
python custom_script.py
```

## Parameter Reference

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | **Required** | Path to base model |
| `--model_name` | str | **Required** | Model identifier (e.g., Llama-3-8b) |
| `--output_dir` | str | `./log` | Output directory for logs |
| `--save_quant_dir` | str | None | Directory to save quantized model |

### Weight Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--wbits` | int | 4 | Weight quantization bits (2, 3, 4, 8, 16) |
| `--weight_group_size` | int | 128 | Group size for weight quantization |

### Activation Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--input_bits` | int | 4 | Activation quantization bits (4, 6, 8, 16) |
| `--input_mode` | str | `static` | `static` or `dynamic` |
| `--input_group_size` | int | 128 | Group size for activation quantization |
| `--activation_clipping` | flag | False | Enable learnable activation clipping |

### KV Cache Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--k_bits` | int | 4 | Key cache quantization bits |
| `--v_bits` | int | 4 | Value cache quantization bits |
| `--kv_mode` | str | `static` | `static` or `dynamic` |
| `--kv_group_size` | int | 128 | Group size for KV cache |

### Rotation (Hadamard)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pre_rotate` | flag | False | Enable pre-rotation |
| `--down_online_had` | flag | False | Online Hadamard for down projection |
| `--qk_online_had` | flag | False | Online Hadamard for Q/K projections |

### PrefixQuant

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--set_prefixed_tokens` | flag | False | Enable prefixed tokens (core feature) |

### Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 0 | Number of fine-tuning epochs (0=no FT) |
| `--quant_lr` | float | 1e-4 | Learning rate for quantizers |
| `--weight_lr` | float | 1e-5 | Learning rate for weights |
| `--batch_size` | int | 1 | Batch size for training |
| `--loss_type` | str | `mse` | Loss function: `mse` or `skip_mse` |
| `--mse_init` | flag | False | Initialize quantizers with MSE |

### Evaluation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval_ppl` | flag | False | Evaluate perplexity |
| `--eval_tasks` | str | None | Comma-separated list of tasks |

### Hardware

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--max_memory` | int | 80 | Max memory per GPU (GB) |
| `--device` | str | `cuda:0` | CUDA device |

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
```bash
# Reduce max_memory
python scripts/train.py ... --max_memory 32

# Use gradient checkpointing (if available in config)
python scripts/train.py ... --gradient_checkpointing

# Reduce batch size
python scripts/train.py ... --batch_size 1

# Use smaller group sizes
python scripts/train.py ... --input_group_size 64 --kv_group_size 64
```

#### 2. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'quantize'
```

**Solution:**
```bash
# Update imports to new structure
# OLD: from quantize.block_ap import block_ap
# NEW: from src.quantization.core.block_ap import block_ap

# Or run from project root
cd /path/to/HELLM_python
python scripts/train.py ...
```

#### 3. Model Loading Issues

**Error:**
```
OSError: /path/to/model does not appear to be a folder containing valid model files
```

**Solution:**
```bash
# Check model path
ls /path/to/model  # Should contain config.json, pytorch_model.bin, etc.

# Verify model is downloaded
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./models/llama-3-8B

# Use absolute path
python scripts/train.py --model_path $(pwd)/models/llama-3-8B ...
```

#### 4. Slow Training

**Issue:** Training takes much longer than expected

**Solutions:**
```bash
# 1. Check GPU utilization
nvidia-smi  # Should show high GPU usage

# 2. Reduce epochs for testing
python scripts/train.py ... --epochs 5

# 3. Skip evaluation during training
python scripts/train.py ... --eval_ppl False

# 4. Use fewer calibration samples
# Edit src/utils/data_utils.py and reduce nsamples
```

#### 5. Poor Accuracy

**Issue:** Quantized model has high perplexity

**Solutions:**
```bash
# 1. Enable fine-tuning
python scripts/train.py ... --epochs 20

# 2. Use 8-bit activations
python scripts/train.py ... --input_bits 8

# 3. Enable all recommended features
python scripts/train.py \
    --mse_init \
    --pre_rotate \
    --down_online_had \
    --qk_online_had \
    --set_prefixed_tokens

# 4. For Llama-2-70B, use skip_mse loss
python scripts/train.py ... --loss_type skip_mse
```

## FAQ

### Q1: What's the difference between static and dynamic quantization?

**Static:** Quantization parameters (scale, zero-point) are pre-computed and fixed
- Pros: Faster inference, lower memory
- Cons: Less flexible

**Dynamic:** Quantization parameters are computed at runtime
- Pros: Better accuracy, adaptive to input
- Cons: Slower inference

### Q2: How long does quantization take?

Typical times on A100 (80GB):

| Model | Mode | Time |
|-------|------|------|
| Llama-3-8B | No FT | ~30 min |
| Llama-3-8B | 20 epochs FT | ~4 hours |
| Llama-3-70B | No FT | ~4 hours |
| Llama-3-70B | 20 epochs FT | ~16 hours |

### Q3: What GPUs do I need?

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| Llama-2-7B | 16 GB | 24 GB |
| Llama-3-8B | 24 GB | 40 GB |
| Llama-3-70B | 160 GB (4x40GB) | 320 GB (4x80GB) |

### Q4: Can I use CPU only?

**A:** Technically yes, but not recommended. Quantization will be extremely slow (days instead of hours).

```bash
# Force CPU (not recommended)
CUDA_VISIBLE_DEVICES="" python scripts/train.py ... --device cpu
```

### Q5: How do I convert to ONNX/TensorRT?

**A:** Currently, you need to manually export. Example:

```python
import torch
import torch.onnx

# Load quantized model
model = torch.load('./models/llama-3-8b-w4a4kv4/model.pth')
model.eval()

# Export to ONNX
dummy_input = torch.randint(0, 32000, (1, 128))
torch.onnx.export(
    model,
    dummy_input,
    './models/llama-3-8b-w4a4kv4.onnx',
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {1: 'sequence'}}
)
```

### Q6: What is the expected accuracy loss?

Typical perplexity on WikiText2:

| Configuration | Llama-3-8B PPL | Accuracy Loss |
|---------------|----------------|---------------|
| FP16 (baseline) | 6.14 | 0% |
| W4A8KV4 (no FT) | 6.85 | +11% |
| W4A8KV4 (20 epochs) | 6.45 | +5% |
| W4A4KV4 (no FT) | 8.12 | +32% |
| W4A4KV4 (20 epochs) | 7.52 | +22% |

### Q7: Can I quantize instruction-tuned models?

**A:** Yes! Use the same commands but with instruction-tuned model paths:

```bash
python scripts/train.py \
    --model_path ./models/Llama-3-8B-Instruct \
    --model_name Llama-3-8b-Instruct \
    ...
```

### Q8: How do I monitor progress?

```bash
# 1. Use tail to monitor logs
tail -f ./log/llama-3-8b-w4a4kv4/train.log

# 2. Use tensorboard (if enabled)
tensorboard --logdir ./log

# 3. Check output_dir for checkpoints
ls -lh ./log/llama-3-8b-w4a4kv4/
```

### Q9: Can I resume interrupted training?

**A:** Currently, you need to restart. We recommend using tmux/screen for long runs:

```bash
# Start tmux session
tmux new -s quantization

# Run training
python scripts/train.py ...

# Detach: Ctrl+B then D
# Reattach: tmux attach -t quantization
```

### Q10: Where can I get help?

1. Check [GitHub Issues](https://github.com/your-repo/HELLM_python/issues)
2. Read the [paper](https://arxiv.org/abs/2410.05265)
3. Check `./examples/` directory for working scripts
