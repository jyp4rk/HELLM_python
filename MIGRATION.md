# Migration Guide: Codebase Restructuring

This guide helps you migrate existing code to the new reorganized structure.

## 🔄 What Changed?

The codebase was reorganized for better maintainability:

- **Old:** Scattered files, duplicate code, unclear organization
- **New:** Clean structure with `src/`, `scripts/`, `tests/`, `benchmarks/`

## 📋 Quick Migration Checklist

- [ ] Update all import statements
- [ ] Update script execution commands
- [ ] Update any hardcoded paths
- [ ] Test your code with new structure
- [ ] Update documentation/notebooks

## 🗺️ Directory Mapping

### Files Moved

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `main.py` | `scripts/train.py` | Main training script |
| `eval.py` | `benchmarks/eval.py` | Evaluation script |
| `eval_perplexity.py` | `benchmarks/eval_perplexity.py` | Perplexity evaluation |
| `plot_activation.py` | `experiments/plot_activation.py` | Visualization |
| `test_*.py` | `tests/test_*.py` | All test files |
| `quantize/` | `src/quantization/` | Quantization module |
| `utils/` | `src/utils/` | Core utilities |
| `train_utils/` | `src/training/` | Training utilities |
| `eval_utils/` | `benchmarks/eval_utils/` | Evaluation utilities |

## 📦 Module Import Changes

### Pattern 1: Quantization Modules

```python
# ❌ OLD (deprecated)
from quantize.quantizer import UniformAffineQuantizer
from quantize.block_ap import block_ap
from quantize.int_linear_fake import QuantLinear
from quantize.int_linear_real import load_quantized_model
from quantize.recon_loss import get_recon_loss
from quantize.quant_norm import QuantRMSNorm
from quantize.triton_utils.kernels import dequant_dim0

# ✅ NEW (current)
from src.quantization.core.quantizer import UniformAffineQuantizer
from src.quantization.core.block_ap import block_ap
from src.quantization.linear.int_linear_fake import QuantLinear
from src.quantization.linear.int_linear_real import load_quantized_model
from src.quantization.losses.recon_loss import get_recon_loss
from src.quantization.layers.quant_norm import QuantRMSNorm
from src.quantization.triton_utils.kernels import dequant_dim0
```

### Pattern 2: Utility Modules

```python
# ❌ OLD
from utils.data_utils import get_loaders, test_ppl
from utils.model_utils import get_kv_cache
from utils.rotation_utils import rotate_model
from utils.hadamard_utils import apply_exact_had_to_linear
from utils.quant_utils import wrap_to_quant_model
from utils.ckks_utils import inject_noise_model
import utils.model_utils as model_utils

# ✅ NEW
from src.utils.data_utils import get_loaders, test_ppl
from src.utils.model_utils import get_kv_cache
from src.utils.rotation_utils import rotate_model
from src.utils.hadamard_utils import apply_exact_had_to_linear
from src.utils.quant_utils import wrap_to_quant_model
from src.utils.ckks_utils import inject_noise_model
import src.utils.model_utils as model_utils
```

### Pattern 3: Training Modules

```python
# ❌ OLD
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.optimizer import create_optimizer
from train_utils.quant_linear import QuantizeLinear
from train_utils.noisy_linear import NoisyLinear

# ✅ NEW
from src.training.fsdp_trainer import FSDPTrainer
from src.training.optimizer import create_optimizer
from src.training.quant_linear import QuantizeLinear
from src.models.components.noisy_linear import NoisyLinear  # Note: moved to models!
```

### Pattern 4: Model Modules

```python
# ❌ OLD
from eval_utils.modeling_llama import LlamaForCausalLM
from train_utils.modeling_llama_CKKS import LlamaForCausalLMCKKS
from train_utils.modeling_llama_quant import LlamaForCausalLMQuant

# ✅ NEW
from src.models.llama import LlamaForCausalLM
from src.models.llama_ckks import LlamaForCausalLMCKKS
from src.models.llama_quant import LlamaForCausalLMQuant
```

### Pattern 5: Evaluation Utilities

```python
# ❌ OLD
from eval_utils.rotation_utils import apply_hadamard
from eval_utils.gptq_utils import gptq_quantize
import eval_utils.main as eval_main

# ✅ NEW
from benchmarks.eval_utils.rotation_utils import apply_hadamard
from benchmarks.eval_utils.gptq_utils import gptq_quantize
import benchmarks.eval_utils.main as eval_main
```

## 🔧 Complete Import Mapping Table

| Old Import | New Import |
|------------|------------|
| `quantize.*` | `src.quantization.*` |
| `quantize.quantizer` | `src.quantization.core.quantizer` |
| `quantize.block_ap` | `src.quantization.core.block_ap` |
| `quantize.fixed_point` | `src.quantization.core.fixed_point` |
| `quantize.int_linear_fake` | `src.quantization.linear.int_linear_fake` |
| `quantize.int_linear_real` | `src.quantization.linear.int_linear_real` |
| `quantize.fixed_linear_fake` | `src.quantization.linear.fixed_linear_fake` |
| `quantize.quant_norm` | `src.quantization.layers.quant_norm` |
| `quantize.noisy_norm` | `src.quantization.layers.noisy_norm` |
| `quantize.noisy_swish` | `src.quantization.layers.noisy_swish` |
| `quantize.recon_loss` | `src.quantization.losses.recon_loss` |
| `quantize.triton_utils` | `src.quantization.triton_utils` |
| `utils.*` | `src.utils.*` |
| `train_utils.*` | `src.training.*` |
| `train_utils.noisy_linear*` | `src.models.components.noisy_linear*` |
| `train_utils.modeling_llama_*` | `src.models.llama_*` |
| `eval_utils.modeling_llama` | `src.models.llama` |
| `eval_utils.*` | `benchmarks.eval_utils.*` |

## 💻 Command Line Changes

### Training Commands

```bash
# ❌ OLD
python main.py --model_path /path/to/model ...

# ✅ NEW
python scripts/train.py --model_path /path/to/model ...
```

### Evaluation Commands

```bash
# ❌ OLD
python eval.py --quant_model /path/to/model ...
python eval_perplexity.py --quant_model /path/to/model ...
python eval_perplexity_ckks.py --quant_model /path/to/model ...
python eval_softmax_sink.py --model_path /path/to/model ...

# ✅ NEW
python benchmarks/eval.py --quant_model /path/to/model ...
python benchmarks/eval_perplexity.py --quant_model /path/to/model ...
python benchmarks/eval_perplexity_ckks.py --quant_model /path/to/model ...
python benchmarks/eval_softmax_sink.py --model_path /path/to/model ...
```

### Experiment Commands

```bash
# ❌ OLD
python plot_activation.py --model_path /path/to/model ...
python stat_activation.py --model_path /path/to/model ...
python optimize_rotation.py --model_path /path/to/model ...

# ✅ NEW
python experiments/plot_activation.py --model_path /path/to/model ...
python experiments/stat_activation.py --model_path /path/to/model ...
python experiments/optimize_rotation.py --model_path /path/to/model ...
```

### Test Commands

```bash
# ❌ OLD
python test_q_k_up_gate.py
python test_ckks_noise_injection.py

# ✅ NEW
python tests/test_q_k_up_gate.py
# Or use pytest:
pytest tests/test_q_k_up_gate.py
pytest tests/test_ckks_noise_injection.py
```

## 📝 Automated Migration Script

You can use this script to automatically update your Python files:

```bash
#!/bin/bash
# migrate_imports.sh

# Update quantize imports
find . -name "*.py" -type f -exec sed -i \
    's/from quantize\./from src.quantization./g;
     s/import quantize\./import src.quantization./g' {} \;

# Update utils imports
find . -name "*.py" -type f -exec sed -i \
    's/from utils\./from src.utils./g;
     s/import utils\./import src.utils./g' {} \;

# Update train_utils imports
find . -name "*.py" -type f -exec sed -i \
    's/from train_utils\./from src.training./g;
     s/import train_utils\./import src.training./g' {} \;

# Update eval_utils imports
find . -name "*.py" -type f -exec sed -i \
    's/from eval_utils\./from benchmarks.eval_utils./g;
     s/import eval_utils\./import benchmarks.eval_utils./g' {} \;

# Fix noisy_linear special case
find . -name "*.py" -type f -exec sed -i \
    's/from src.training.noisy_linear/from src.models.components.noisy_linear/g' {} \;

# Fix modeling_llama special cases
find . -name "*.py" -type f -exec sed -i \
    's/from src.training.modeling_llama_CKKS/from src.models.llama_ckks/g;
     s/from src.training.modeling_llama_quant/from src.models.llama_quant/g;
     s/from benchmarks.eval_utils.modeling_llama/from src.models.llama/g' {} \;

echo "Migration complete! Please review changes and test your code."
```

**Usage:**
```bash
chmod +x migrate_imports.sh
./migrate_imports.sh
```

## 🧪 Testing Your Migration

### Step 1: Syntax Check

```bash
# Check all Python files compile
find . -name "*.py" -not -path "./.git/*" -not -path "./cache/*" \
    -exec python -m py_compile {} \; 2>&1 | grep -v "^$"

# If no output, all files are syntactically correct!
```

### Step 2: Import Test

```python
# test_imports.py
try:
    # Test core imports
    from src.utils.data_utils import get_loaders
    from src.quantization.core.block_ap import block_ap
    from src.training.optimizer import create_optimizer
    from src.models.llama import LlamaForCausalLM
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

```bash
python test_imports.py
```

### Step 3: Run Tests

```bash
# Run test suite
pytest tests/ -v

# Run a simple test
pytest tests/test_prefix_cache.py -v
```

### Step 4: Test Commands

```bash
# Test training script (dry run)
python scripts/train.py --help

# Test evaluation script
python benchmarks/eval.py --help

# Test experiment script
python experiments/plot_activation.py --help
```

## 🔍 Common Migration Issues

### Issue 1: Cannot Import Module

**Error:**
```python
ModuleNotFoundError: No module named 'quantize'
```

**Solution:**
```python
# Update the import
# OLD: from quantize.block_ap import block_ap
# NEW: from src.quantization.core.block_ap import block_ap
```

### Issue 2: Script Not Found

**Error:**
```bash
python main.py
# python: can't open file 'main.py': [Errno 2] No such file or directory
```

**Solution:**
```bash
# Use new script location
python scripts/train.py
```

### Issue 3: Relative Import Errors

**Error:**
```python
ImportError: attempted relative import with no known parent package
```

**Solution:**
```bash
# Run from project root
cd /path/to/HELLM_python
python scripts/train.py ...

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/HELLM_python:$PYTHONPATH
```

### Issue 4: Hardcoded Paths

**Error:**
```python
# Your code has:
sys.path.append('./utils')  # This will break!
```

**Solution:**
```python
# Update to:
sys.path.append('./src/utils')

# Or better, use proper imports:
from src.utils import data_utils
```

## 📚 Real-World Migration Examples

### Example 1: Custom Training Script

**Before:**
```python
# my_train.py (OLD)
import sys
sys.path.append('./')

from utils.data_utils import get_loaders
from quantize.block_ap import block_ap
from train_utils.optimizer import create_optimizer
from eval_utils.modeling_llama import LlamaForCausalLM

def main():
    model = LlamaForCausalLM.from_pretrained("...")
    train_loader, test_loader = get_loaders("wikitext2")
    # ... rest of code
```

**After:**
```python
# my_train.py (NEW)
# No need for sys.path manipulation!

from src.utils.data_utils import get_loaders
from src.quantization.core.block_ap import block_ap
from src.training.optimizer import create_optimizer
from src.models.llama import LlamaForCausalLM

def main():
    model = LlamaForCausalLM.from_pretrained("...")
    train_loader, test_loader = get_loaders("wikitext2")
    # ... rest of code (unchanged)
```

### Example 2: Jupyter Notebook

**Before:**
```python
# notebook.ipynb (OLD)
import sys
sys.path.append('/path/to/HELLM_python')

from utils.data_utils import get_loaders
from quantize.int_linear_real import load_quantized_model
import utils.model_utils as model_utils
```

**After:**
```python
# notebook.ipynb (NEW)
import sys
sys.path.append('/path/to/HELLM_python')  # Still needed for notebooks

from src.utils.data_utils import get_loaders
from src.quantization.linear.int_linear_real import load_quantized_model
import src.utils.model_utils as model_utils
```

### Example 3: Bash Script

**Before:**
```bash
#!/bin/bash
# run_experiments.sh (OLD)

python main.py --model_path /models/llama-3-8B --wbits 4
python eval.py --quant_model /models/llama-3-8b-quantized
python plot_activation.py --model_path /models/llama-3-8B
```

**After:**
```bash
#!/bin/bash
# run_experiments.sh (NEW)

python scripts/train.py --model_path /models/llama-3-8B --wbits 4
python benchmarks/eval.py --quant_model /models/llama-3-8b-quantized
python experiments/plot_activation.py --model_path /models/llama-3-8B
```

## ✅ Migration Verification Checklist

After migration, verify:

- [ ] All imports work without errors
- [ ] Scripts can be executed from project root
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Training command works: `python scripts/train.py --help`
- [ ] Evaluation command works: `python benchmarks/eval.py --help`
- [ ] Existing quantized models still load correctly
- [ ] Custom scripts/notebooks updated and tested
- [ ] Documentation updated with new paths
- [ ] Team members notified of changes

## 🆘 Getting Help

If you encounter issues during migration:

1. **Check this guide** - Most common issues are covered
2. **Search existing issues** - [GitHub Issues](https://github.com/your-repo/HELLM_python/issues)
3. **Create new issue** - Include error message and code snippet
4. **Ask the team** - Contact maintainers

## 📖 Additional Resources

- [README.md](README.md) - Updated usage instructions
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed examples and commands
- [examples/](examples/) - Working example scripts
- [tests/](tests/) - Example test files showing proper imports

---

**Note:** The old structure is no longer supported. All new code should use the new import paths.
