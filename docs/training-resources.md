# Mimir Training Guide

## Executive Summary

Training requires a GPU. All presets use **batch_size=256**, which is optimal for T4 GPU.

## Recommended Strategy: Colab Free → Colab Pro

| Phase | Preset | Platform | Purpose |
|-------|--------|----------|---------|
| 1. Quick test | Small | Colab Free (T4) | Verify notebook runs, dependencies work |
| 2. Validation | Medium | Colab Free (T4) | Validate convergence, check if model improves over small |
| 3. Production | Large | Colab Pro (A100) | Train final model |

**Why this approach:**
- Same notebook works on both Free and Pro
- Small model converges in ~5 epochs, gives fast feedback
- Medium model shows if larger capacity helps
- Only subscribe to Pro after confirming everything works

---

## Empirical Observations (T4 GPU)

### Batch Size

| Batch Size | Speed | Notes |
|------------|-------|-------|
| 32 | 63 it/s | Baseline |
| 256 | ~48 it/s per batch, but 8x fewer batches | **10x faster overall** |
| 512 | ~31 it/s per batch | Only 25% faster than 256, diminishing returns |

**Conclusion:** batch_size=256 is optimal for T4.

### Convergence (Small Model)

| Epoch | CE Loss | Notes |
|-------|---------|-------|
| 1 | 1.17 | |
| 2 | 1.09 | |
| 3 | 1.09 | Plateau begins |
| 4 | 1.09 | |
| 10 | 1.09 | No further improvement |

**Observations:**
- Small model (839K params) plateaus at ce≈1.09 by epoch 3-4
- This is a **capacity limit**, not a training time issue
- Random baseline would be ce≈3.2 (log of 25 vocab tokens)
- Model is learning but limited by size

### Loss Interpretation

| Metric | Meaning |
|--------|---------|
| `ce` | Cross-entropy: prediction accuracy (lower = better) |
| `vb` | Variational bound: KL divergence term |
| `loss` | Total: `ce + hybrid_loss_coeff * vb` |

Reference points for ce:
- Random guessing: ~3.2
- Small model plateau: ~1.09
- Target for larger models: <1.0

---

## Preset Configuration

All presets now use batch_size=256.

| Preset | Parameters | Epochs | Batch Size | Learning Rate | Batches/Epoch |
|--------|-----------|--------|------------|---------------|---------------|
| Small | 839K | 20 | 256 | 5e-4 | 3,920 |
| Medium | 3.9M | 75 | 256 | 3e-4 | 3,920 |
| Large | 21.5M | 150 | 256 | 1e-4 | 3,920 |

Learning rates are set fresh for batch_size=256 (not extrapolated from smaller batches).

---

## Training Times (T4 GPU, Measured)

| Preset | Time/Epoch | Epochs | Total |
|--------|------------|--------|-------|
| Small | ~1.5 min | 20 | ~30 min |
| Medium | TBD | 75 | TBD |
| Large | TBD | 150 | TBD |

*Note: Epoch counts may be reduced based on convergence observations.*

---

## Google Colab

### Free vs Pro

| Feature | Colab Free | Colab Pro ($10/month) |
|---------|------------|----------------------|
| GPU | T4 (16GB) | T4, V100, or A100 |
| Session limit | 4-12 hours (variable) | 24 hours |
| Idle timeout | 90 minutes | 90 minutes |

### What Free Tier Validates

| Validation | Why It Matters |
|------------|----------------|
| Data upload works | Dataset transfers correctly |
| Dependencies install | All packages available |
| Memory fits | Model + batch fits in GPU RAM |
| Training loop runs | No code bugs or crashes |
| Loss decreases | Model is learning |

---

## Quick Start

```bash
# Small preset (~30 min on T4)
python scripts/train.py --preset small --run-name small_test -v

# Medium preset
python scripts/train.py --preset medium --run-name medium_test -v

# Large preset
python scripts/train.py --preset large --run-name large_final -v
```

---

## Resources

- **Notebook**: `notebooks/train_colab.ipynb` - Self-contained Colab notebook with instructions
