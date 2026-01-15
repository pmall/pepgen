# Mimir Training Guide

## Executive Summary

Training the **large preset** (21.5M parameters, 150 epochs) on CPU is impractical (~26 days). 
A GPU is required, with estimated training times of **6-15 hours** depending on hardware.

## Recommended Strategy: Colab Free → Colab Pro

| Phase | Preset | Platform | Time | Purpose |
|-------|--------|----------|------|---------|
| 1. Quick test | Small | Colab Free (T4) | ~20-30 min | Verify notebook runs, dependencies work |
| 2. Full validation | Medium | Colab Free (T4) | ~4 hours | Validate training loop, convergence, checkpoints |
| 3. Final training | Large | Colab Pro (A100) | ~6-10 hours | Train production model |

**Why this approach:**
- Same notebook works on both Free and Pro (no code changes needed)
- Small model gives fast feedback (~30 min) to catch obvious issues
- Medium model validates full training works within Free tier limits
- Only subscribe to Pro after confirming everything works
- Pro subscription is only $10/month, cancelled after training

---

## Benchmark Results (Measured on CPU)

| Preset | Parameters | Batch Size | Batches/Epoch | Time/Batch | Time/Epoch | Total (CPU) |
|--------|-----------|------------|---------------|------------|------------|-------------|
| Small | 839K | 32 | 31,359 | 0.05s | 25 min | 8 hours (20 epochs) |
| Medium | 3.9M | 64 | 15,680 | 0.22s | 59 min | 73 hours (75 epochs) |
| Large | 21.5M | 128 | 7,840 | 1.94s | 4.2 hours | 634 hours (150 epochs) |

---

## GPU Speedup Estimation

### How GPU Time is Estimated from CPU Benchmarks

GPU acceleration comes from three main factors:

1. **Parallel matrix multiplication**: GPUs have thousands of cores vs CPU's ~8-16 cores
2. **Memory bandwidth**: GPU (300-900 GB/s) vs CPU (~50 GB/s)  
3. **Batch efficiency**: Larger batches utilize GPU parallelism better

For transformer models like ours (21.5M params, batch_size=128), typical speedups are:

| GPU | Memory Bandwidth | Tensor Cores | Expected Speedup |
|-----|-----------------|--------------|------------------|
| T4 (Colab free) | 320 GB/s | Yes (FP16) | 15-25x |
| V100 | 900 GB/s | Yes | 20-35x |
| A100 | 1,555 GB/s | Yes (TF32) | 40-60x |

### Estimated Training Times by Preset and GPU

| Preset | CPU Time | T4 (15-25x) | A100 (40-60x) |
|--------|----------|-------------|---------------|
| Small (20 epochs) | 8 hours | 20-30 min | 8-12 min |
| Medium (75 epochs) | 73 hours | **3.5-5 hours** ✅ | 1-2 hours |
| Large (150 epochs) | 634 hours | 25-42 hours ❌ | **6-10 hours** ✅ |

✅ = Fits in session limits, ❌ = Exceeds session limits

**Key insight**: Medium on T4 (3.5-5h) fits within Colab Free's 4-12h session limit, making it perfect for validation before committing to Pro.

---

## Google Colab Analysis

### Free vs Pro Comparison

| Feature | Colab Free | Colab Pro ($10/month) |
|---------|------------|----------------------|
| GPU | T4 (16GB) | T4, V100, or **A100** (priority) |
| Session limit | 4-12 hours (variable) | 24 hours |
| Idle timeout | 90 minutes | 90 minutes (with background exec) |
| Suitable for | Medium preset (~4h) | Large preset (~6-10h on A100) |

### Key Advantage: Same Notebook

The same notebook works on both Free and Pro tiers. This enables our strategy:

1. **Develop and test on Free** - no cost, catch all issues early
2. **Switch to Pro for final training** - just change runtime type to A100

### What the Free Tier Validates

| Validation | Why It Matters |
|------------|----------------|
| Data upload works | Dataset transfers correctly to Colab |
| Dependencies install | All packages available |
| Memory fits | Model + batch fits in GPU RAM |
| Training loop runs | No code bugs or crashes |
| Checkpoints save | Can download trained model |
| Loss decreases | Model is learning correctly |

If medium trains successfully, large will too (just takes longer).

---

## Training Strategy (Step-by-Step)

### Phase 1: Quick Test with Small (Cost: $0, ~30 min)

1. Open notebook in Google Colab
2. Select **T4 GPU** runtime
3. Upload dataset
4. Run training with `--preset small`
5. Verify:
   - Dependencies install correctly
   - Dataset loads without errors
   - Training loop runs
   - GPU is being utilized
   - Loss decreases over first few epochs

### Phase 2: Full Validation with Medium (Cost: $0, ~4 hours)

1. Same notebook, same T4 runtime
2. Run training with `--preset medium`
3. Verify:
   - Training completes fully (~4 hours)
   - Loss curve shows proper convergence
   - Sample generations look like valid peptides
   - Checkpoint saves and downloads correctly

### Phase 3: Train Large on Colab Pro (Cost: ~$10, ~6-10 hours)

1. Subscribe to Colab Pro
2. Open same notebook
3. Select **A100 GPU** runtime (Pro benefit)
4. Run training with `--preset large`
5. Training completes in ~6-10 hours
6. Download final checkpoint
7. Cancel Pro subscription if desired

### Why This Strategy Works

| Risk | How We Mitigate |
|------|-----------------|
| Notebook doesn't work | Caught in Phase 1 (free) |
| Dependencies broken | Caught in Phase 1 (free) |
| Out of memory | Caught in Phase 1 (free) |
| Training doesn't converge | Medium results predict large behavior |
| Wasted Pro subscription | Only subscribe after validation |

---

## Alternative Options (For Reference)

If Colab doesn't work for some reason:

| Platform | Cost | Training Time (Large) | Setup Difficulty |
|----------|------|----------------------|------------------|
| Vast.ai | ~$5-15 | 10-20 hours | Medium |
| RunPod | ~$5-15 | 10-20 hours | Medium |
| Lambda Labs | ~$20-30 | 10-16 hours | Easy |
| AWS/GCP Spot | ~$10-25 | 10-20 hours | Hard |

---

## Memory Requirements

| Preset | Model Size | Batch 128 VRAM | Batch 64 VRAM |
|--------|-----------|----------------|---------------|
| Small | ~3 MB | ~2 GB | ~1.5 GB |
| Medium | ~12 MB | ~4 GB | ~3 GB |
| Large | ~82 MB | ~8 GB | ~5 GB |

The large preset fits comfortably on a T4 (16GB) with batch_size=128.

---

## Quick Start Commands

```bash
# Phase 1: Quick test with small on Colab Free (T4) - ~30 min
python scripts/train.py --preset small --run-name small_test -v

# Phase 2: Full validation with medium on Colab Free (T4) - ~4 hours
python scripts/train.py --preset medium --run-name medium_test -v

# Phase 3: Train large on Colab Pro (A100) - ~6-10 hours
python scripts/train.py --preset large --run-name large_final -v
```

---

## Resources

- **Notebook**: `notebooks/train_colab.ipynb` - Self-contained Colab notebook with instructions
