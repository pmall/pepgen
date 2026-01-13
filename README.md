# PepGen - Peptide Generation with Discrete Diffusion

## The Approach

### The Problem

We want to generate novel peptide sequences (short protein fragments, 4-20 amino acids) that can interact with specific human proteins. This has applications in drug discovery - finding new peptides that bind to disease-related proteins could lead to new therapeutics.

We have two types of training data:
- **~2,000 known interaction pairs**: Experimentally validated (viral peptide, human protein) pairs from curated databases. Each pair tells us "this specific peptide binds to this specific human protein."
- **~500,000 random viral sequences**: Variable-length fragments (4-20 AA) sampled from viral proteins. These have no known interaction with human proteins. The length distribution matches the interacting sequences to prevent length-based bias.

### The Insight

Imagine the space of all possible peptide sequences as a vast landscape. Most sequences are "generic" - valid protein fragments that don't bind to anything interesting. But scattered throughout this landscape are rare sequences that can bind to human proteins.

Our training strategy uses both datasets with distinct purposes:

1. **The 500,000 random sequences teach structure**: The model learns what valid peptide sequences look like - amino acid frequencies, common motifs, structural patterns. This is the "background" of the landscape.

2. **The 2,000 interaction pairs mark the targets**: These samples show the model where the interesting regions are - the sequences that actually bind. Crucially, they also provide the mapping between peptides and their human protein targets.

### The Method: Discrete Diffusion (D3PM)

We use **D3PM** (Discrete Denoising Diffusion Probabilistic Models), adapted from the same family of generative models behind DALL-E and Stable Diffusion, but designed for discrete tokens (amino acids) instead of continuous pixels.

#### Training Phase: Learning the Landscape

1. Take a real peptide sequence (e.g., `MKTAYIAKQRQISFVK`)
2. Gradually corrupt it by randomly replacing amino acids with noise over many steps
3. Train the model to reverse this process - predict the original sequence from the noisy version
4. Provide two conditioning signals with each sequence:
   - **Interaction label**: Is this sequence interacting (1) or non-interacting (0)?
   - **Target protein ID**: Which human protein does it interact with? (or "unknown" for non-interacting)

The model learns to denoise differently based on these conditions - it learns the statistical patterns of interacting vs non-interacting sequences, and the patterns specific to each target protein.

#### Generation Phase: Guided Exploration

1. Start with pure random noise (random amino acids at each position)
2. Apply the learned denoising process iteratively (50 steps by default)
3. Condition on desired properties:
   - `label=1` → "Generate an interacting peptide"
   - `target=Q96C01` → "Make it interact with human protein Q96C01"
4. The model steers the denoising toward regions of sequence space matching these conditions
5. Output: A novel peptide sequence the model believes could interact

### Why This Works

The key insight is **conditional generation with imbalanced data**:

- The 500k non-interacting samples provide a strong prior on general peptide structure
- The 2k interacting samples, though rare, provide enough signal to learn the distinguishing features
- Balanced sampling during training ensures the model sees both classes equally despite the 100:1 imbalance
- At generation time, conditioning on `label=1` biases the output toward the interacting distribution

---

## Setup

```bash
uv sync
cp .env.example .env
# Edit .env with PostgreSQL credentials
```

## Usage

### 1. Generate Dataset

```bash
uv run python scripts/generate_dataset.py -v
```

### 2. Train Model

```bash
# Quick start with recommended preset
uv run python scripts/train.py --preset medium -v

# Name your run to avoid overwriting previous models
uv run python scripts/train.py --preset medium --run-name experiment1 -v

# Or customize parameters
uv run python scripts/train.py --epochs 300 --dim 384 --run-name custom1 -v
```

### 3. Generate Peptides

```bash
# Generate interacting peptides (any target)
uv run python scripts/generate.py -n 10

# Generate from a specific run
uv run python scripts/generate.py --run-name experiment1 -n 10

# Generate for specific human protein target
uv run python scripts/generate.py -t Q96C01 -n 10

# Generate with length constraints (8-12 amino acids)
uv run python scripts/generate.py --min-length 8 --max-length 12 -n 10

# Combine all conditions
uv run python scripts/generate.py --run-name experiment1 -t Q96C01 --min-length 10 --max-length 15 -n 20
```

---

## Run Management

By default, training saves checkpoints to `checkpoints/`. **Starting a new training run will overwrite existing files.**

To preserve multiple experiments, use `--run-name`:

```bash
# First experiment
uv run python scripts/train.py --preset medium --run-name baseline -v

# Second experiment with different settings
uv run python scripts/train.py --preset large --run-name large_model -v

# Compare by generating from each
uv run python scripts/generate.py --run-name baseline -n 10
uv run python scripts/generate.py --run-name large_model -n 10
```

This creates separate directories:
```
checkpoints/
├── baseline/
│   ├── best_model.pt
│   ├── config.json
│   ├── targets.json
│   └── tokenizer.txt
└── large_model/
    ├── best_model.pt
    ├── config.json
    ├── targets.json
    └── tokenizer.txt
```

---

## Training Configuration

### Presets

We provide three presets optimized for different scenarios:

| Preset | Use Case | VRAM | Training Time (GPU) | Quality |
|--------|----------|------|---------------------|---------|
| `small` | Quick experiments, debugging | <4GB | ~10 min | Lower |
| `medium` | **Recommended starting point** | ~6GB | ~30-60 min | Good |
| `large` | Best results, final training | >8GB | ~2-4 hours | Best |

```bash
uv run python scripts/train.py --preset small -v   # Fast experimentation
uv run python scripts/train.py --preset medium -v  # Recommended
uv run python scripts/train.py --preset large -v   # Best quality
```

### Preset Configurations

#### Small Preset
```
epochs: 50, batch_size: 32, lr: 2e-4, dim: 128, layers: 3, timesteps: 20
```
- **Purpose**: Rapid iteration, testing pipeline, low-resource environments
- **Trade-offs**: May underfit, lower generation quality, but trains in ~10 min on GPU

#### Medium Preset (Recommended)
```
epochs: 200, batch_size: 64, lr: 1e-4, dim: 256, layers: 4, timesteps: 50
```
- **Purpose**: Balanced quality and training time
- **Trade-offs**: Good results for most use cases, reasonable training time

#### Large Preset
```
epochs: 500, batch_size: 128, lr: 5e-5, dim: 512, layers: 6, timesteps: 100
```
- **Purpose**: Maximum quality for final model
- **Trade-offs**: Slower training, requires more VRAM, risk of overfitting without validation

### Parameter Reference

#### Training Parameters

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `--epochs` | 100 | Training epochs | With balanced sampling (~400k samples/epoch), 100-500 epochs typical. Watch loss plateau. |
| `--batch-size` | 64 | Batch size | Increase for faster training if VRAM allows. 32-256 typical. |
| `--lr` | 1e-4 | Learning rate | Higher (2e-4) for small models, lower (5e-5) for large. |
| `--warmup-steps` | 1000 | LR warmup steps | Prevents early training instability. Scale with dataset size. |

#### Model Architecture

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `--dim` | 256 | Embedding dimension | Model capacity. 128 (small) → 512 (large). Affects VRAM linearly. |
| `--num-layers` | 4 | Transformer layers | Depth. 3-6 typical. More layers = more capacity but slower. |
| `--num-heads` | 4 | Attention heads | Should divide `dim` evenly. 4-8 typical. |
| `--n-timesteps` | 50 | Diffusion steps | For peptides (20 tokens), 20-100 steps is sufficient. Unlike images, discrete sequences don't need 1000 steps. |

#### Loss Function

| Parameter | Default | Description | Tuning Guidance |
|-----------|---------|-------------|-----------------|
| `--hybrid-loss-coeff` | 0.001 | VB loss weight | Weight of variational bound vs cross-entropy. Start with 0, increase to 0.01 if generation quality is poor. |

### Understanding the Parameters

#### Why `dim` matters
The embedding dimension controls model capacity. With only 2,000 positive samples:
- Too large (512+): Risk of memorizing the training data
- Too small (64-128): May not capture the complexity of interactions
- Sweet spot: 256 for medium, 384-512 for large with regularization

#### Why `epochs` matters
Each epoch with balanced sampling sees ~400,000 samples (200k negative + 200k oversampled positive). The model needs enough epochs to:
1. Learn general peptide structure (fast, ~20 epochs)
2. Learn interaction patterns (slower, ~100 epochs)
3. Learn target-specific patterns (slowest, ~200+ epochs)

#### Why `hybrid_loss_coeff` matters
D3PM uses two loss components:
- **Cross-entropy (CE)**: Direct prediction of clean sequence from noisy
- **Variational bound (VB)**: Theoretical diffusion loss

In practice, CE alone often works well. Increase VB weight if:
- Generated sequences look random/incoherent
- The model converges too fast without quality improvement

#### Why learning rate warmup matters
Without warmup, large initial gradients can:
- Push embeddings to extreme values
- Cause unstable early training
- Lead to poor final convergence

Warmup gradually increases LR from 0 to target over `warmup_steps`, allowing stable initial learning.

### Monitoring Training

Watch for these patterns:

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Loss decreases, then flat | Normal convergence | Stop when flat for 10+ epochs |
| Loss oscillates wildly | LR too high | Reduce `--lr` by 2-5x |
| Loss decreases very slowly | LR too low | Increase `--lr` by 2x |
| Loss plateaus immediately | Model too small | Increase `--dim` or `--num-layers` |
| Loss goes to ~0 | Overfitting/memorization | Reduce model size, add early stopping |

### Recommended Training Procedure

1. **Start with medium preset**:
   ```bash
   uv run python scripts/train.py --preset medium -v
   ```

2. **Monitor the loss curve** - it should decrease steadily then plateau

3. **Check generated samples** every 10 epochs - they should look like valid peptides

4. **If overfitting** (loss too low, samples look like training data):
   - Reduce `--dim` to 192 or 128
   - Reduce `--epochs`
   - Increase `--hybrid-loss-coeff` to 0.01

5. **If underfitting** (poor generation quality):
   - Increase `--epochs` to 300-500
   - Increase `--dim` to 384 or 512
   - Check that `--balanced` is enabled

---

## Project Structure

```
pepgen/
├── data/                   # Dataset (gitignored)
│   └── dataset.csv         # Generated training data
├── checkpoints/            # Model checkpoints (gitignored)
│   ├── best_model.pt       # Best model weights
│   ├── config.json         # Model configuration
│   ├── targets.json        # Target protein mapping
│   └── tokenizer.txt       # Vocabulary
├── pepgen/                 # Python package
│   ├── __init__.py
│   ├── tokenizer.py        # Amino acid tokenization (24 tokens)
│   ├── dataset.py          # PyTorch datasets with balanced sampling
│   └── model.py            # D3PM + Transformer architecture
└── scripts/                # CLI scripts
    ├── generate_dataset.py # Database → CSV dataset
    ├── train.py            # Training with presets
    └── generate.py         # Conditional generation
```

## References

- [D3PM Paper](https://arxiv.org/abs/2107.03006): Structured Denoising Diffusion Models in Discrete State-Spaces (Austin et al., 2021)
- [d3pm implementation](https://github.com/cloneofsimo/d3pm): Minimal PyTorch D3PM by @cloneofsimo
