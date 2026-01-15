# Mimir - Peptide Generation with Discrete Diffusion

## The Approach

### The Problem

We want to generate novel peptide sequences (short protein fragments, 4-20 amino acids) that can interact with specific human proteins. This has applications in drug discovery - finding new peptides that bind to disease-related proteins could lead to new therapeutics.

We have two types of training data:

- **~3,500 known interaction pairs**: Experimentally validated peptide-target pairs from curated databases (both human-human and virus-human interactions). Each pair tells us "this specific peptide binds to this specific human protein."
- **~1,000,000 background sequences**: Random peptides (4-20 AA) sampled evenly from human proteins (500k) and viral proteins (500k). These have no known interaction with human proteins. The length distribution matches the interacting sequences to prevent length-based bias.

### The Insight

Imagine the space of all possible peptide sequences as a vast landscape. Most sequences are "generic" - valid protein fragments that don't bind to anything interesting. But scattered throughout this landscape are rare sequences that can bind to human proteins.

Our training strategy uses both datasets with distinct purposes:

1. **The 1M background sequences teach structure**: The model learns what valid peptide sequences look like - amino acid frequencies, common motifs, structural patterns. This is the "background" of the landscape.

2. **The 3.5k interaction pairs mark the targets**: These samples show the model where the interesting regions are - the sequences that actually bind. Crucially, they also provide the mapping between peptides and their human protein targets.

### The Method: Masked Diffusion

We use **Masked Diffusion**, a discrete generative model similar to BERT's masked language modeling, but adapted for iterative generation.

#### Training Phase: Learning to Fill the Blanks

1. Take a real peptide sequence (e.g., `MKTAYIAKQRQISFVK`)
2. Randomly mask a fraction of positions with `[MASK]` tokens
3. Train the model to predict the original amino acids at masked positions
4. Provide two conditioning signals:
   - **Interaction label**: Is this sequence interacting (1) or background (0)?
   - **Target protein ID**: Which human protein does it interact with?

**Linear Masking Schedule**: The number of masked positions scales with both timestep and sequence length:

```
num_masks = ceil(t × L / T)
```

This ensures short peptides (4 AA) and long peptides (20 AA) both see all masking levels during training.

#### Generation Phase: Iterative Unmasking

1. Start fully masked: `[MASK][MASK][MASK]...[PAD][PAD]`
2. For each timestep t = T, T-1, ..., 1:
   - Predict amino acids for all positions
   - Unmask highest-confidence predictions first
3. Condition generation with:
   - `label=1` → "Generate an interacting peptide"
   - `target=Q96C01` → "Make it interact with protein Q96C01"
4. Output: A novel peptide the model believes could interact

### Why This Works

The key insight is **conditional generation without class balancing**:

- The 1M background samples teach the model what valid peptides look like
- The 3.5k interacting samples provide enough signal to learn interaction patterns
- Unlike classifiers, diffusion models learn conditional distributions - no balanced classes needed
- At generation time, conditioning steers toward interacting sequences
- No oversampling means no risk of memorizing the small interacting set

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

This fetches interaction data from the database and generates `data/dataset.csv` with:

- Interacting pairs from human-human (hh) and virus-human (vh) interactions
- Background peptides sampled evenly from human and viral proteins

### 2. Verify Dataset

```bash
uv run python scripts/verify_dataset.py -v
```

Checks all dataset constraints:

- Peptide lengths (4-20 AA)
- Unique sequences
- No overlap between background and interacting sets
- Length distribution matching

### 3. Train Model

```bash
# Quick start with recommended preset
uv run python scripts/train.py --preset medium -v

# Name your run to avoid overwriting previous models
uv run python scripts/train.py --preset medium --run-name experiment1 -v

# Or customize parameters
uv run python scripts/train.py --epochs 100 --dim 384 --run-name custom1 -v
```

### 4. Generate Peptides

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

| Preset   | Use Case                       | VRAM | Training Time (GPU) | Quality |
| -------- | ------------------------------ | ---- | ------------------- | ------- |
| `small`  | Quick experiments, debugging   | <4GB | ~20 min             | Lower   |
| `medium` | **Recommended starting point** | ~6GB | ~1-2 hours          | Good    |
| `large`  | Best results, final training   | >8GB | ~4-8 hours          | Best    |

```bash
uv run python scripts/train.py --preset small -v   # Fast experimentation
uv run python scripts/train.py --preset medium -v  # Recommended
uv run python scripts/train.py --preset large -v   # Best quality
```

### Preset Configurations

#### Small Preset

```
epochs: 20, batch_size: 32, lr: 2e-4, dim: 128, layers: 3, timesteps: 20
```

- **Purpose**: Rapid iteration, testing pipeline, low-resource environments
- **Trade-offs**: May underfit, lower generation quality, but trains in ~20 min on GPU

#### Medium Preset (Recommended)

```
epochs: 75, batch_size: 64, lr: 1e-4, dim: 256, layers: 4, timesteps: 20
```

- **Purpose**: Balanced quality and training time
- **Trade-offs**: Good results for most use cases, ~1-2 hours training

#### Large Preset

```
epochs: 150, batch_size: 128, lr: 5e-5, dim: 512, layers: 6, timesteps: 20
```

- **Purpose**: Maximum quality for final model
- **Trade-offs**: ~4-8 hours training, requires more VRAM, risk of overfitting without validation

### Parameter Reference

#### Training Parameters

| Parameter        | Default | Description     | Tuning Guidance                                                    |
| ---------------- | ------- | --------------- | ------------------------------------------------------------------ |
| `--epochs`       | 75      | Training epochs | With ~1M samples/epoch, 20-150 epochs typical. Watch loss plateau. |
| `--batch-size`   | 64      | Batch size      | Increase for faster training if VRAM allows. 32-256 typical.       |
| `--lr`           | 1e-4    | Learning rate   | Higher (2e-4) for small models, lower (5e-5) for large.            |
| `--warmup-steps` | 1000    | LR warmup steps | Prevents early training instability. Scale with dataset size.      |

#### Model Architecture

| Parameter       | Default | Description         | Tuning Guidance                                                      |
| --------------- | ------- | ------------------- | -------------------------------------------------------------------- |
| `--dim`         | 256     | Embedding dimension | Model capacity. 128 (small) → 512 (large). Affects VRAM linearly.    |
| `--num-layers`  | 4       | Transformer layers  | Depth. 3-6 typical. More layers = more capacity but slower.          |
| `--num-heads`   | 4       | Attention heads     | Should divide `dim` evenly. 4-8 typical.                             |
| `--n-timesteps` | 20      | Diffusion steps     | Must be >= max_length (20). This is the minimum for proper training. |

#### Loss Function

Masked Diffusion uses cross-entropy loss computed **only on masked positions** (BERT-style). This forces the model to learn context rather than simply copying visible tokens.

#### Sampling

| Parameter    | Default | Description                  | Tuning Guidance                                                   |
| ------------ | ------- | ---------------------------- | ----------------------------------------------------------------- |
| `--balanced` | False   | Oversample interacting class | Not recommended. Risks memorizing the 3.5k interacting sequences. |

### Understanding the Parameters

#### Why `dim` matters

The embedding dimension controls model capacity. With only ~3,500 positive samples:

- Too large (512+): Risk of memorizing the training data
- Too small (64-128): May not capture the complexity of interactions
- Sweet spot: 256 for medium, 384-512 for large with regularization

#### Why `epochs` matters

Each epoch sees ~1M samples (the full dataset). The model needs enough epochs to:

1. Learn general peptide structure (fast, ~5-10 epochs)
2. Learn interaction patterns (slower, ~30-50 epochs)
3. Learn target-specific patterns (slowest, ~75-150 epochs)

#### Why `n_timesteps` matters

Masked Diffusion uses timesteps to control the masking level via the **linear masking formula**:

```
num_masks = ceil(t × L / T)
```

- `n_timesteps` must be >= max_length (20) to ensure all mask counts (1 to L) are trained
- Higher values are allowed but provide diminishing returns
- Default of 20 matches max peptide length for optimal efficiency

#### Why unbalanced sampling (default)

Unlike classifiers, diffusion models learn conditional distributions `P(sequence | label)`. With unbalanced data:

- The 1M background samples teach what valid peptides look like
- The 3.5k interacting samples are enough to learn the conditional signal
- No oversampling = no memorization risk
- Conditioning steers generation toward interacting patterns at inference time

The `--balanced` flag is available but not recommended - it oversamples the 3.5k interacting sequences ~285x per epoch, risking memorization instead of generalization.

#### Why learning rate warmup matters

Without warmup, large initial gradients can:

- Push embeddings to extreme values
- Cause unstable early training
- Lead to poor final convergence

Warmup gradually increases LR from 0 to target over `warmup_steps`, allowing stable initial learning.

### Monitoring Training

Watch for these patterns:

| Pattern                    | Diagnosis                | Action                                |
| -------------------------- | ------------------------ | ------------------------------------- |
| Loss decreases, then flat  | Normal convergence       | Stop when flat for 10+ epochs         |
| Loss oscillates wildly     | LR too high              | Reduce `--lr` by 2-5x                 |
| Loss decreases very slowly | LR too low               | Increase `--lr` by 2x                 |
| Loss plateaus immediately  | Model too small          | Increase `--dim` or `--num-layers`    |
| Loss goes to ~0            | Overfitting/memorization | Reduce model size, add early stopping |

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
   - Increase `--epochs` to 100-150
   - Increase `--dim` to 384 or 512

---

## Project Structure

```
mimir/
├── data/                   # Dataset (gitignored)
│   └── dataset.csv         # Generated training data
├── checkpoints/            # Model checkpoints (gitignored)
│   ├── best_model.pt       # Best model weights
│   ├── config.json         # Model configuration
│   ├── targets.json        # Target protein mapping
│   └── tokenizer.txt       # Vocabulary
├── mimir/                  # Python package
│   ├── __init__.py
│   ├── tokenizer.py        # Amino acid tokenization (vocabulary built from dataset)
│   ├── dataset.py          # PyTorch dataset loading
│   └── model.py            # MaskedDiffusion + Transformer architecture
└── scripts/                # CLI scripts
    ├── generate_dataset.py # Database → CSV dataset
    ├── verify_dataset.py   # Dataset constraint verification
    ├── train.py            # Training with presets
    └── generate.py         # Conditional generation
```

## References

- [D3PM Paper](https://arxiv.org/abs/2107.03006): Structured Denoising Diffusion Models in Discrete State-Spaces (Austin et al., 2021)
- [d3pm implementation](https://github.com/cloneofsimo/d3pm): Minimal PyTorch D3PM by @cloneofsimo
