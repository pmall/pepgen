# MÍMIR – Peptide Generation with Masked Diffusion

A target-conditioned peptide generator using masked diffusion. Given a human protein target, MÍMIR generates novel peptide sequences that may interact with it.

## Overview

### The Problem

We want to generate peptides (4-20 amino acids) that interact with specific human proteins. Our training data:

- **~3,500 interaction pairs**: Validated peptide-target pairs from human-human and virus-human interactions
- **~1,000,000 background sequences**: Random peptides from human/viral proteins with no known interactions

### The Solution: Masked Diffusion

MÍMIR uses **masked diffusion**, a generative model that learns to reconstruct sequences from partially masked inputs—similar to BERT, but designed for iterative generation.

**Training**: Mask random positions → predict original amino acids → learn sequence patterns  
**Generation**: Start fully masked → iteratively unmask highest-confidence predictions → output sequence

### Key Design Choices

#### Hierarchical Conditioning

The model receives two conditioning signals during training:

1. **Interaction flag** (0 or 1): Distinguishes background sequences from binders
2. **Target ID**: Identifies which human protein the peptide interacts with

This teaches the model the difference between "generic peptide grammar" and "patterns that bind to target X".

#### Linear Masking

The masking ratio scales linearly with timestep `t`:

```
mask_ratio = t / T
num_masks = ceil(t × L / T)
```

Where `L` = sequence length, `T` = total timesteps (20). This ensures all sequences—short or long—experience the full range of masking levels.

#### Curriculum Learning

Training starts easy and gets harder. The maximum timestep (`t_ceiling`) rises linearly over warmup epochs:

```
t_ceiling = 1 + (T - 1) × min(epoch / warmup_epochs, 1)
```

Early training: low `t` → few masks → local pattern learning  
Late training: high `t` → many masks → global reconstruction

#### Weighted Loss

After curriculum warmup, harder tasks (more masks) receive stronger gradients:

```
weight = mask_ratio^α    (after warmup)
weight = 1.0             (during warmup)
```

With α=1.5, fully masked sequences get ~3× more gradient than half-masked ones. This prioritizes the harder generative task. During curriculum warmup, weighting is disabled—difficulty progression is handled by `t_ceiling` alone.

#### Training Stabilization

- **LR warmup**: Ramps from 1e-7 to target LR over 2000 steps (prevents early instability)
- **Gradient clipping**: Caps gradient norm at 1.0
- **Weight decay**: AdamW with 0.01 decay regularizes embeddings

---

## Setup

```bash
uv sync
cp .env.example .env  # Edit with PostgreSQL credentials
```

## Usage

### 1. Generate Dataset

```bash
uv run python scripts/generate_dataset.py -v
```

### 2. Train Model

```bash
uv run python scripts/train.py --preset medium -v
uv run python scripts/train.py --preset medium --run-name exp1 -v  # Named run
```

### 3. Generate Peptides

```bash
uv run python scripts/generate.py -n 10                    # Any target
uv run python scripts/generate.py -t Q96C01 -n 10          # Specific target
uv run python scripts/generate.py --min-length 8 --max-length 12 -n 10
```

---

## Training Configuration

### Presets

| Preset   | Epochs | Dim | Layers | LR     | Curriculum | Use Case              |
|----------|--------|-----|--------|--------|------------|-----------------------|
| `small`  | 20     | 128 | 3      | 5e-4   | 5 epochs   | Quick experiments     |
| `medium` | 75     | 256 | 4      | 3e-4   | 10 epochs  | **Recommended**       |
| `large`  | 150    | 512 | 6      | 1e-4   | 15 epochs  | Best quality          |

All presets use batch size 256, 20 timesteps, and 2000 warmup steps.

```bash
uv run python scripts/train.py --preset small -v
uv run python scripts/train.py --preset medium -v
uv run python scripts/train.py --preset large -v
```

### Parameters

#### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 256 | Batch size |
| `--lr` | 1e-4 | Peak learning rate |
| `--warmup-steps` | 2000 | LR warmup from 1e-7 to target |

#### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dim` | 256 | Embedding dimension |
| `--num-layers` | 4 | Transformer layers |
| `--num-heads` | 4 | Attention heads (must divide dim) |
| `--n-timesteps` | 20 | Diffusion steps (must be ≥ max_length) |

#### Curriculum & Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--curriculum-warmup-epochs` | 10 | Epochs to ramp `t_ceiling` from 1 to T |
| `--loss-weight-alpha` | 1.5 | Exponent for mask-ratio weighting (0 = disabled) |

#### Sampling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--balanced` | false | Oversample interacting class (not recommended) |

---

## Run Management

Use `--run-name` to preserve multiple experiments:

```bash
uv run python scripts/train.py --preset medium --run-name baseline -v
uv run python scripts/train.py --preset large --run-name large_model -v

uv run python scripts/generate.py --run-name baseline -n 10
uv run python scripts/generate.py --run-name large_model -n 10
```

Creates separate checkpoint directories under `checkpoints/`.

---

## Project Structure

```
mimir/
├── mimir/
│   ├── model.py      # MaskedDiffusion + PeptideTransformer
│   ├── dataset.py    # Data loading with dynamic padding
│   └── tokenizer.py  # Amino acid tokenization
├── scripts/
│   ├── train.py      # Training with presets
│   ├── generate.py   # Conditional generation
│   ├── generate_dataset.py
│   └── verify_dataset.py
├── data/             # Dataset (gitignored)
└── checkpoints/      # Model weights (gitignored)
```

## References

- [MDLM](https://s-sahoo.com/mdlm/): Masked Diffusion Language Models (Sahoo et al.)
- [D3PM](https://arxiv.org/abs/2107.03006): Structured Denoising Diffusion Models (Austin et al., 2021)
