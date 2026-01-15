"""
Training script for peptide MaskedDiffusion model.

The model is trained on BOTH interacting and background sequences:
- Background (~1M): Random peptides from human/viral proteins, teaches general peptide structure
- Interacting (~3.5k): Curated peptide-target pairs from hh and vh interactions

The model learns to distinguish via two conditions:
1. Interaction label: interacting (1) vs background (0)
2. Target protein: which human protein the peptide interacts with

Usage:
    uv run python scripts/train.py -v
    uv run python scripts/train.py --preset large -v
    uv run python scripts/train.py --epochs 200 --batch-size 128 -v
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from mimir.dataset import BalancedPeptideDataset, PeptideDataset, create_dynamic_collate_fn
from mimir.model import MaskedDiffusion, PeptideTransformer
from mimir.tokenizer import AminoAcidTokenizer

ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "data" / "dataset.csv"
CHECKPOINTS_BASE = ROOT_DIR / "checkpoints"
MAX_LENGTH = 20

# Training presets optimized for different scenarios
# Note: n_timesteps scaled for average peptide length of 8-12 AA.
# Heuristic: timesteps ≈ avg_length × 2-3. Too many steps = noisy gradients.
# Batch size 256 is optimal for T4 GPU (diminishing returns beyond).
PRESETS = {
    "small": {
        "description": "Fast experimentation, validation",
        "epochs": 20,
        "batch_size": 256,
        "lr": 5e-4,
        "dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "n_timesteps": 20,
        "warmup_steps": 500,
    },
    "medium": {
        "description": "Balanced quality/speed, recommended",
        "epochs": 75,
        "batch_size": 256,
        "lr": 3e-4,
        "dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "n_timesteps": 20,
        "warmup_steps": 1000,
    },
    "large": {
        "description": "Best quality, production model",
        "epochs": 150,
        "batch_size": 256,
        "lr": 1e-4,
        "dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "n_timesteps": 20,
        "warmup_steps": 2000,
    },
}


def log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, file=sys.stderr)


def apply_preset(args, explicit_args):
    """Apply preset configuration if specified."""
    if args.preset:
        if args.preset not in PRESETS:
            raise ValueError(
                f"Unknown preset: {args.preset}. Available: {list(PRESETS.keys())}"
            )

        preset = PRESETS[args.preset]
        for key, value in preset.items():
            if key == "description":
                continue
            # Only apply preset value if user didn't explicitly set it
            arg_key = key.replace("-", "_")
            if arg_key not in explicit_args:
                setattr(args, arg_key, value)

    return args


def train(args, explicit_args=None):
    # Apply preset if specified
    if explicit_args is None:
        explicit_args = set()
    args = apply_preset(args, explicit_args)

    # Set up checkpoint directory
    if args.run_name:
        checkpoints_dir = CHECKPOINTS_BASE / args.run_name
    else:
        checkpoints_dir = CHECKPOINTS_BASE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}", args.verbose)
    log(f"Checkpoints: {checkpoints_dir}", args.verbose)

    if args.preset:
        log(
            f"Using preset: {args.preset} - {PRESETS[args.preset]['description']}",
            args.verbose,
        )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    log("Loading tokenizer...", args.verbose)
    tokenizer = AminoAcidTokenizer.from_dataset(DATASET_PATH)
    tokenizer.save(checkpoints_dir / "tokenizer.txt")
    log(f"  Vocabulary: {tokenizer.vocab_size} tokens", args.verbose)

    # Dataset - always load ALL data (interacting + non-interacting)
    log("Loading dataset...", args.verbose)
    full_dataset = PeptideDataset(
        DATASET_PATH,
        tokenizer,
        max_length=MAX_LENGTH,
        interacting_only=False,
    )

    num_interacting = sum(1 for s in full_dataset.samples if s["label"] == 1)
    num_background = len(full_dataset) - num_interacting
    log(f"  Interacting: {num_interacting:,}", args.verbose)
    log(f"  Background: {num_background:,}", args.verbose)
    log(f"  Target proteins: {full_dataset.num_targets}", args.verbose)

    # Sampling strategy
    if args.overfit_test > 0:
        # Subset for overfit testing
        from torch.utils.data import Subset
        indices = list(range(min(args.overfit_test, len(full_dataset))))
        dataset = Subset(full_dataset, indices)
        log(f"  OVERFIT TEST: {len(dataset)} samples only", args.verbose)
    elif args.balanced:
        dataset = BalancedPeptideDataset(full_dataset, oversample_ratio=1.0)
        log(f"  Balanced sampling: {len(dataset):,} samples per epoch", args.verbose)
        log("  Warning: high risk of memorizing interacting sequences", args.verbose)
    else:
        dataset = full_dataset
        log(f"  Unbalanced sampling: {len(dataset):,} samples per epoch", args.verbose)

    # DataLoader with dynamic padding
    # This is a key optimization: instead of padding all sequences to max_length=20,
    # we pad each batch only to the longest sequence in that batch.
    # This reduces memory usage and speeds up attention computation.
    collate_fn = create_dynamic_collate_fn(pad_idx=tokenizer.pad_idx)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        collate_fn=collate_fn,  # Dynamic padding to batch-max length
    )

    # Model - using MaskedDiffusion instead of D3PM for faster convergence
    log("Building model (MaskedDiffusion)...", args.verbose)
    x0_model = PeptideTransformer(
        vocab_size=tokenizer.vocab_size,
        max_length=MAX_LENGTH,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_targets=full_dataset.num_targets,
        use_label_cond=True,
    )

    # MaskedDiffusion uses MASK-based corruption instead of D3PM's random swaps:
    # - Linear masking: num_masks = ceil(t × L / T)
    # - BERT-style loss: computed only on masked positions
    # - n_timesteps must be >= max_length (20) for proper training
    model = MaskedDiffusion(
        x0_model=x0_model,
        n_timesteps=args.n_timesteps,
        max_length=MAX_LENGTH,  # Validates n_timesteps >= max_length
        mask_idx=tokenizer.mask_idx,
        pad_idx=tokenizer.pad_idx,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {num_params:,}", args.verbose)

    # Save config
    config = {
        "vocab_size": tokenizer.vocab_size,
        "max_length": MAX_LENGTH,
        "dim": args.dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_targets": full_dataset.num_targets,
        "n_timesteps": args.n_timesteps,
        "model_type": "masked_diffusion",  # Track which model type
    }
    with open(checkpoints_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save target mapping
    with open(checkpoints_dir / "targets.json", "w") as f:
        json.dump(full_dataset.target_to_id, f)

    # Optimizer with warmup + cosine decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(dataloader)
    warmup_steps = getattr(args, "warmup_steps", 1000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    log(f"\nTraining for {args.epochs} epochs...", args.verbose)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not args.verbose,
        )

        for batch in pbar:
            tokens = batch["tokens"].to(device)
            lengths = batch["length"].to(device)
            
            # Conditioning signals for the diffusion model
            cond = {
                "label": batch["label"].to(device),
                "target_id": batch["target_id"].to(device),
            }
            
            # Padding mask for attention masking
            pad_mask = batch["pad_mask"].to(device)

            # Forward pass: mask t/T positions, predict masked, compute CE loss
            loss, info = model(tokens, cond, lengths=lengths, pad_mask=pad_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_ce += info["ce_loss"]
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{info['ce_loss']:.4f}",
            )

        avg_loss = epoch_loss / num_batches
        avg_ce = epoch_ce / num_batches

        log(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f} ce={avg_ce:.4f}",
            args.verbose,
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                checkpoints_dir / "best_model.pt",
            )
            log(f"  Saved best model (loss={best_loss:.4f})", args.verbose)

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoints_dir / f"checkpoint_{epoch + 1}.pt",
            )

        # Sample generation - showcase masked diffusion with variable lengths
        if (epoch + 1) % args.sample_every == 0:
            model.eval()
            log("\nSample generations (variable length 4-20):", args.verbose)

            with torch.no_grad():
                # Generate interacting peptides with random lengths between 4-20
                num_samples = 4
                lengths = torch.randint(4, 21, (num_samples,))
                
                cond = {
                    "label": torch.ones(num_samples, dtype=torch.long, device=device),
                    "target_id": torch.full((num_samples,), -1, dtype=torch.long, device=device),
                }
                samples = model.sample(
                    batch_size=num_samples,
                    lengths=lengths,
                    cond=cond,
                    device=device,
                )

                log("  Interacting (no target, variable lengths):", args.verbose)
                for i, s in enumerate(samples):
                    peptide = tokenizer.decode(s.cpu().tolist())
                    log(f"    [{lengths[i].item():2d} AA] {peptide}", args.verbose)

            log("", args.verbose)

    log("Training complete!", args.verbose)


def add_arguments(parser):
    """Add all training arguments to parser."""
    # Run management
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this run (saves to checkpoints/{run_name}/)",
    )

    # Preset
    parser.add_argument(
        "--preset",
        type=str,
        choices=["small", "medium", "large"],
        help="Use predefined configuration (small/medium/large)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="LR warmup steps"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Model
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=20,
        help="Diffusion steps (must be >= max_length=20)",
    )

    # Sampling
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help="Oversample interacting class (risk of memorization)",
    )
    parser.add_argument("--no-balanced", action="store_false", dest="balanced")
    parser.add_argument(
        "--overfit-test",
        type=int,
        default=0,
        help="Train on N samples only (to test if model can overfit)",
    )

    # Checkpoints
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save every N epochs"
    )
    parser.add_argument(
        "--sample-every", type=int, default=10, help="Sample every N epochs"
    )

    # Output
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def get_explicit_args(parser, argv=None):
    """Get set of argument names that were explicitly passed on command line."""
    import sys
    if argv is None:
        argv = sys.argv[1:]
    
    explicit = set()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('--'):
            # Extract argument name (handle --arg=value and --arg value)
            if '=' in arg:
                name = arg[2:].split('=')[0].replace('-', '_')
            else:
                name = arg[2:].replace('-', '_')
            explicit.add(name)
        elif arg.startswith('-') and len(arg) == 2:
            # Short argument like -v
            for action in parser._actions:
                if action.option_strings and arg in action.option_strings:
                    explicit.add(action.dest)
                    break
        i += 1
    return explicit


def main():
    parser = argparse.ArgumentParser(
        description="Train peptide MaskedDiffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    explicit_args = get_explicit_args(parser)
    args = parser.parse_args()
    train(args, explicit_args)


if __name__ == "__main__":
    main()
