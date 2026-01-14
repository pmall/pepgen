"""
Training script for peptide D3PM model.

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

from mimir.dataset import BalancedPeptideDataset, PeptideDataset
from mimir.model import D3PM, PeptideTransformer
from mimir.tokenizer import AminoAcidTokenizer

ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "data" / "dataset.csv"
CHECKPOINTS_BASE = ROOT_DIR / "checkpoints"
MAX_LENGTH = 20

# Training presets optimized for different scenarios
# Note: n_timesteps scaled for average peptide length of 8-12 AA.
# Heuristic: timesteps ≈ avg_length × 2-3. Too many steps = noisy gradients.
PRESETS = {
    "small": {
        "description": "Fast experimentation, low memory (<4GB VRAM), ~20 min",
        "epochs": 20,
        "batch_size": 32,
        "lr": 2e-4,
        "dim": 128,
        "num_layers": 3,
        "num_heads": 4,
        "n_timesteps": 10,
        "hybrid_loss_coeff": 0.0,
        "warmup_steps": 500,
    },
    "medium": {
        "description": "Balanced quality/speed, recommended, ~1-2 hours",
        "epochs": 75,
        "batch_size": 64,
        "lr": 1e-4,
        "dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "n_timesteps": 25,
        "hybrid_loss_coeff": 0.001,
        "warmup_steps": 1000,
    },
    "large": {
        "description": "Best quality, requires >8GB VRAM, ~4-8 hours",
        "epochs": 150,
        "batch_size": 128,
        "lr": 5e-5,
        "dim": 512,
        "num_layers": 6,
        "num_heads": 8,
        "n_timesteps": 35,
        "hybrid_loss_coeff": 0.01,
        "warmup_steps": 2000,
    },
}


def log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, file=sys.stderr)


def apply_preset(args):
    """Apply preset configuration if specified."""
    if args.preset:
        if args.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {args.preset}. Available: {list(PRESETS.keys())}")
        
        preset = PRESETS[args.preset]
        for key, value in preset.items():
            if key == "description":
                continue
            # Only apply preset value if user didn't override
            arg_key = key.replace("-", "_")
            if getattr(args, arg_key, None) == getattr(get_default_args(), arg_key, None):
                setattr(args, arg_key, value)
    
    return args


def get_default_args():
    """Get default argument values for comparison."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args([])


def train(args):
    # Apply preset if specified
    args = apply_preset(args)
    
    # Set up checkpoint directory
    if args.run_name:
        checkpoints_dir = CHECKPOINTS_BASE / args.run_name
    else:
        checkpoints_dir = CHECKPOINTS_BASE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}", args.verbose)
    log(f"Checkpoints: {checkpoints_dir}", args.verbose)
    
    if args.preset:
        log(f"Using preset: {args.preset} - {PRESETS[args.preset]['description']}", args.verbose)

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
    if args.balanced:
        dataset = BalancedPeptideDataset(full_dataset, oversample_ratio=1.0)
        log(f"  Balanced sampling: {len(dataset):,} samples per epoch", args.verbose)
        log("  Warning: high risk of memorizing interacting sequences", args.verbose)
    else:
        dataset = full_dataset
        log(f"  Unbalanced sampling: {len(dataset):,} samples per epoch", args.verbose)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    # Model
    log("Building model...", args.verbose)
    x0_model = PeptideTransformer(
        vocab_size=tokenizer.vocab_size,
        max_length=MAX_LENGTH,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_targets=full_dataset.num_targets,
        use_label_cond=True,
    )

    d3pm = D3PM(
        x0_model=x0_model,
        n_timesteps=args.n_timesteps,
        num_classes=tokenizer.vocab_size,
        hybrid_loss_coeff=args.hybrid_loss_coeff,
    ).to(device)

    num_params = sum(p.numel() for p in d3pm.parameters())
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
        "hybrid_loss_coeff": args.hybrid_loss_coeff,
    }
    with open(checkpoints_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save target mapping
    with open(checkpoints_dir / "targets.json", "w") as f:
        json.dump(full_dataset.target_to_id, f)

    # Optimizer with warmup + cosine decay
    optimizer = torch.optim.AdamW(d3pm.parameters(), lr=args.lr)
    
    total_steps = args.epochs * len(dataloader)
    warmup_steps = getattr(args, 'warmup_steps', 1000)
    
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
        d3pm.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_vb = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=not args.verbose)

        for batch in pbar:
            tokens = batch["tokens"].to(device)
            cond = {
                "label": batch["label"].to(device),
                "target_id": batch["target_id"].to(device),
            }

            loss, info = d3pm(tokens, cond)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(d3pm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_ce += info["ce_loss"]
            epoch_vb += info["vb_loss"]
            num_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ce=f"{info['ce_loss']:.4f}",
            )

        avg_loss = epoch_loss / num_batches
        avg_ce = epoch_ce / num_batches
        avg_vb = epoch_vb / num_batches

        log(f"Epoch {epoch + 1}: loss={avg_loss:.4f} ce={avg_ce:.4f} vb={avg_vb:.4f}", args.verbose)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": d3pm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, checkpoints_dir / "best_model.pt")
            log(f"  Saved best model (loss={best_loss:.4f})", args.verbose)

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": d3pm.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, checkpoints_dir / f"checkpoint_{epoch + 1}.pt")

        # Sample generation
        if (epoch + 1) % args.sample_every == 0:
            d3pm.eval()
            log("\nSample generations:", args.verbose)

            with torch.no_grad():
                # Generate interacting peptides
                cond = {
                    "label": torch.ones(4, dtype=torch.long, device=device),
                    "target_id": torch.full((4,), -1, dtype=torch.long, device=device),
                }
                samples = d3pm.sample(4, MAX_LENGTH, cond=cond, device=device)

                log("  Interacting (no target):", args.verbose)
                for s in samples:
                    peptide = tokenizer.decode(s.cpu().tolist())
                    log(f"    {peptide}", args.verbose)

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
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Model
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-timesteps", type=int, default=25, help="Diffusion steps (10-35 for peptides)")
    parser.add_argument("--hybrid-loss-coeff", type=float, default=0.001, help="VB loss weight")

    # Sampling
    parser.add_argument("--balanced", action="store_true", default=False, help="Oversample interacting class (risk of memorization)")
    parser.add_argument("--no-balanced", action="store_false", dest="balanced")

    # Checkpoints
    parser.add_argument("--save-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every N epochs")

    # Output
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")


def main():
    parser = argparse.ArgumentParser(
        description="Train peptide D3PM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
