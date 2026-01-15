"""
Generate peptide sequences using trained D3PM model.

Supports conditioning on:
- Interaction label (interacting vs non-interacting)
- Target protein (specific human protein)
- Length constraints (min/max amino acids)

Usage:
    uv run python scripts/generate.py --num-samples 10
    uv run python scripts/generate.py --target Q96C01 --num-samples 10
    uv run python scripts/generate.py --min-length 8 --max-length 12 --num-samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mimir.model import D3PM, PeptideTransformer
from mimir.tokenizer import AminoAcidTokenizer

ROOT_DIR = Path(__file__).parent.parent
CHECKPOINTS_BASE = ROOT_DIR / "checkpoints"


def load_model(checkpoint_path: Path, checkpoints_dir: Path, device: str):
    """Load model from checkpoint."""
    with open(checkpoints_dir / "config.json") as f:
        config = json.load(f)

    tokenizer = AminoAcidTokenizer.load(checkpoints_dir / "tokenizer.txt")

    with open(checkpoints_dir / "targets.json") as f:
        target_to_id = json.load(f)

    x0_model = PeptideTransformer(
        vocab_size=config["vocab_size"],
        max_length=config["max_length"],
        dim=config["dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_targets=config["num_targets"],
        use_label_cond=True,
    )

    d3pm = D3PM(
        x0_model=x0_model,
        n_timesteps=config["n_timesteps"],
        num_classes=config["vocab_size"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    d3pm.load_state_dict(checkpoint["model_state_dict"])
    d3pm.eval()

    return d3pm, tokenizer, target_to_id, config


def generate(args):
    """
    Generate peptides with optional conditioning and length constraints.
    
    TEMPLATE-GUIDED SAMPLING
    ------------------------
    Previously, we used a retry-based approach: generate many peptides and
    filter to keep only those matching the desired length. This was wasteful.
    
    Now we use template-guided sampling:
    1. Pre-specify the exact length for each peptide
    2. Initialize a template with noise for active positions, PAD for rest
    3. Denoise only the active positions, preserving PAD throughout
    4. Result: 100% of generated peptides match the requested length
    
    This eliminates wasted compute and guarantees the output length.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine checkpoints directory
    if args.run_name:
        checkpoints_dir = CHECKPOINTS_BASE / args.run_name
    else:
        checkpoints_dir = CHECKPOINTS_BASE
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoints_dir / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        print("Train a model first: uv run python scripts/train.py -v", file=sys.stderr)
        sys.exit(1)

    d3pm, tokenizer, target_to_id, config = load_model(checkpoint_path, checkpoints_dir, device)

    # Build conditioning
    label = 1 if args.interacting else 0

    # Resolve target protein
    target_id_value = -1
    if args.target:
        if args.target in target_to_id:
            target_id_value = target_to_id[args.target]
            print(f"# Target: {args.target}", file=sys.stderr)
        else:
            print(f"Error: Target '{args.target}' not in training data", file=sys.stderr)
            print(f"Available targets (first 10): {list(target_to_id.keys())[:10]}", file=sys.stderr)
            sys.exit(1)

    # === TEMPLATE-GUIDED GENERATION ===
    # Sample random lengths for each peptide within the specified range
    # This replaces the old retry-based approach with deterministic length control
    lengths = torch.randint(
        args.min_length, 
        args.max_length + 1,  # +1 because randint is exclusive on the upper bound
        (args.num_samples,),
    )
    
    # Prepare conditioning tensors
    cond = {
        "label": torch.full((args.num_samples,), label, dtype=torch.long, device=device),
        "target_id": torch.full((args.num_samples,), target_id_value, dtype=torch.long, device=device),
    }
    
    # Generate with template-guided sampling
    # Each peptide will have exactly the length specified in 'lengths'
    print(f"# Generating {args.num_samples} peptides (length {args.min_length}-{args.max_length})...", file=sys.stderr)
    
    with torch.no_grad():
        samples = d3pm.sample_with_template(
            batch_size=args.num_samples,
            lengths=lengths,
            pad_idx=tokenizer.pad_idx,
            cond=cond,
            device=device,
        )

    # Decode and output
    for sample in samples:
        peptide = tokenizer.decode(sample.cpu().tolist())
        print(peptide)


def main():
    parser = argparse.ArgumentParser(
        description="Generate peptides with D3PM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-n", "--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--run-name", type=str, help="Run name (loads from checkpoints/{run_name}/)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (overrides run-name)")

    # Conditioning
    parser.add_argument("-t", "--target", type=str, help="Target protein accession")
    parser.add_argument("--interacting", action="store_true", default=True, help="Generate interacting")
    parser.add_argument("--non-interacting", action="store_false", dest="interacting")

    # Length constraints
    parser.add_argument("--min-length", type=int, default=4, help="Minimum peptide length")
    parser.add_argument("--max-length", type=int, default=20, help="Maximum peptide length")

    args = parser.parse_args()

    if args.min_length > args.max_length:
        parser.error("--min-length cannot be greater than --max-length")

    generate(args)


if __name__ == "__main__":
    main()
