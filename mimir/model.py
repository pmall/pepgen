"""
Diffusion models for peptide generation.

Contains:
- MaskedDiffusion: MASK-based diffusion with linear masking schedule
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feedforward layers."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PeptideTransformer(nn.Module):
    """
    Transformer model for predicting x0 from noisy peptide sequences.

    Supports conditioning on:
        - Timestep (required)
        - Interaction label (optional): 0=non-interacting, 1=interacting
        - Target protein ID (optional): which human protein is the target
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 20,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_targets: int = 0,
        use_label_cond: bool = True,
    ):
        """
        Args:
            vocab_size: Number of tokens (amino acids + padding)
            max_length: Maximum sequence length
            dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_targets: Number of target proteins (0 to disable conditioning)
            use_label_cond: Whether to condition on interaction label
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim = dim
        self.use_label_cond = use_label_cond
        self.num_targets = num_targets

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_length, dim)

        self.time_embedding = SinusoidalPositionEmbeddings(dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        if use_label_cond:
            self.label_embedding = nn.Embedding(2, dim)

        if num_targets > 0:
            self.target_embedding = nn.Embedding(num_targets + 1, dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict x0 logits from noisy input.

        Args:
            x: Noisy token indices [batch, seq_len]
            t: Timesteps [batch]
            cond: Optional conditioning dict with keys:
                - 'label': interaction labels [batch] (0 or 1)
                - 'target_id': target protein IDs [batch] (-1 for no target)
            pad_mask: Boolean mask [batch, seq_len] where True = PAD position

        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        h = self.token_embedding(x) + self.position_embedding(positions)

        t_emb = self.time_mlp(self.time_embedding(t))

        cond_emb = t_emb
        if cond is not None:
            if self.use_label_cond and "label" in cond:
                cond_emb = cond_emb + self.label_embedding(cond["label"].long())

            if self.num_targets > 0 and "target_id" in cond:
                target_id = cond["target_id"].long()
                target_id = torch.where(
                    target_id >= 0, target_id, torch.full_like(target_id, self.num_targets)
                )
                cond_emb = cond_emb + self.target_embedding(target_id)

        h = h + cond_emb.unsqueeze(1)

        for layer in self.layers:
            h = layer(h, key_padding_mask=pad_mask)

        h = self.norm(h)
        logits = self.output(h)

        return logits





class MaskedDiffusion(nn.Module):
    """
    Masked Diffusion Model for peptide generation.
    
    MASKED DIFFUSION vs D3PM
    ------------------------
    Instead of random token swapping (D3PM), this model uses MASK-based corruption:
    
    D3PM:   MKTAY → XYZAB → QWERT (random tokens at each step)
    MASKED: MKTAY → MK___ → _____ (progressive masking)
    
    Benefits:
    1. Simpler learning task: "fill in the blanks" vs "reverse random swaps"
    2. BERT-style training: learn context, not just copy visible tokens
    3. No complex transition matrices: just linear masking schedule
    
    LINEAR MASKING SCHEDULE
    -----------------------
    The number of masks scales linearly with timestep AND sequence length:
    
        num_masks = ceil(t × L / T)
    
    Where:
        t = current timestep (1 to T)
        L = sequence length
        T = n_timesteps (should equal max_length, default 20)
    
    This ensures:
        - t=T: fully masked (L masks)
        - t=0: fully unmasked (target sequence)
        - Every sequence sees all masking levels from 1 to L
    
    Example for L=4, T=20:
        t ∈ [1-5]:   1 mask (25%)
        t ∈ [6-10]:  2 masks (50%)
        t ∈ [11-15]: 3 masks (75%)
        t ∈ [16-20]: 4 masks (100%)
    
    CURRICULUM MASKING
    ------------------
    Training starts with easy tasks (low t = few masks) and gradually increases
    difficulty. The ceiling on t rises linearly over warmup epochs:
    
        t_ceiling = 1 + (T - 1) × min(epoch / warmup_epochs, 1)
    
    This helps the model learn local patterns before tackling full reconstruction.
    
    WEIGHTED LOSS
    -------------
    High-noise states (more masks) receive higher gradient weight:
    
        weighted_loss = CE × (mask_ratio)^α
    
    With α=1.5, fully masked sequences get ~1.5x more gradient than half-masked.
    This prioritizes the harder generative task over easy local predictions.
    """

    def __init__(
        self,
        x0_model: nn.Module,
        n_timesteps: int = 20,
        max_length: int = 20,
        mask_idx: int = 1,
        pad_idx: int = 0,
        curriculum_warmup_epochs: int = 10,
        loss_weight_alpha: float = 1.5,
    ):
        """
        Initialize MaskedDiffusion model.
        
        Args:
            x0_model: Transformer model that predicts x0 from masked input
            n_timesteps: Number of diffusion timesteps. Must be >= max_length.
                        Default 20 (matches typical max peptide length).
            max_length: Maximum sequence length. Used for validation.
            mask_idx: Token index for [MASK] (default 1)
            pad_idx: Token index for [PAD] (default 0)
            curriculum_warmup_epochs: Epochs to ramp t_ceiling from 1 to T
            loss_weight_alpha: Exponent for mask-ratio weighting (0 = no weighting)
        
        Raises:
            ValueError: If n_timesteps < max_length
        """
        super().__init__()
        
        if n_timesteps < max_length:
            raise ValueError(
                f"n_timesteps ({n_timesteps}) must be >= max_length ({max_length}). "
                f"This ensures all mask counts from 1 to {max_length} are trained."
            )
        
        self.x0_model = x0_model
        self.n_timesteps = n_timesteps
        self.max_length = max_length
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.loss_weight_alpha = loss_weight_alpha

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: mask positions using LINEAR masking schedule.
        
        LINEAR MASKING FORMULA
        ----------------------
        num_masks = ceil(t × L / T)
        
        Where:
            t = current timestep (1 to T)
            L = sequence length
            T = max timesteps (n_timesteps)
        
        This ensures:
            - t=T: L masks (fully masked)
            - t=0: 0 masks (fully unmasked = target)
            - Linear interpolation between
        
        For a 4-AA sequence with T=20:
            t ∈ [1-5]:   1 mask (25%)
            t ∈ [6-10]:  2 masks (50%)
            t ∈ [11-15]: 3 masks (75%)
            t ∈ [16-20]: 4 masks (100%)
        
        For a 20-AA sequence with T=20:
            t=1: 1 mask, t=2: 2 masks, ..., t=20: 20 masks (1:1 mapping)
        
        Args:
            x_0: Clean token indices [batch, seq_len]
            t: Timesteps [batch], values in [1, T]
            lengths: Actual sequence lengths [batch]
        
        Returns:
            x_t: Masked tokens [batch, seq_len]
            mask_positions: Boolean mask [batch, seq_len] where True = masked
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # LINEAR MASKING: num_masks = ceil(t × L / T)
        # This ensures proper scaling for all sequence lengths
        num_to_mask = torch.ceil(
            t.float() * lengths.float() / self.n_timesteps
        ).long()
        
        # Clamp to valid range: at least 1, at most L
        # Use torch.minimum for tensor max, clamp for scalar min
        num_to_mask = torch.clamp(num_to_mask, min=1)
        num_to_mask = torch.minimum(num_to_mask, lengths)
        
        # For each sample, randomly select positions to mask
        # We use a trick: generate random scores, sort, take top-k
        random_scores = torch.rand(batch_size, seq_len, device=device)
        
        # Set PAD positions to -inf so they're never selected for masking
        pad_mask = torch.arange(seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        random_scores = random_scores.masked_fill(pad_mask, -float('inf'))
        
        # Get rank of each position (highest scores first)
        _, indices = random_scores.sort(dim=1, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(1, indices, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1))
        
        # Mask positions where rank < num_to_mask
        mask_positions = ranks < num_to_mask.unsqueeze(1)
        
        # Apply masking
        x_t = torch.where(mask_positions, self.mask_idx, x_0)
        
        return x_t, mask_positions

    def forward(
        self,
        x: torch.Tensor,
        cond: dict | None = None,
        lengths: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        epoch: int = 0,
        max_epochs: int = 1,
    ) -> tuple[torch.Tensor, dict]:
        """
        Training forward pass with curriculum masking and weighted loss.
        
        TRAINING PROCEDURE
        ------------------
        1. Compute t_ceiling based on curriculum (epoch / warmup_epochs)
        2. Sample random timestep t ∈ [1, t_ceiling]
        3. Mask t/T fraction of positions
        4. Predict original tokens at ALL positions
        5. Compute CE loss ONLY on MASKED positions
        6. Weight loss by (mask_ratio)^α to prioritize generative states
        
        Args:
            x: Clean token indices [batch, seq_len]
            cond: Conditioning dict with 'label' and 'target_id'
            lengths: Actual sequence lengths [batch]
            pad_mask: Boolean mask [batch, seq_len] where True = PAD
            epoch: Current training epoch (0-indexed)
            max_epochs: Total number of training epochs
        
        Returns:
            loss: Weighted cross-entropy loss on masked positions
            info: Dict with loss values for logging
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # If lengths not provided, compute from pad_mask
        if lengths is None:
            if pad_mask is not None:
                lengths = (~pad_mask).sum(dim=1)
            else:
                lengths = torch.full((batch_size,), seq_len, device=device)
        
        # Curriculum: t_ceiling rises linearly from 1 to T over warmup epochs
        curriculum_progress = min(epoch / max(self.curriculum_warmup_epochs, 1), 1.0)
        t_ceiling = int(1 + (self.n_timesteps - 1) * curriculum_progress)
        t_ceiling = max(1, min(t_ceiling, self.n_timesteps))
        
        # Sample random timesteps up to curriculum ceiling
        t = torch.randint(1, t_ceiling + 1, (batch_size,), device=device)
        
        # Forward diffusion: mask t/T positions
        x_t, mask_positions = self.q_sample(x, t, lengths)
        
        # Predict original tokens from partially masked input
        logits = self.x0_model(x_t, t, cond, pad_mask=pad_mask)
        
        # Compute CE loss ONLY on masked positions
        logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, vocab]
        targets_flat = x.view(-1)  # [batch * seq_len]
        mask_flat = mask_positions.view(-1)  # [batch * seq_len]
        
        # Select only masked positions for loss
        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]
        
        if masked_logits.numel() > 0:
            ce_loss = F.cross_entropy(masked_logits, masked_targets)
            
            # Compute mask ratio for logging
            num_masked = mask_positions.sum(dim=1).float()  # [batch]
            mask_ratios = num_masked / lengths.float()  # [batch]
            avg_mask_ratio = mask_ratios.mean()
            
            # Weighted penalty: only apply after curriculum warmup
            # During warmup, curriculum controls difficulty; after, weighting prioritizes hard tasks
            if epoch >= self.curriculum_warmup_epochs and self.loss_weight_alpha > 0:
                weight = avg_mask_ratio ** self.loss_weight_alpha
                loss = ce_loss * weight
            else:
                weight = 1.0
                loss = ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=device)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            avg_mask_ratio = torch.tensor(0.0, device=device)
            weight = 1.0
        
        return loss, {
            "ce_loss": ce_loss.item(),
            "weighted_loss": loss.item(),
            "mask_ratio": avg_mask_ratio.item() if isinstance(avg_mask_ratio, torch.Tensor) else avg_mask_ratio,
            "t_ceiling": t_ceiling,
            "num_masked": mask_flat.sum().item(),
        }

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        lengths: torch.Tensor,
        cond: dict | None = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate samples by iteratively unmasking.
        
        GENERATION PROCEDURE
        --------------------
        1. Start fully masked: x = [MASK, MASK, ..., MASK, PAD, PAD, ...]
        2. For t = T, T-1, ..., 1:
           a. Predict all positions: logits = model(x, t)
           b. Compute confidence for each MASK position
           c. Unmask ~1/t fraction of remaining masks (highest confidence first)
        3. Result: fully unmasked sequence
        
        Args:
            batch_size: Number of samples to generate
            lengths: Target length for each sample [batch_size]
            cond: Conditioning dict with 'label' and 'target_id'
            device: Device to generate on
        
        Returns:
            Generated token indices [batch_size, max_len]
        """
        lengths = lengths.to(device)
        max_len = lengths.max().item()
        
        # Build pad_mask: True where position >= length
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        pad_mask = positions >= lengths.unsqueeze(1)
        
        # Initialize: all active positions are MASK, PAD positions are PAD
        x = torch.full((batch_size, max_len), self.mask_idx, device=device, dtype=torch.long)
        x = torch.where(pad_mask, self.pad_idx, x)
        
        # Track which positions are still masked
        is_masked = ~pad_mask  # Initially all non-PAD positions are "masked"
        
        for t in range(self.n_timesteps, 0, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict all positions
            logits = self.x0_model(x, t_tensor, cond, pad_mask=pad_mask)
            
            # Get predicted tokens and confidence (max probability)
            probs = F.softmax(logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
            
            # Only consider currently masked positions for unmasking
            confidence = confidence.masked_fill(~is_masked, -float('inf'))
            
            # Determine how many to unmask this step
            # We want to unmask all by the end, so unmask ~remaining/t each step
            num_still_masked = is_masked.sum(dim=1).float()
            num_to_unmask = torch.ceil(num_still_masked / t).long()
            
            # Unmask highest-confidence positions
            for i in range(batch_size):
                if num_to_unmask[i] > 0:
                    # Get indices of masked positions sorted by confidence
                    mask_indices = is_masked[i].nonzero(as_tuple=True)[0]
                    if len(mask_indices) > 0:
                        conf_at_masked = confidence[i, mask_indices]
                        _, sorted_idx = conf_at_masked.sort(descending=True)
                        to_unmask = mask_indices[sorted_idx[:num_to_unmask[i]]]
                        
                        # Unmask these positions
                        x[i, to_unmask] = pred_tokens[i, to_unmask]
                        is_masked[i, to_unmask] = False
        
        return x

