"""
Diffusion models for peptide generation.

Contains:
- D3PM: Discrete Denoising Diffusion (for reference/comparison)
- MaskedDiffusion: MASK-based diffusion with linear masking schedule (preferred)
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


class D3PM(nn.Module):
    """
    Discrete Denoising Diffusion Probabilistic Model.

    Implements the forward diffusion process (adding noise) and reverse process
    (denoising) for discrete sequences like peptides.
    """

    def __init__(
        self,
        x0_model: nn.Module,
        n_timesteps: int = 1000,
        num_classes: int = 24,
        hybrid_loss_coeff: float = 0.001,
    ):
        """
        Args:
            x0_model: Model that predicts x0 from noisy input
            n_timesteps: Number of diffusion timesteps
            num_classes: Number of discrete classes (vocab size)
            hybrid_loss_coeff: Weight for VB loss term (0 = CE only)
        """
        super().__init__()
        self.x0_model = x0_model
        self.n_timesteps = n_timesteps
        self.num_classes = num_classes
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-6

        steps = torch.arange(n_timesteps + 1, dtype=torch.float32) / n_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1],
            torch.ones_like(alpha_bar[1:]) * 0.999,
        )

        q_onestep_mats = []
        for beta in beta_t:
            mat = torch.ones(num_classes, num_classes, dtype=torch.float32) * beta / num_classes
            mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
            q_onestep_mats.append(mat)

        q_onestep_mats = torch.stack(q_onestep_mats, dim=0)
        q_onestep_transposed = q_onestep_mats.transpose(1, 2)

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, n_timesteps):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)

        self.register_buffer("q_onestep_transposed", q_onestep_transposed)
        self.register_buffer("q_mats", q_mats)

    def _at(self, a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Index into transition matrices at timestep t for tokens x."""
        t = t.reshape((-1, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :]

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:
        """Forward diffusion: sample x_t from x_0. PAD tokens are not noised."""
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_t = torch.argmax(logits + gumbel_noise, dim=-1)
        
        # Keep PAD positions unchanged (don't noise them)
        pad_mask = x_0 == pad_idx
        x_t = torch.where(pad_mask, x_0, x_t)
        return x_t

    def q_posterior_logits(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute q(x_{t-1} | x_t, x_0) in log space."""
        if x_0.dtype in (torch.int32, torch.int64):
            x_0_logits = torch.log(F.one_hot(x_0, self.num_classes).float() + self.eps)
        else:
            x_0_logits = x_0

        fact1 = self._at(self.q_onestep_transposed, t, x_t)
        softmaxed = F.softmax(x_0_logits, dim=-1)

        t_idx = (t - 2).clamp(min=0)
        qmats2 = self.q_mats[t_idx]
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * x_t.dim()))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def vb_loss(
        self, 
        dist1: torch.Tensor, 
        dist2: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute variational bound loss (KL divergence) with optional padding mask.
        
        MASKED VB LOSS
        --------------
        The VB loss measures how well the model's predicted posterior matches
        the true posterior. However, PAD positions have no biological meaning,
        so we should NOT penalize the model for what happens in those slots.
        
        Without masking: Model wastes capacity learning to predict PAD→PAD transitions
        With masking: Gradients flow only from meaningful amino acid positions
        
        Args:
            dist1: True posterior logits [batch, seq_len, vocab_size]
            dist2: Predicted posterior logits [batch, seq_len, vocab_size]
            pad_mask: Boolean mask [batch, seq_len] where True = PAD position
                      If None, no masking is applied (backward compatible)
        
        Returns:
            Scalar KL divergence loss averaged over non-PAD positions
        """
        # Flatten spatial dimensions: [batch * seq_len, vocab_size]
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)
        
        # KL divergence: sum over vocab dimension
        out = F.softmax(dist1 + self.eps, dim=-1) * (
            F.log_softmax(dist1 + self.eps, dim=-1)
            - F.log_softmax(dist2 + self.eps, dim=-1)
        )
        kl_per_position = out.sum(dim=-1)  # [batch * seq_len]
        
        if pad_mask is not None:
            # Flatten mask to match: [batch * seq_len]
            # Invert: we want True for ACTIVE positions (not PAD)
            active_mask = (~pad_mask).flatten().float()
            
            # Weighted sum over active positions only
            # This ensures gradients only flow from real amino acids
            masked_kl = kl_per_position * active_mask
            return masked_kl.sum() / (active_mask.sum() + self.eps)
        
        return kl_per_position.mean()

    def forward(
        self, 
        x: torch.Tensor, 
        cond: dict | None = None,
        pad_idx: int = 0,
        pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Training forward pass with masked loss calculation.
        
        MASKED LOSS CALCULATION
        -----------------------
        This is critical for variable-length peptide training. Without masking:
        - CE loss penalizes wrong predictions at PAD positions (meaningless)
        - VB loss includes KL divergence at PAD positions (adds noise)
        - Model capacity is wasted learning PAD→PAD transitions
        
        With masking:
        - CE uses ignore_index to skip PAD positions entirely
        - VB multiplies by (1 - pad_mask) before averaging
        - Gradients flow only from actual amino acid positions
        
        This dramatically improves training efficiency and final model quality.

        Args:
            x: Clean token indices [batch, seq_len]
            cond: Optional conditioning dict with 'label' and 'target_id'
            pad_idx: Token index for PAD (for CE ignore_index)
            pad_mask: Boolean mask [batch, seq_len] where True = PAD position
                      Used for VB loss masking and attention masking

        Returns:
            loss: Combined CE + λ*VB loss value
            info: Dict with individual loss components for logging
        """
        # Sample random timesteps for each sequence in batch
        t = torch.randint(1, self.n_timesteps + 1, (x.shape[0],), device=x.device)

        # Forward diffusion: x_0 → x_t (add noise)
        # Note: q_sample already preserves PAD positions (noising invariance)
        noise = torch.rand((*x.shape, self.num_classes), device=x.device)
        x_t = self.q_sample(x, t, noise, pad_idx=pad_idx)

        # Predict x_0 from noisy x_t (with attention masking via pad_mask)
        predicted_x0_logits = self.x0_model(x_t, t, cond, pad_mask=pad_mask)

        # === VARIATIONAL BOUND LOSS ===
        # KL divergence between true and predicted posteriors
        # With masking: only compute KL for non-PAD positions
        true_q_posterior = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, x_t, t)
        vb_loss = self.vb_loss(true_q_posterior, pred_q_posterior, pad_mask=pad_mask)

        # === CROSS-ENTROPY LOSS ===
        # Direct x_0 prediction loss with ignore_index for PAD tokens
        # This is the key fix: PAD positions contribute ZERO to the loss
        ce_loss = F.cross_entropy(
            predicted_x0_logits.flatten(0, -2), 
            x.flatten(),
            ignore_index=pad_idx,  # Critical: don't penalize PAD predictions
        )

        # Combine losses with configurable weighting
        loss = self.hybrid_loss_coeff * vb_loss + ce_loss

        return loss, {"vb_loss": vb_loss.item(), "ce_loss": ce_loss.item()}

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: x_t → x_{t-1}.
        
        Args:
            x: Current noisy tokens [batch, seq_len]
            t: Current timestep [batch]
            cond: Optional conditioning dict
            pad_mask: Boolean mask for attention [batch, seq_len]
        
        Returns:
            Denoised tokens [batch, seq_len]
        """
        # Predict clean x_0 from noisy x_t
        predicted_x0_logits = self.x0_model(x, t, cond, pad_mask=pad_mask)
        
        # Compute posterior: q(x_{t-1} | x_t, predicted_x_0)
        pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, x, t)

        # Gumbel-softmax sampling for discrete tokens
        noise = torch.rand((*x.shape, self.num_classes), device=x.device)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        
        # At t=1, take argmax directly (no noise)
        # At t>1, add Gumbel noise for stochastic sampling
        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

        return torch.argmax(pred_q_posterior + gumbel_noise * not_first_step, dim=-1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        cond: dict | None = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate samples from noise (fixed length, no template).
        
        This is the basic generation method. For length-controlled generation,
        use sample_with_template() instead.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length (all samples will be this length)
            cond: Optional conditioning dict
            device: Device to generate on

        Returns:
            Generated token indices [batch_size, seq_len]
        """
        # Initialize with random tokens (pure noise)
        x = torch.randint(0, self.num_classes, (batch_size, seq_len), device=device)

        # Reverse diffusion: T → T-1 → ... → 1 → 0
        for t in range(self.n_timesteps, 0, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, cond)

        return x

    @torch.no_grad()
    def sample_with_template(
        self,
        batch_size: int,
        lengths: torch.Tensor,
        pad_idx: int,
        cond: dict | None = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate samples with user-specified length constraints (template-guided).
        
        TEMPLATE-GUIDED SAMPLING
        ------------------------
        Instead of the model "guessing" the peptide length, the user provides
        explicit length constraints. This is essential for practical applications
        where you need peptides of specific lengths.
        
        HOW IT WORKS
        ------------
        1. Create a "template" tensor where:
           - Positions 0..L-1 are initialized with random noise (to be denoised)
           - Positions L..max_len-1 are PAD tokens (preserved throughout)
        
        2. During reverse diffusion:
           - Only denoise the active positions (0..L-1)
           - PAD positions remain PAD at every timestep
           - Attention masking prevents PAD from influencing denoising
        
        3. Result: Each peptide has exactly the requested length
        
        Args:
            batch_size: Number of samples to generate
            lengths: Target length for each sample [batch_size]
                     Each value should be in range [4, max_length]
            pad_idx: Token index for PAD (typically 0)
            cond: Optional conditioning dict with 'label' and 'target_id'
            device: Device to generate on
        
        Returns:
            Generated token indices [batch_size, max_len] with PAD at end
        
        Example:
            >>> lengths = torch.randint(8, 13, (10,))  # 10 peptides, 8-12 AA each
            >>> samples = model.sample_with_template(10, lengths, pad_idx=0)
            >>> # Each sample[i] has exactly lengths[i] amino acids, rest are PAD
        """
        # Determine max length needed for this batch
        max_len = lengths.max().item()
        lengths = lengths.to(device)
        
        # Build template: noise for active slots, PAD for rest
        # This is the key insight: we initialize the "structure" of the output
        x = torch.randint(0, self.num_classes, (batch_size, max_len), device=device)
        
        # Create padding mask: True where position >= target length
        # Example: length=5, max_len=8 → mask = [F, F, F, F, F, T, T, T]
        positions = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
        pad_mask = positions >= lengths.unsqueeze(1)  # [batch, max_len]
        
        # Initialize PAD positions with pad_idx (not random noise)
        x = torch.where(pad_mask, torch.full_like(x, pad_idx), x)
        
        # Reverse diffusion with template preservation
        for t in range(self.n_timesteps, 0, -1):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Denoise (with attention masking for efficiency)
            x_new = self.p_sample(x, t_tensor, cond, pad_mask=pad_mask)
            
            # CRITICAL: Preserve PAD positions after each step
            # Without this, PAD tokens could be replaced with amino acids
            x = torch.where(pad_mask, torch.full_like(x, pad_idx), x_new)
        
        return x


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
    
    LOSS COMPUTATION
    ----------------
    Loss is computed ONLY on masked positions (BERT-style).
    This forces the model to learn context, not just copy visible tokens.
    """

    def __init__(
        self,
        x0_model: nn.Module,
        n_timesteps: int = 20,
        max_length: int = 20,
        mask_idx: int = 1,
        pad_idx: int = 0,
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
    ) -> tuple[torch.Tensor, dict]:
        """
        Training forward pass.
        
        TRAINING PROCEDURE
        ------------------
        1. Sample random timestep t for each sequence
        2. Mask t/T fraction of positions
        3. Predict original tokens at ALL positions
        4. Compute CE loss ONLY on MASKED positions
        
        WHY LOSS ON MASKED POSITIONS ONLY?
        ----------------------------------
        Computing loss on visible tokens would:
        - Artificially inflate the score (trivial to predict visible tokens)
        - Make the model lazy (just copy instead of learning context)
        
        This is the BERT-style approach: focus on the fill-in-the-blank task.
        Training takes more epochs but learns better representations.
        
        Args:
            x: Clean token indices [batch, seq_len]
            cond: Conditioning dict with 'label' and 'target_id'
            lengths: Actual sequence lengths [batch]
            pad_mask: Boolean mask [batch, seq_len] where True = PAD
        
        Returns:
            loss: Cross-entropy loss on masked positions only
            info: Dict with loss value for logging
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # If lengths not provided, compute from pad_mask
        if lengths is None:
            if pad_mask is not None:
                lengths = (~pad_mask).sum(dim=1)
            else:
                lengths = torch.full((batch_size,), seq_len, device=device)
        
        # Sample random timesteps
        t = torch.randint(1, self.n_timesteps + 1, (batch_size,), device=device)
        
        # Forward diffusion: mask t/T positions
        x_t, mask_positions = self.q_sample(x, t, lengths)
        
        # Predict original tokens from partially masked input
        logits = self.x0_model(x_t, t, cond, pad_mask=pad_mask)
        
        # Compute CE loss ONLY on masked positions
        # This forces the model to learn context, not just copy visible tokens
        logits_flat = logits.view(-1, logits.size(-1))  # [batch * seq_len, vocab]
        targets_flat = x.view(-1)  # [batch * seq_len]
        mask_flat = mask_positions.view(-1)  # [batch * seq_len]
        
        # Select only masked positions for loss
        masked_logits = logits_flat[mask_flat]
        masked_targets = targets_flat[mask_flat]
        
        if masked_logits.numel() > 0:
            loss = F.cross_entropy(masked_logits, masked_targets)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss, {"ce_loss": loss.item(), "num_masked": mask_flat.sum().item()}

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

