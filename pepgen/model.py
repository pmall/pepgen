"""
D3PM (Discrete Denoising Diffusion Probabilistic Model) for peptide generation.

Adapted from https://github.com/cloneofsimo/d3pm for 1D peptide sequences.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
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
    ) -> torch.Tensor:
        """
        Predict x0 logits from noisy input.

        Args:
            x: Noisy token indices [batch, seq_len]
            t: Timesteps [batch]
            cond: Optional conditioning dict with keys:
                - 'label': interaction labels [batch] (0 or 1)
                - 'target_id': target protein IDs [batch] (-1 for no target)

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
            h = layer(h)

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

        steps = torch.arange(n_timesteps + 1, dtype=torch.float64) / n_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1],
            torch.ones_like(alpha_bar[1:]) * 0.999,
        )

        q_onestep_mats = []
        for beta in beta_t:
            mat = torch.ones(num_classes, num_classes) * beta / num_classes
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
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: sample x_t from x_0."""
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

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
        qmats2 = self.q_mats[t_idx].to(dtype=softmaxed.dtype)
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * x_t.dim()))
        return torch.where(t_broadcast == 1, x_0_logits, out)

    def vb_loss(self, dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
        """Compute variational bound loss (KL divergence)."""
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)
        out = F.softmax(dist1 + self.eps, dim=-1) * (
            F.log_softmax(dist1 + self.eps, dim=-1)
            - F.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def forward(
        self, x: torch.Tensor, cond: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Training forward pass.

        Args:
            x: Clean token indices [batch, seq_len]
            cond: Optional conditioning dict

        Returns:
            loss: Combined loss value
            info: Dict with individual loss components
        """
        t = torch.randint(1, self.n_timesteps, (x.shape[0],), device=x.device)

        noise = torch.rand((*x.shape, self.num_classes), device=x.device)
        x_t = self.q_sample(x, t, noise)

        predicted_x0_logits = self.x0_model(x_t, t, cond)

        true_q_posterior = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, x_t, t)
        vb_loss = self.vb_loss(true_q_posterior, pred_q_posterior)

        ce_loss = F.cross_entropy(
            predicted_x0_logits.flatten(0, -2), x.flatten()
        )

        loss = self.hybrid_loss_coeff * vb_loss + ce_loss

        return loss, {"vb_loss": vb_loss.item(), "ce_loss": ce_loss.item()}

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict | None = None,
    ) -> torch.Tensor:
        """Single reverse diffusion step."""
        predicted_x0_logits = self.x0_model(x, t, cond)
        pred_q_posterior = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.rand((*x.shape, self.num_classes), device=x.device)
        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))

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
        Generate samples from noise.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            cond: Optional conditioning dict
            device: Device to generate on

        Returns:
            Generated token indices [batch_size, seq_len]
        """
        x = torch.randint(0, self.num_classes, (batch_size, seq_len), device=device)

        for t in reversed(range(1, self.n_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, cond)

        return x
