"""Stage 2 reranker: ParT-style pairwise-bias self-attention encoder.

Reuses ParticleTransformer components (PairEmbed, Block, Embed) from the
existing codebase. The key difference from standard ParT:
    - Per-track scoring head instead of CLS token classification
    - Stage 1 scores concatenated as an extra input feature
    - Trained with ranking loss (same as TrackPreFilter)

Architecture:
    1. Input embedding: cat(features, stage1_score) → MLP → (P, B, embed_dim)
    2. Pairwise features: lorentz_vectors → PairEmbed → (B*H, P, P) attention bias
       Pairwise physics features: ln kT, ln z, ln ΔR, ln m²
       These encode track-track relationships (e.g. ρ→ππ invariant mass).
    3. N transformer blocks with pairwise bias in attention:
       Attn(Q,K,V) = softmax(QK^T / √d_k + pairwise_bias) × V
    4. Per-track scoring MLP: encoded embedding → scalar score

Reference: Qu et al., ICML 2022 (arXiv:2202.03772) — Particle Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.ParticleTransformer import (
    Block,
    Embed,
    PairEmbed,
)


class CascadeReranker(nn.Module):
    """ParT-style pairwise-bias encoder for Stage 2 track reranking.

    Args:
        input_dim: Number of per-track features (default: 16).
        embed_dim: Transformer embedding dimension (default: 128).
        num_heads: Number of attention heads (default: 4).
        num_layers: Number of transformer blocks (default: 3).
        pair_input_dim: Number of pairwise Lorentz vector features (default: 4).
            4 = ln kT, ln z, ln ΔR, ln m².
        pair_embed_dims: MLP dims for pairwise feature embedding.
        ffn_ratio: Feed-forward expansion ratio in transformer blocks.
        dropout: Dropout rate in transformer blocks.
        ranking_num_samples: Negatives sampled per positive in ranking loss.
        ranking_temperature: Temperature for ranking loss.
    """

    def __init__(
        self,
        input_dim: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        pair_input_dim: int = 4,
        pair_embed_dims: list[int] | None = None,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
    ):
        super().__init__()
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature

        if pair_embed_dims is None:
            pair_embed_dims = [64, 64]

        # Input embedding: cat(features, stage1_score) → embed_dim
        # +1 for stage1_score channel
        self.embed = Embed(
            input_dim + 1,
            dims=[embed_dim],
            normalize_input=True,
        )

        # Pairwise feature embedding: lorentz_vectors → attention bias
        # PairEmbed outputs (B*num_heads, P, P) used as attn_mask in Block
        self.pair_embed = PairEmbed(
            pairwise_lv_dim=pair_input_dim,
            pairwise_input_dim=0,
            dims=pair_embed_dims + [num_heads],
            remove_self_pair=False,
            use_pre_activation_pair=True,
        )

        # Transformer blocks with pairwise attention bias
        block_config = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            attn_dropout=dropout,
            activation_dropout=dropout,
            activation='gelu',
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        self.blocks = nn.ModuleList([
            Block(**block_config) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        # Per-track scoring head: encoded embedding → scalar score
        self.scoring_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Score each track using pairwise-bias self-attention.

        Args:
            points: (B, 2, K1) coordinates in (η, φ).
            features: (B, input_dim, K1) per-track features.
            lorentz_vectors: (B, 4, K1) raw 4-vectors (px, py, pz, E).
            mask: (B, 1, K1) validity mask.
            stage1_scores: (B, K1) scores from Stage 1 pre-filter.

        Returns:
            scores: (B, K1) per-track scores. Padded tracks get -inf.
        """
        valid_mask = mask.squeeze(1).bool()  # (B, K1)
        padding_mask = ~valid_mask  # (B, K1) — True for padded positions
        mask_float = mask.float()

        # Concatenate stage1 scores as an extra feature channel.
        # Replace -inf (padded tracks from select_top_k) with 0 before
        # multiplication — (-inf * 0.0 = NaN) in float arithmetic.
        safe_stage1_scores = stage1_scores.masked_fill(
            ~valid_mask, 0.0,
        )
        stage1_channel = safe_stage1_scores.unsqueeze(1)  # (B, 1, K1)
        combined_features = torch.cat(
            [features, stage1_channel], dim=1,
        ) * mask_float  # (B, input_dim+1, K1)

        # Input embedding: (B, C, K1) → (K1, B, embed_dim)
        track_embeddings = self.embed(combined_features)
        track_embeddings = track_embeddings.masked_fill(
            ~mask.bool().permute(2, 0, 1), 0,
        )

        # Pairwise attention bias from Lorentz vectors:
        # PairEmbed computes ln kT, ln z, ln ΔR, ln m² for all pairs
        # and projects to (B, num_heads, K1, K1) → reshape to (B*H, K1, K1)
        #
        # .detach() prevents NaN gradients from pairwise_lv_fts():
        # sqrt(ΔR²) has gradient 0.5/sqrt(0) = inf for self-pairs (ΔR=0).
        # Pairwise features are physics constants used as attention bias —
        # they don't need gradients w.r.t. input 4-vectors.
        # .float() ensures float32 precision for the ln/sqrt operations
        # even when AMP casts the rest to float16.
        lorentz_for_pairs = (lorentz_vectors * mask_float).detach().float()
        attention_bias = self.pair_embed(
            lorentz_for_pairs, uu=None,
        )  # (B, num_heads, K1, K1)
        num_heads = attention_bias.shape[1]
        attention_bias = attention_bias.view(
            -1, num_heads, attention_bias.shape[2], attention_bias.shape[3],
        ).reshape(-1, attention_bias.shape[2], attention_bias.shape[3])
        # (B*num_heads, K1, K1)

        # Transformer blocks with pairwise bias
        # Block expects: x=(K1, B, embed_dim), padding_mask=(B, K1),
        #                attn_mask=(B*H, K1, K1)
        encoded = track_embeddings
        for block in self.blocks:
            encoded = block(
                encoded,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=attention_bias,
            )

        # Per-track scoring: (K1, B, embed_dim) → (B, K1)
        encoded = self.output_norm(encoded)  # (K1, B, embed_dim)
        encoded = encoded.permute(1, 0, 2)  # (B, K1, embed_dim)
        scores = self.scoring_head(encoded).squeeze(-1)  # (B, K1)

        # Mask padded tracks
        scores = scores.masked_fill(padding_mask, float('-inf'))

        return scores

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute temperature-scaled pairwise ranking loss.

        Loss = T × softplus((s_neg - s_pos) / T)

        Same ranking loss as TrackPreFilter, operating on the filtered
        K1-track set where positives are enriched relative to the full event.

        Returns:
            dict with 'total_loss', 'ranking_loss', '_scores'.
        """
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        valid_mask = mask.squeeze(1).bool()
        labels = (
            track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()
        )

        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        event_losses = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = labels[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels == 0.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=scores.device,
            )
            sampled_negatives = negative_indices[sample_idx]

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            # L = T × log(1 + exp((s_neg - s_pos) / T))
            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)
            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            ranking_loss = torch.tensor(
                0.0, device=scores.device, dtype=scores.dtype,
            )
        else:
            ranking_loss = torch.stack(event_losses).mean()

        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
            '_scores': scores,
        }
