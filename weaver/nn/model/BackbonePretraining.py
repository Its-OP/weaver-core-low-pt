"""Masked Track Reconstruction for Backbone Pretraining.

Self-supervised pretraining via masked track reconstruction:
    1. Randomly mask 40% of input tracks
    2. Encode visible tracks through HierarchicalGraphBackbone → 64 tokens
    3. Decode masked tracks via cross-attention to backbone tokens
    4. Reconstruct masked track features (standardized by weaver's pipeline)

The decoder is discarded after pretraining. Only the backbone is kept.

MaskedTrackPretrainer.forward() returns (B,) per-event loss tensor.
The custom training script (pretrain_backbone.py) calls .mean().backward().

Design note on decoder simplicity:
    The decoder is intentionally minimal — just cross-attention + output MLP.
    No self-attention among masked queries, no (η, φ) positional encoding.
    This prevents the decoder from bypassing the backbone:

    Problem: With (η, φ) PE + self-attention, the decoder can learn spatial
    statistics (local density, typical momentum at a position) from the
    masked queries alone, without needing the backbone. This causes the loss
    to plateau at ~0.75 (the positional-encoding floor) with zero gradient
    pressure on the backbone.

    Fix: Learnable index embeddings (no physics info) differentiate queries,
    and cross-attention is the ONLY way to access event-specific information.
    This forces the backbone to encode useful representations.
"""
import torch
import torch.nn as nn

from weaver.nn.model.HierarchicalGraphBackbone import HierarchicalGraphBackbone


class MaskedTrackDecoder(nn.Module):
    """Minimal decoder for masked track reconstruction.

    Intentionally simple to prevent decoder shortcut / backbone bypass.
    The decoder must extract ALL event-specific information from the
    backbone tokens via cross-attention — it has no other source.

    Architecture:
        1. Learnable query embeddings (physics-free query differentiation)
        2. Cross-attention: masked track queries attend to backbone tokens
        3. Output MLP: project cross-attention output → predicted features

    Why learnable query embeddings (not shared token + noise):
        Each masked query needs a distinct embedding so that after W_Q
        projection, different queries produce different attention patterns
        over the 64 backbone tokens. A shared token + noise approach fails
        because the noise gets attenuated through W_Q projection and
        dot-product with keys: for decoder_dim=128 with 4 heads (d_head=32),
        achieving O(1) logit variation would require noise_std ≈ 6 — large
        enough to drown the token itself. Learnable embeddings avoid this
        by having proper scale from the start (initialized at std=1/√d).

    Why no (η, φ) positional encoding:
        Sinusoidal PE from (η, φ) leaks spatial information into queries.
        Combined with self-attention, this lets the decoder learn spatial
        statistics (density, typical pT at a position) without the backbone.
        Learnable index embeddings only differentiate queries — they carry
        zero physics information since particle sets are permutation-invariant.

    Why no self-attention among masked queries:
        Self-attention lets ~450 masked queries exchange positional info,
        creating an information pathway that bypasses the backbone entirely.
        Without it, each query independently cross-attends to backbone
        tokens, forcing the backbone to be the sole information source.

    Note: We only reconstruct pf_features (standardized by weaver), not
    pf_vectors (raw 4-momenta). The 4-vectors (px, py, pz, E) are fully
    derivable from the features (which already contain px, py, pz) so
    reconstructing them separately would add no new learning signal.

    Args:
        backbone_dim: Channel dimension of backbone tokens (default: 256).
        decoder_dim: Internal dimension of the decoder (default: 128).
        num_heads: Number of cross-attention heads (default: 4).
        num_output_features: Number of track features to reconstruct (default: 7).
        max_masked_tracks: Maximum number of masked tracks (vocab size for
            query embeddings). Must be >= mask_ratio × max_tracks_per_event.
            Default: 1200 (supports 0.4 × 2800 = 1120 masked tracks).
        dropout: Dropout rate in cross-attention (default: 0.0).
    """

    def __init__(
        self,
        backbone_dim: int = 256,
        decoder_dim: int = 128,
        num_heads: int = 4,
        num_output_features: int = 7,
        max_masked_tracks: int = 1200,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.num_output_features = num_output_features

        # Learnable query embeddings: one per masked-track slot.
        # Each embedding is a distinct learned vector that produces a unique
        # attention pattern over backbone tokens via W_Q projection.
        #
        # These embeddings carry zero physics information — they're just
        # arbitrary slot indices (particle sets are permutation-invariant,
        # so index 0 vs index 1 has no physical meaning). The only way for
        # a query to produce a meaningful prediction is through cross-
        # attention to backbone tokens.
        self.query_embeddings = nn.Embedding(max_masked_tracks, decoder_dim)
        nn.init.normal_(self.query_embeddings.weight, std=decoder_dim ** -0.5)

        # LayerNorm on queries and keys before cross-attention.
        # Attention logits = (Q·K^T)/√d_head require Q and K at comparable
        # scales for non-degenerate softmax. Without normalization:
        #   - At init: backbone output std ≈ 0.05, queries std ≈ 0.09
        #     → logits ≈ 0 → uniform attention → no gradient signal
        #   - After training: backbone output std ≈ 1.8, queries still ≈ 0.09
        #     → logits dominated by key scale → queries ignored
        # LayerNorm on both ensures O(1) logit variance from the start.
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.memory_norm = nn.LayerNorm(decoder_dim)

        # Project backbone tokens to decoder dimension
        self.backbone_projection = nn.Linear(backbone_dim, decoder_dim)

        # Cross-attention: masked queries attend to backbone tokens.
        # This is the ONLY way event-specific information enters the decoder.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(decoder_dim)

        # Output MLP: cross-attention output → predicted features.
        # One hidden layer with GELU provides enough capacity to decode
        # backbone representations into 7 output features, without being
        # so powerful that it can memorize patterns independently.
        self.output_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, num_output_features),
        )

    def forward(
        self,
        backbone_tokens: torch.Tensor,
        num_masked_tracks: int,
    ) -> torch.Tensor:
        """Decode masked tracks from backbone tokens.

        Args:
            backbone_tokens: (B, C_backbone, M) dense tokens from backbone.
            num_masked_tracks: N_masked — number of masked tracks to predict.

        Returns:
            predicted_features: (B, num_output_features, N_masked)
        """
        batch_size = backbone_tokens.shape[0]
        device = backbone_tokens.device

        # Project backbone tokens: (B, C_backbone, M) → (B, M, decoder_dim)
        # LayerNorm stabilizes key scale so attention logits are well-scaled
        # regardless of backbone output magnitude (which changes over training).
        memory = self.memory_norm(
            self.backbone_projection(backbone_tokens.transpose(1, 2))
        )  # (B, M, decoder_dim)

        # Build masked track queries from learnable embeddings.
        # Each query gets a unique learned vector (no physics info, just
        # slot differentiation) that produces a distinct W_Q projection,
        # yielding diverse attention patterns over backbone tokens.
        # LayerNorm ensures query scale matches key scale for well-scaled
        # attention logits from the start of training.
        query_indices = torch.arange(
            num_masked_tracks, device=device
        )  # (N_masked,)
        queries = self.query_norm(
            self.query_embeddings(query_indices)
        )  # (N_masked, decoder_dim)
        queries = queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, N_masked, decoder_dim)

        # Cross-attention: each masked query independently attends to
        # backbone tokens. This is the sole information pathway from
        # the encoder — no self-attention, no positional encoding leak.
        cross_attention_output, _ = self.cross_attention(
            query=queries, key=memory, value=memory
        )
        queries = self.cross_attention_norm(queries + cross_attention_output)

        # Output MLP: (B, N_masked, decoder_dim) → (B, N_masked, F)
        predictions = self.output_mlp(queries)

        # Transpose to (B, num_output_features, N_masked)
        return predictions.transpose(1, 2)


class MaskedTrackPretrainer(nn.Module):
    """Wrapper combining masking + backbone + decoder for pretraining.

    Performs masked track reconstruction:
        1. Randomly mask a fraction of input tracks
        2. Encode visible tracks through backbone
        3. Decode masked tracks via cross-attention
        4. Compute feature reconstruction loss (MSE) on masked tracks

    Only features (pf_features) are reconstructed — not 4-vectors, because
    pf_vectors (px, py, pz, E) are fully derivable from pf_features (which
    already contain px, py, pz). Reconstructing them would add no new
    learning signal.

    Features arrive already standardized by weaver's data pipeline
    (preprocess.method: auto → median-centering + IQR scaling + clipping
    to [-5, 5]), so the MSE loss is naturally well-scaled.

    Returns per-event loss tensor (B,) for integration with weaver's
    train_regression mode.

    Args:
        backbone_kwargs: Keyword arguments for HierarchicalGraphBackbone.
        decoder_kwargs: Keyword arguments for MaskedTrackDecoder.
        mask_ratio: Fraction of tracks to mask (default: 0.4).
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        mask_ratio: float = 0.4,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.mask_ratio = mask_ratio

        self.backbone = HierarchicalGraphBackbone(**backbone_kwargs)

        # Set decoder backbone_dim to match backbone output
        decoder_kwargs.setdefault('backbone_dim', self.backbone.output_dim)
        decoder_kwargs.setdefault(
            'num_output_features', backbone_kwargs.get('input_dim', 7)
        )
        self.decoder = MaskedTrackDecoder(**decoder_kwargs)

    def _create_random_mask(
        self, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create random visible/masked split respecting the input mask.

        Args:
            mask: (B, 1, P) boolean input mask, True for valid tracks.

        Returns:
            visible_mask: (B, 1, P) True for visible (not masked) valid tracks.
            masked_mask: (B, 1, P) True for masked valid tracks.
        """
        batch_size, _, num_points = mask.shape
        device = mask.device

        valid_mask = mask.squeeze(1).bool()  # (B, P)

        # Generate random scores, set invalid to +∞ so they sort last
        random_scores = torch.rand(
            batch_size, num_points, device=device, dtype=torch.float32
        )
        random_scores.masked_fill_(~valid_mask, float('inf'))

        # Sort: lowest scores first → these become masked
        sorted_indices = random_scores.argsort(dim=1)  # (B, P)

        # For each event, the first floor(mask_ratio × num_valid) are masked
        num_valid = valid_mask.sum(dim=1, keepdim=True)  # (B, 1)
        num_to_mask = (self.mask_ratio * num_valid.float()).long()  # (B, 1)

        # Create rank tensor: rank[b, sorted_indices[b, r]] = r
        ranks = torch.zeros_like(random_scores, dtype=torch.long)
        ranks.scatter_(
            1, sorted_indices,
            torch.arange(num_points, device=device).unsqueeze(0).expand(
                batch_size, -1
            ),
        )

        # Points with rank < num_to_mask are masked
        is_masked = (ranks < num_to_mask) & valid_mask  # (B, P)
        is_visible = valid_mask & ~is_masked  # (B, P)

        return (
            is_visible.unsqueeze(1),  # (B, 1, P)
            is_masked.unsqueeze(1),   # (B, 1, P)
        )

    def _gather_masked_tracks(
        self,
        tensor: torch.Tensor,
        masked_mask: torch.Tensor,
        max_masked: int,
    ) -> torch.Tensor:
        """Gather masked track data into a dense tensor.

        Args:
            tensor: (B, C, P) input tensor.
            masked_mask: (B, 1, P) boolean mask for masked tracks.
            max_masked: Maximum number of masked tracks across the batch.

        Returns:
            gathered: (B, C, max_masked) dense tensor of masked track data.
        """
        batch_size, num_channels, _ = tensor.shape
        device = tensor.device

        masked_flat = masked_mask.squeeze(1)  # (B, P)

        gathered = torch.zeros(
            batch_size, num_channels, max_masked,
            device=device, dtype=tensor.dtype,
        )
        for batch_idx in range(batch_size):
            indices = masked_flat[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            num_masked = indices.numel()
            if num_masked > 0:
                gathered[batch_idx, :, :num_masked] = tensor[
                    batch_idx, :, indices
                ]

        return gathered

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: mask → encode → decode → compute feature MSE loss.

        Features (pf_features) arrive already standardized by weaver's data
        pipeline (preprocess.method: auto → median-centering + IQR scaling
        + clipping to [-5, 5]). The decoder reconstructs these standardized
        values, and the MSE loss is naturally well-scaled.

        4-vectors (pf_vectors) arrive raw — the backbone needs raw momenta
        for pairwise_lv_fts(). They are NOT reconstructed because they are
        fully derivable from the features (which contain px, py, pz).

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) per-track features (standardized).
            lorentz_vectors: (B, 4, P) per-track 4-vectors (raw px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            per_event_loss: (B,) feature reconstruction loss per event.
        """
        # Step 1: Create random visible/masked split
        visible_mask, masked_mask = self._create_random_mask(mask)

        # Count masked tracks per event for gathering
        num_masked_per_event = masked_mask.squeeze(1).sum(dim=1)  # (B,)
        max_masked = num_masked_per_event.max().item()

        if max_masked == 0:
            # Edge case: no tracks to mask
            return torch.zeros(
                features.shape[0], device=features.device, dtype=features.dtype
            )

        # Step 2: Encode visible tracks through backbone
        # Zero out masked tracks so they don't influence the backbone
        visible_features = features * visible_mask.float()
        visible_lorentz_vectors = lorentz_vectors * visible_mask.float()
        visible_points = points * visible_mask.float()

        backbone_tokens, _, _ = self.backbone(
            visible_points, visible_features, visible_lorentz_vectors,
            visible_mask.float(),
        )  # (B, C_backbone, M)

        # Step 3: Gather ground truth for masked tracks
        masked_true_features = self._gather_masked_tracks(
            features, masked_mask, max_masked
        )  # (B, input_dim, max_masked)

        # Step 4: Decode masked tracks (features only)
        # The decoder receives only backbone tokens and the count of masked
        # tracks. No (η, φ) coordinates — this prevents the decoder from
        # bypassing the backbone via spatial shortcuts.
        predicted_features = self.decoder(
            backbone_tokens, max_masked
        )  # (B, num_output_features, max_masked)

        # Step 5: Compute feature reconstruction loss per event
        # L = (1 / N_masked) × Σ_i Σ_f (pred_f_i - true_f_i)²
        #
        # Features are already standardized (clipped to [-5, 5]) by weaver,
        # so MSE is well-scaled across all feature channels.

        # Per-track valid mask for the gathered dense tensor
        # (some events may have fewer masked tracks than max_masked)
        track_valid = torch.zeros(
            features.shape[0], 1, max_masked,
            device=features.device, dtype=features.dtype,
        )
        for batch_idx in range(features.shape[0]):
            track_valid[batch_idx, :, :num_masked_per_event[batch_idx]] = 1.0

        # MSE on weaver-standardized features, averaged per event
        feature_error = (
            (predicted_features - masked_true_features).square() * track_valid
        )  # (B, num_features, max_masked)
        per_event_loss = feature_error.sum(dim=(1, 2)) / (
            num_masked_per_event.float() * features.shape[1]
        ).clamp(min=1.0)  # (B,)

        return per_event_loss


