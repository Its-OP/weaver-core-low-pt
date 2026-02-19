"""Masked Track Reconstruction for Backbone Pretraining.

Self-supervised pretraining via masked track reconstruction:
    1. Randomly mask 40% of input tracks
    2. Encode visible tracks through HierarchicalGraphBackbone → 64 tokens
    3. Decode masked tracks via cross-attention to backbone tokens
    4. Reconstruct masked track features (standardized by weaver's pipeline)

The decoder is discarded after pretraining. Only the backbone is kept.

MaskedTrackPretrainer.forward() returns (B,) per-event loss tensor.
The custom training script (pretrain_backbone.py) calls .mean().backward().
"""
import math
import torch
import torch.nn as nn

from weaver.nn.model.HierarchicalGraphBackbone import HierarchicalGraphBackbone


class PositionalEncoding2D(nn.Module):
    """Sinusoidal positional encoding from (η, φ) coordinates.

    Used only in the pretraining decoder to differentiate identical [MASK]
    tokens — each masked track query needs to know *which* (η, φ) location
    it should reconstruct. This does NOT break permutation invariance of
    the backbone, which never uses positional encoding.

    Encodes 2D spatial positions into a high-dimensional vector using
    sinusoidal functions at different frequencies.

    For each coordinate c ∈ {η, φ} and frequency index k:
        PE(c, 2k)   = sin(c / 10000^(2k/d))
        PE(c, 2k+1) = cos(c / 10000^(2k/d))

    Total output dimension = encoding_dim (split equally between η and φ).

    Args:
        encoding_dim: Output dimension of the positional encoding.
            Must be divisible by 4 (sin/cos for each of η and φ).
    """

    def __init__(self, encoding_dim: int):
        super().__init__()
        assert encoding_dim % 4 == 0, (
            f"encoding_dim must be divisible by 4, got {encoding_dim}"
        )
        self.encoding_dim = encoding_dim
        quarter_dim = encoding_dim // 4

        # Frequency bands: 1 / 10000^(2k/d) for k = 0, ..., quarter_dim-1
        frequencies = torch.exp(
            torch.arange(quarter_dim, dtype=torch.float32)
            * (-math.log(10000.0) / quarter_dim)
        )
        self.register_buffer('frequencies', frequencies)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode (η, φ) coordinates into positional embeddings.

        Args:
            coordinates: (B, 2, N) where dim 1 is (η, φ).

        Returns:
            positional_encoding: (B, encoding_dim, N).
        """
        eta = coordinates[:, 0:1, :]  # (B, 1, N)
        phi = coordinates[:, 1:2, :]  # (B, 1, N)

        # frequencies: (quarter_dim,) → (1, quarter_dim, 1) for broadcasting
        freq = self.frequencies.view(1, -1, 1)

        # (B, 1, N) × (1, quarter_dim, 1) → (B, quarter_dim, N)
        eta_enc = torch.cat([
            torch.sin(eta * freq),
            torch.cos(eta * freq),
        ], dim=1)  # (B, half_dim, N)

        phi_enc = torch.cat([
            torch.sin(phi * freq),
            torch.cos(phi * freq),
        ], dim=1)  # (B, half_dim, N)

        return torch.cat([eta_enc, phi_enc], dim=1)  # (B, encoding_dim, N)


class MaskedTrackDecoder(nn.Module):
    """Decoder for masked track reconstruction.

    Similar to MAE/Point-MAE decoders. Reconstructs per-track features
    for masked tracks by cross-attending to backbone tokens.

    Architecture:
        1. Learnable [MASK] embedding + positional encoding from (η, φ)
        2. Cross-attention: masked track queries attend to backbone tokens
        3. Self-attention: N_layers among masked track queries
        4. Output projection: Linear(decoder_dim → num_output_features)

    Note: We only reconstruct pf_features (standardized by weaver), not
    pf_vectors (raw 4-momenta). The 4-vectors (px, py, pz, E) are fully
    derivable from the features (which already contain px, py, pz) so
    reconstructing them separately would add no new learning signal.

    Args:
        backbone_dim: Channel dimension of backbone tokens (default: 256).
        decoder_dim: Internal dimension of the decoder (default: 128).
        num_heads: Number of attention heads (default: 4).
        num_self_attention_layers: Number of self-attention layers (default: 2).
        num_output_features: Number of track features to reconstruct (default: 7).
        dropout: Dropout rate in attention layers (default: 0.0).
    """

    def __init__(
        self,
        backbone_dim: int = 256,
        decoder_dim: int = 128,
        num_heads: int = 4,
        num_self_attention_layers: int = 2,
        num_output_features: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.num_output_features = num_output_features

        # Learnable [MASK] token embedding
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional encoding from (η, φ) coordinates
        self.positional_encoding = PositionalEncoding2D(decoder_dim)

        # Project backbone tokens to decoder dimension
        self.backbone_projection = nn.Linear(backbone_dim, decoder_dim)

        # Cross-attention: masked queries attend to backbone tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(decoder_dim)

        # Self-attention layers among masked track queries
        self.self_attention_layers = nn.ModuleList()
        self.self_attention_norms = nn.ModuleList()
        self.feedforward_layers = nn.ModuleList()
        self.feedforward_norms = nn.ModuleList()
        for _ in range(num_self_attention_layers):
            self.self_attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=decoder_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.self_attention_norms.append(nn.LayerNorm(decoder_dim))
            self.feedforward_layers.append(nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(decoder_dim * 4, decoder_dim),
                nn.Dropout(dropout),
            ))
            self.feedforward_norms.append(nn.LayerNorm(decoder_dim))

        # Output projection: predict features per masked track
        self.output_projection = nn.Linear(
            decoder_dim, num_output_features
        )

    def forward(
        self,
        backbone_tokens: torch.Tensor,
        masked_track_coordinates: torch.Tensor,
        num_masked_tracks: int,
    ) -> torch.Tensor:
        """Decode masked tracks from backbone tokens.

        Args:
            backbone_tokens: (B, C_backbone, M) dense tokens from backbone.
            masked_track_coordinates: (B, 2, N_masked) (η, φ) of masked tracks.
            num_masked_tracks: N_masked — number of masked tracks per event.

        Returns:
            predicted_features: (B, num_output_features, N_masked)
        """
        batch_size = backbone_tokens.shape[0]

        # Project backbone tokens: (B, C_backbone, M) → (B, M, decoder_dim)
        memory = self.backbone_projection(
            backbone_tokens.transpose(1, 2)
        )  # (B, M, decoder_dim)

        # Build masked track queries: [MASK] embedding + positional encoding
        # mask_token: (1, decoder_dim, 1) → (B, decoder_dim, N_masked)
        queries = self.mask_token.expand(
            batch_size, -1, num_masked_tracks
        )  # (B, decoder_dim, N_masked)

        # Add positional encoding from masked track (η, φ) coordinates
        position_encoding = self.positional_encoding(
            masked_track_coordinates
        )  # (B, decoder_dim, N_masked)
        queries = queries + position_encoding

        # Transpose to (B, N_masked, decoder_dim) for nn.MultiheadAttention
        queries = queries.transpose(1, 2)  # (B, N_masked, decoder_dim)

        # Cross-attention: masked queries attend to backbone tokens
        cross_attention_output, _ = self.cross_attention(
            query=queries, key=memory, value=memory
        )
        queries = self.cross_attention_norm(queries + cross_attention_output)

        # Self-attention layers
        for (
            self_attention,
            self_attention_norm,
            feedforward,
            feedforward_norm,
        ) in zip(
            self.self_attention_layers,
            self.self_attention_norms,
            self.feedforward_layers,
            self.feedforward_norms,
        ):
            # Self-attention with residual
            self_attention_output, _ = self_attention(
                query=queries, key=queries, value=queries
            )
            queries = self_attention_norm(queries + self_attention_output)

            # Feedforward with residual
            feedforward_output = feedforward(queries)
            queries = feedforward_norm(queries + feedforward_output)

        # Output projection: (B, N_masked, decoder_dim) → (B, N_masked, F)
        predictions = self.output_projection(queries)

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
        masked_coordinates = self._gather_masked_tracks(
            points, masked_mask, max_masked
        )  # (B, 2, max_masked)
        masked_true_features = self._gather_masked_tracks(
            features, masked_mask, max_masked
        )  # (B, input_dim, max_masked)

        # Step 4: Decode masked tracks (features only)
        predicted_features = self.decoder(
            backbone_tokens, masked_coordinates, max_masked
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


