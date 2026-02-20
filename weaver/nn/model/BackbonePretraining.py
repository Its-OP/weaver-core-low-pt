"""Masked Track Reconstruction for Backbone Pretraining.

Self-supervised pretraining via masked track reconstruction with the
two-stage Enrich-Compact backbone:
    1. Enrich ALL tracks with neighbor context (ParticleNeXt-style)
    2. Randomly mask 40% of tracks
    3. Densely pack visible enriched tracks → compact via PointNet++
    4. Decode masked tracks via cross-attention to backbone tokens
    5. Reconstruct original 7 raw features (standardized by weaver)

Masking happens BETWEEN enrichment and compaction. This way, visible
tracks carry partial information about masked neighbors from the
enrichment message passing — making reconstruction solvable but
not trivial (the decoder can't just copy).

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

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone


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
    """Wrapper combining enrichment + masking + compaction + decoder.

    Two-stage pretraining forward flow:
        1. ENRICH: All tracks → ParticleNeXt MultiScaleEdgeConv → (B, 256, P)
        2. MASK: Randomly select 40% of valid tracks to mask
        3. GATHER VISIBLE: Densely pack visible enriched tracks (FPS needs
           contiguous valid points — can't just zero out masked tracks)
        4. COMPACT: Visible tracks → PointNet++ set abstraction → 128 tokens
        5. GATHER GT: Densely pack ground truth raw features for masked tracks
        6. DECODE: Cross-attention from query embeddings to backbone tokens
        7. LOSS: MSE between predicted and true raw 7 features

    Masking between enrichment and compaction means:
        - Visible tracks carry partial info about masked neighbors (from
          enrichment message passing) — the reconstruction target is solvable
        - But the decoder can't just copy — it must decode from compressed
          backbone tokens, forcing the backbone to learn useful representations

    Features arrive already standardized by weaver's data pipeline
    (preprocess.method: auto → median-centering + IQR scaling + clipping
    to [-5, 5]), so the MSE loss is naturally well-scaled.

    Returns per-event loss tensor (B,) for the custom training script.

    Args:
        backbone_kwargs: Keyword arguments for EnrichCompactBackbone.
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

        self.backbone = EnrichCompactBackbone(**backbone_kwargs)

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

    @staticmethod
    def _gather_tracks(
        tensor: torch.Tensor,
        selection_mask: torch.Tensor,
        max_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Densely pack selected tracks into a contiguous tensor.

        Used for both visible tracks (before compaction) and masked tracks
        (ground truth for loss). Dense packing is required because FPS in
        CompactionStage operates on coordinates — zeroed-out masked tracks
        at coordinate (0, 0) would be selected as centroids, corrupting
        the spatial downsampling.

        Fully vectorized — no Python loops over the batch dimension.
        Uses argsort on the inverted selection mask to push selected tracks
        to the front of each row, then slices [:max_count].

        Args:
            tensor: (B, C, P) input tensor.
            selection_mask: (B, 1, P) boolean mask, True for selected tracks.
            max_count: Maximum number of selected tracks across the batch.

        Returns:
            Tuple of:
                gathered: (B, C, max_count) dense tensor of selected tracks.
                    Positions beyond per-event count are zero-padded.
                validity_mask: (B, 1, max_count) boolean mask, True for valid
                    slots (False for zero-padding at the end).
        """
        batch_size, num_channels, num_points = tensor.shape
        device = tensor.device

        selection_flat = selection_mask.squeeze(1)  # (B, P)

        # Argsort trick: sort so selected (True=1) come first.
        # ~selection_flat converts True→0, False→1; argsort puts 0s first.
        # Within the selected group, original order is preserved (stable sort).
        sorted_indices = (~selection_flat).long().argsort(
            dim=1, stable=True
        )  # (B, P)

        # Take the first max_count indices per event
        gather_indices = sorted_indices[:, :max_count]  # (B, max_count)

        # Expand indices for gathering: (B, C, max_count)
        gather_indices_expanded = gather_indices.unsqueeze(1).expand(
            -1, num_channels, -1
        )

        # Gather features
        gathered = tensor.gather(2, gather_indices_expanded)  # (B, C, max_count)

        # Build validity mask: position j is valid if j < num_selected[b]
        num_selected = selection_flat.sum(dim=1)  # (B,)
        position_indices = torch.arange(
            max_count, device=device
        ).unsqueeze(0)  # (1, max_count)
        validity_mask = (
            position_indices < num_selected.unsqueeze(1)
        ).unsqueeze(1)  # (B, 1, max_count)

        # Zero out padding positions (gathered may contain garbage from
        # unselected tracks that landed in the first max_count slots
        # when num_selected < max_count)
        gathered = gathered * validity_mask.float()

        return gathered, validity_mask

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: enrich → mask → compact → decode → loss.

        Flow:
            1. ENRICH all tracks (ParticleNeXt MultiScaleEdgeConv)
            2. MASK 40% of valid tracks
            3. GATHER visible enriched tracks into dense tensor
            4. COMPACT visible tracks (PointNet++ set abstraction → 128 tokens)
            5. GATHER ground truth raw features for masked tracks
            6. DECODE masked tracks from backbone tokens
            7. LOSS: MSE on raw 7 features

        Features (pf_features) arrive already standardized by weaver's data
        pipeline (preprocess.method: auto → median-centering + IQR scaling
        + clipping to [-5, 5]). The decoder reconstructs these standardized
        values, and the MSE loss is naturally well-scaled.

        4-vectors (pf_vectors) arrive raw — the enrichment stage needs raw
        momenta for pairwise_lv_fts(). They are NOT reconstructed because
        they are fully derivable from the features (which contain px, py, pz).

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) per-track features (standardized).
            lorentz_vectors: (B, 4, P) per-track 4-vectors (raw px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            per_event_loss: (B,) feature reconstruction loss per event.
        """
        batch_size = features.shape[0]
        num_features = features.shape[1]

        # Step 1: ENRICH all tracks with neighbor context
        # All tracks participate — no masking yet. Each track accumulates
        # information from its kNN neighborhood via MultiScaleEdgeConv.
        enriched_features = self.backbone.enrich(
            points, features, lorentz_vectors, mask,
        )  # (B, enrichment_output_dim, P)

        # Step 2: MASK — create random visible/masked split
        visible_mask, masked_mask = self._create_random_mask(mask)

        # Count per event
        num_visible_per_event = visible_mask.squeeze(1).sum(dim=1)  # (B,)
        num_masked_per_event = masked_mask.squeeze(1).sum(dim=1)  # (B,)
        max_visible = num_visible_per_event.max().item()
        max_masked = num_masked_per_event.max().item()

        if max_masked == 0:
            # Edge case: no tracks to mask
            return torch.zeros(
                batch_size, device=features.device, dtype=features.dtype
            )

        # Step 3: GATHER visible enriched tracks into dense tensors
        # Dense packing is required because FPS in CompactionStage operates
        # on coordinates — zeroed-out masked tracks at (0, 0) would be
        # selected as centroids, corrupting the spatial downsampling.
        visible_enriched, visible_validity = self._gather_tracks(
            enriched_features, visible_mask, max_visible,
        )  # (B, enrichment_dim, max_visible), (B, 1, max_visible)

        visible_coordinates, _ = self._gather_tracks(
            points, visible_mask, max_visible,
        )  # (B, 2, max_visible), _

        # Step 4: COMPACT visible tracks → dense backbone tokens
        backbone_tokens, _ = self.backbone.compact(
            visible_coordinates, visible_enriched, visible_validity,
        )  # (B, output_dim, M)

        # Step 5: GATHER ground truth raw features for masked tracks
        # Target is raw 7-feature standardized input (not enriched features):
        # stable, physically meaningful, interpretable loss.
        masked_true_features, masked_validity = self._gather_tracks(
            features, masked_mask, max_masked,
        )  # (B, input_dim, max_masked), (B, 1, max_masked)

        # Step 6: DECODE masked tracks from backbone tokens
        # The decoder receives only backbone tokens and the count of masked
        # tracks. No (η, φ) coordinates — this prevents the decoder from
        # bypassing the backbone via spatial shortcuts.
        predicted_features = self.decoder(
            backbone_tokens, max_masked,
        )  # (B, num_output_features, max_masked)

        # Step 7: Compute feature reconstruction loss per event
        # L = (1 / N_masked) × Σ_i Σ_f (pred_f_i - true_f_i)²
        #
        # Features are already standardized (clipped to [-5, 5]) by weaver,
        # so MSE is well-scaled across all feature channels.
        # Use the validity mask to exclude zero-padded slots.
        track_valid = masked_validity.float()  # (B, 1, max_masked)

        feature_error = (
            (predicted_features - masked_true_features).square() * track_valid
        )  # (B, num_features, max_masked)
        per_event_loss = feature_error.sum(dim=(1, 2)) / (
            num_masked_per_event.float() * num_features
        ).clamp(min=1.0)  # (B,)

        return per_event_loss
