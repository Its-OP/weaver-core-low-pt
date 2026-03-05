"""Masked Track Reconstruction for Backbone Pretraining.

Self-supervised pretraining via masked track reconstruction with the
two-stage Enrich-Compact backbone:
    1. Enrich ALL tracks with neighbor context (ParticleNeXt-style)
    2. Randomly mask a fraction of tracks
    3. Densely pack visible enriched tracks → compact via PointNet++
    4. Decode masked tracks via cross-attention to backbone tokens
    5. Reconstruct original 7 raw features (standardized by weaver)
    6. Hungarian-match predicted set to ground-truth set, compute MSE

The decoder outputs a SET of predicted tracks with no inherent ordering.
Hungarian matching (optimal bipartite assignment) finds the best 1-to-1
pairing between predictions and targets before computing the loss. This
eliminates the spurious assignment problem that would otherwise inject
noise into gradients.

Masking happens BETWEEN enrichment and compaction. This way, visible
tracks carry partial information about masked neighbors from the
enrichment message passing — making reconstruction solvable but
not trivial (the decoder can't just copy).

The decoder is discarded after pretraining. Only the backbone is kept.

MaskedTrackPretrainer.forward() returns (B,) per-event loss tensor.
The custom training script (pretrain_backbone.py) calls .mean().backward().

Design note on decoder:
    The decoder uses DETR-style stacked transformer decoder layers
    (self-attention → cross-attention → FFN), with learnable index
    embeddings as queries (no (η, φ) positional encoding).

    Learnable index embeddings carry zero physics information — particle
    sets are permutation-invariant, so index 0 vs index 1 has no physical
    meaning. The only way for queries to access event-specific information
    is through cross-attention to backbone tokens, which forces the backbone
    to encode useful representations.

    The number of decoder layers is configurable (default: 1). With multiple
    layers, self-attention among queries helps coordinate predictions to
    avoid duplicates (standard in DETR-family models).
"""
import torch
import torch.nn as nn

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.hungarian_matcher import hungarian_matcher
from weaver.nn.model.hungarian_matcher import sinkhorn_matcher


class MaskedTrackDecoder(nn.Module):
    """DETR-style decoder for masked track reconstruction.

    Uses stacked transformer decoder layers (self-attention → cross-attention
    → FFN) to predict masked track features from backbone tokens.

    Architecture:
        1. Learnable query embeddings (physics-free slot differentiation)
        2. Backbone projection + LayerNorm (project backbone tokens to
           decoder dimension with stable scale)
        3. N stacked TransformerDecoderLayers, each containing:
           - Self-attention among queries (coordinate predictions, avoid
             duplicates — standard in DETR-family models)
           - Cross-attention to backbone tokens (sole event-specific
             information source)
           - Feedforward network (4× expansion, GELU activation)
           All with pre-norm (LayerNorm before attention) for stability.
        4. Output MLP: project decoded queries → predicted features

    Why learnable query embeddings (not positional encoding):
        Each masked query needs a distinct embedding so that after W_Q
        projection, different queries produce different attention patterns
        over the backbone tokens. These embeddings carry zero physics
        information — particle sets are permutation-invariant, so index 0
        vs index 1 has no physical meaning. The only way for a query to
        access event-specific information is through cross-attention to
        backbone tokens, which forces the backbone to be the sole encoder.

    Note: We only reconstruct pf_features (standardized by weaver), not
    pf_vectors (raw 4-momenta). The 4-vectors (px, py, pz, E) are fully
    derivable from the features (which already contain px, py, pz) so
    reconstructing them separately would add no new learning signal.

    Args:
        backbone_dim: Channel dimension of backbone tokens (default: 256).
        decoder_dim: Internal dimension of the decoder (default: 128).
        num_heads: Number of attention heads (default: 4).
        num_decoder_layers: Number of stacked TransformerDecoderLayers
            (default: 1). More layers give queries more capacity to
            coordinate and refine predictions.
        num_output_features: Number of track features to reconstruct (default: 7).
        max_masked_tracks: Maximum number of masked tracks (vocab size for
            query embeddings). Must be >= mask_ratio × max_tracks_per_event.
            Default: 1200 (supports 0.4 × 2800 = 1120 masked tracks).
        dropout: Dropout rate in attention layers (default: 0.0).
    """

    def __init__(
        self,
        backbone_dim: int = 256,
        decoder_dim: int = 128,
        num_heads: int = 4,
        num_decoder_layers: int = 1,
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

        # LayerNorm on queries and memory before the decoder stack.
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

        # Stacked DETR-style decoder layers.
        # Each layer: self-attention → cross-attention → FFN
        # Pre-norm (norm_first=True) for training stability.
        # dim_feedforward = 4 × decoder_dim (standard transformer expansion).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Output MLP: decoded queries → predicted features.
        # One hidden layer with GELU provides enough capacity to decode
        # backbone representations into output features, without being
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
            num_masked_tracks, device=device,
        )  # (N_masked,)
        queries = self.query_norm(
            self.query_embeddings(query_indices),
        )  # (N_masked, decoder_dim)
        queries = queries.unsqueeze(0).expand(
            batch_size, -1, -1,
        )  # (B, N_masked, decoder_dim)

        # Stacked decoder layers: self-attention among queries (coordinate
        # predictions) + cross-attention to backbone tokens (event-specific
        # information). Each layer refines query representations.
        decoded = self.transformer_decoder(
            tgt=queries, memory=memory,
        )  # (B, N_masked, decoder_dim)

        # Output MLP: (B, N_masked, decoder_dim) → (B, N_masked, F)
        predictions = self.output_mlp(decoded)

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
        train_matcher: Matching algorithm for training assignment.
            'hungarian' — exact optimal (scipy CPU, ~50ms overhead).
            'sinkhorn' — approximate (GPU-native, non-bijective).
            Validation always uses exact Hungarian for honest metrics.
    """

    VALID_MATCHERS = ('hungarian', 'sinkhorn')

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        mask_ratio: float = 0.4,
        train_matcher: str = 'hungarian',
    ):
        super().__init__()

        if train_matcher not in self.VALID_MATCHERS:
            raise ValueError(
                f'train_matcher must be one of {self.VALID_MATCHERS}, '
                f'got {train_matcher!r}'
            )

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.mask_ratio = mask_ratio
        self.train_matcher = train_matcher

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

    def _hungarian_reconstruction_loss(
        self,
        predicted_features: torch.Tensor,
        true_features: torch.Tensor,
        validity_mask: torch.Tensor,
        num_valid_per_event: torch.Tensor,
        num_feature_channels: int,
    ) -> torch.Tensor:
        """Compute MSE reconstruction loss with Hungarian-matched assignment.

        The decoder outputs a SET of predicted tracks with no inherent ordering.
        Query embeddings are learnable slot differentiators, not tied to specific
        masked positions. We use the Hungarian algorithm to find the optimal
        1-to-1 assignment between predicted and ground-truth tracks that
        minimizes total MSE, then compute the loss on matched pairs.

        Cost matrix:
            C[i, j] = Σ_f (pred_f_i − true_f_j)²
        where i indexes predicted slots and j indexes ground-truth tracks.
        Hungarian finds permutation σ minimizing Σ_i C[i, σ(i)].

        The assignment is computed with no gradients (it returns indices).
        The MSE loss is computed on GPU with matched pairs, so gradients
        flow through the decoder predictions normally.

        Args:
            predicted_features: (B, F, K) predicted features per slot.
            true_features: (B, F, K) ground-truth features per track.
            validity_mask: (B, 1, K) boolean mask for valid (non-padded) slots.
            num_valid_per_event: (B,) count of valid tracks per event.
            num_feature_channels: F — number of feature channels.

        Returns:
            per_event_loss: (B,) MSE loss per event.
        """
        batch_size = predicted_features.shape[0]
        max_tracks = predicted_features.shape[2]
        device = predicted_features.device

        # Build pairwise cost matrix: C[b, i, j] = Σ_f (pred[b,f,i] − true[b,f,j])²
        # pred: (B, F, K) → (B, K, F) for easier pairwise computation
        predicted_transposed = predicted_features.transpose(1, 2)  # (B, K, F)
        true_transposed = true_features.transpose(1, 2)  # (B, K, F)

        # cost[b, i, j] = ||pred[b, i, :] - true[b, j, :]||²
        # = Σ_f (pred[b,i,f] − true[b,j,f])²
        # Using expansion: ||a-b||² = ||a||² + ||b||² - 2 a·b
        predicted_squared = (predicted_transposed ** 2).sum(dim=2)  # (B, K)
        true_squared = (true_transposed ** 2).sum(dim=2)  # (B, K)
        cross_term = torch.bmm(
            predicted_transposed, true_transposed.transpose(1, 2),
        )  # (B, K, K)
        cost_matrix = (
            predicted_squared.unsqueeze(2)
            + true_squared.unsqueeze(1)
            - 2 * cross_term
        )  # (B, K, K)

        # Mask out invalid slots: set cost to large value for padded positions
        # so Hungarian never matches them
        validity_flat = validity_mask.squeeze(1)  # (B, K)
        large_cost = 1e6
        # Invalid predicted slots (rows)
        cost_matrix = cost_matrix.masked_fill(
            ~validity_flat.unsqueeze(2), large_cost,
        )
        # Invalid true slots (columns)
        cost_matrix = cost_matrix.masked_fill(
            ~validity_flat.unsqueeze(1), large_cost,
        )

        # Optimal assignment: find best 1-to-1 matching (no gradients).
        # Validation always uses exact Hungarian for honest metrics.
        # Training matcher is configurable via self.train_matcher:
        #   'hungarian' — exact optimal (scipy CPU, bijective)
        #   'sinkhorn'  — approximate (GPU-native, non-bijective)
        # indices: (B, 2, K) — indices[:, 0] = predicted slot, indices[:, 1] = true slot
        if not self.training or self.train_matcher == 'hungarian':
            indices = hungarian_matcher(cost_matrix.detach())
        else:
            indices = sinkhorn_matcher(
                cost_matrix.detach(), deduplicate=False,
            )
        matched_pred_indices = indices[:, 0, :]  # (B, K)
        matched_true_indices = indices[:, 1, :]  # (B, K)

        # Gather matched predictions and targets using the optimal assignment
        # Expand indices for feature gathering: (B, K) → (B, F, K)
        pred_gather = matched_pred_indices.unsqueeze(1).expand(
            -1, num_feature_channels, -1,
        )
        true_gather = matched_true_indices.unsqueeze(1).expand(
            -1, num_feature_channels, -1,
        )

        matched_predictions = predicted_features.gather(2, pred_gather)  # (B, F, K)
        matched_targets = true_features.gather(2, true_gather)  # (B, F, K)

        # Build validity mask for matched pairs
        matched_validity = validity_flat.gather(
            1, matched_pred_indices,
        ).unsqueeze(1)  # (B, 1, K)

        # L = (1 / N_valid) × Σ_i Σ_f (matched_pred_f_i − matched_true_f_i)²
        feature_error = (
            (matched_predictions - matched_targets).square()
            * matched_validity.float()
        )  # (B, F, K)
        per_event_loss = feature_error.sum(dim=(1, 2)) / (
            num_valid_per_event.float() * num_feature_channels
        ).clamp(min=1.0)  # (B,)

        return per_event_loss

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

        # Step 7: Hungarian-matched feature reconstruction loss per event
        #
        # The decoder outputs a SET of predicted tracks with no inherent
        # ordering — query embeddings are learnable slot differentiators,
        # not tied to specific masked positions. Therefore we use the
        # Hungarian algorithm to find the optimal 1-to-1 assignment between
        # predicted and ground-truth tracks that minimizes total MSE.
        #
        # Cost matrix: C[i, j] = Σ_f (pred_f_i − true_f_j)²
        #   where i indexes predicted slots and j indexes ground-truth tracks.
        # Hungarian finds permutation σ minimizing Σ_i C[i, σ(i)].
        #
        # The assignment is computed with no gradients (returns indices only).
        # The MSE loss is computed on GPU with matched pairs, so gradients
        # flow back through the decoder predictions normally.
        #
        # Features are already standardized (clipped to [-5, 5]) by weaver,
        # so MSE is well-scaled across all feature channels.
        per_event_loss = self._hungarian_reconstruction_loss(
            predicted_features, masked_true_features,
            masked_validity, num_masked_per_event, num_features,
        )  # (B,)

        return per_event_loss
