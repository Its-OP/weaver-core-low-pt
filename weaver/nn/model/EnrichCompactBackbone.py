"""Two-stage Enrich-Compact Backbone for per-track tau classification.

Stage 1 (Enrichment): ParticleNeXt-style MultiScaleEdgeConv layers enrich
ALL tracks with neighbor information via static kNN graph. No downsampling.
Track identity and count are preserved.

Stage 2 (Compaction): PointNet++-style set abstraction progressively
downsamples enriched tracks into dense tokens via FPS + kNN + MLP + max-pool.
No edge features needed — tracks are already contextually enriched.

Masking happens BETWEEN the two stages in the pretrainer:
    1. All tracks participate in enrichment
    2. Masked tracks are hidden
    3. Only visible enriched tracks enter compaction
    4. Decoder reconstructs original 7 raw features

Architecture:
    Enrichment: 7 raw features → BN+Conv2d(7→32) → 3× MultiScaleEdgeConv → 256-dim
    Compaction: ~700 visible × 256-dim → FPS+kNN+MLP+pool → 256 → 128 tokens × 256-dim

The backbone exposes enrich() and compact() as separate methods so that
MaskedTrackPretrainer can insert masking between them.
"""
import torch
import torch.nn as nn
from functools import partial

from weaver.nn.model.ParticleNeXt import (
    MultiScaleEdgeConv,
    knn,
    get_graph_feature,
)
from weaver.nn.model.HierarchicalGraphBackbone import (
    farthest_point_sampling,
    cross_set_knn,
    cross_set_gather,
)


class CompactionStage(nn.Module):
    """PointNet++-style set abstraction for spatial compaction.

    Reduces point count: P_in → P_out via FPS + kNN + MLP + max-pool.
    No pairwise Lorentz-vector edge features — tracks are already enriched
    with neighbor context from the enrichment stage.

    Steps:
        1. FPS selects P_out centroids in (η, φ) space
        2. kNN finds K nearest input points per centroid
        3. Gather neighbor features → (B, C_in, P_out, K)
        4. MLP processes concat(center, neighbor) features
        5. Max-pool over K neighbors → (B, C_out, P_out)
        6. Residual: output = ReLU(pooled + shortcut(centroid_features))

    Uses concat(center, neighbor) rather than concat(center, neighbor − center)
    because enriched tracks already carry full neighborhood context — absolute
    features are more informative than differences between contextualized
    representations.

    No self-match exclusion in kNN: since we do not compute pairwise_lv_fts(),
    ΔR=0 self-matches are harmless. The centroid appearing as its own neighbor
    provides a natural self-loop in the max-pool aggregation.

    Args:
        input_channels: C_in — feature channels of input points.
        output_channels: C_out — feature channels of output centroids.
        num_output_points: P_out — number of centroids after downsampling.
        num_neighbors: K — neighbors per centroid for kNN.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_output_points: int,
        num_neighbors: int,
    ):
        super().__init__()
        self.num_output_points = num_output_points
        self.num_neighbors = num_neighbors

        # MLP input: concat(center_features, neighbor_features) = 2 × C_in
        mlp_input_dim = 2 * input_channels

        # MLP: two Conv2d layers operating on (B, C, P_out, K)
        # No ReLU after last BN — applied after residual addition
        # (post-addition activation, following ResNet / ParticleNet)
        self.aggregation_mlp = nn.Sequential(
            nn.Conv2d(mlp_input_dim, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        # Residual shortcut: projects centroid features from C_in → C_out
        # Conv1d(kernel=1) + BN matches ParticleNet's shortcut design
        self.residual_shortcut = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        # Post-residual activation
        self.activation = nn.ReLU()

    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: downsample P_in → P_out.

        Args:
            coordinates: (B, 2, P_in) in (η, φ).
            features: (B, C_in, P_in) per-point enriched features.
            mask: (B, 1, P_in) boolean mask, True for valid.

        Returns:
            Tuple of:
                output_features: (B, C_out, P_out)
                output_coordinates: (B, 2, P_out)
                output_mask: (B, 1, P_out) — all True (FPS only selects valid)
        """
        batch_size = coordinates.shape[0]

        # Step 1: Farthest Point Sampling → select P_out centroid indices
        centroid_indices = farthest_point_sampling(
            coordinates, mask, self.num_output_points
        )  # (B, P_out)

        # Gather centroid data using the selected indices
        idx_expanded = centroid_indices.unsqueeze(1)  # (B, 1, P_out)
        centroid_coordinates = coordinates.gather(
            2, idx_expanded.expand(-1, 2, -1)
        )  # (B, 2, P_out)
        centroid_features = features.gather(
            2, idx_expanded.expand(-1, features.shape[1], -1)
        )  # (B, C_in, P_out)

        # Step 2: Cross-set kNN — find K nearest reference points per centroid
        # No self-match exclusion (query_reference_indices=None) because we
        # do not compute pairwise_lv_fts, so ΔR=0 is harmless.
        neighbor_indices = cross_set_knn(
            centroid_coordinates, coordinates, self.num_neighbors, mask,
            query_reference_indices=None,
        )  # (B, P_out, K)

        # Step 3: Gather neighbor features
        neighbor_features = cross_set_gather(
            features, neighbor_indices
        )  # (B, C_in, P_out, K)

        # Step 4: Build MLP input = concat(center_expanded, neighbor_features)
        center_expanded = centroid_features.unsqueeze(-1).expand_as(
            neighbor_features
        )  # (B, C_in, P_out, K)
        mlp_input = torch.cat(
            [center_expanded, neighbor_features], dim=1
        )  # (B, 2*C_in, P_out, K)

        # Step 5: MLP → messages (B, C_out, P_out, K)
        messages = self.aggregation_mlp(mlp_input)

        # Step 6: Max-pool aggregation over K neighbors
        # Mask invalid neighbors before max-pool: set to -inf so they can't win
        neighbor_mask = cross_set_gather(
            mask.float(), neighbor_indices
        )  # (B, 1, P_out, K)
        messages_masked = messages.masked_fill(neighbor_mask == 0, float('-inf'))
        pooled_features = messages_masked.max(dim=-1)[0]  # (B, C_out, P_out)
        # Handle all-masked case: replace -inf with 0
        pooled_features = pooled_features.nan_to_num(0.0)
        pooled_features = pooled_features.masked_fill(
            pooled_features == float('-inf'), 0.0
        )

        # Step 7: Residual connection
        # shortcut(centroid_features): (B, C_in, P_out) → (B, C_out, P_out)
        # output = ReLU(pooled + shortcut)
        shortcut = self.residual_shortcut(centroid_features)
        output_features = self.activation(pooled_features + shortcut)

        # Output mask: all centroids are valid (FPS only selects valid points)
        output_mask = torch.ones(
            batch_size, 1, self.num_output_points,
            device=features.device, dtype=mask.dtype,
        )

        return output_features, centroid_coordinates, output_mask


class EnrichCompactBackbone(nn.Module):
    """Two-stage backbone: enrichment (ParticleNeXt) + compaction (PointNet++).

    Stage 1 (enrichment) processes ALL tracks with ParticleNeXt's
    MultiScaleEdgeConv layers to build contextually-rich per-track features.
    Static kNN graph computed once and shared across all enrichment layers.

    Stage 2 (compaction) downsamples visible (unmasked) enriched tracks into
    dense tokens via PointNet++-style set abstraction (FPS + kNN + MLP +
    max-pool). No further message passing — tracks are already enriched.

    The backbone exposes enrich() and compact() as separate methods so that
    MaskedTrackPretrainer can insert masking between them.

    Args:
        input_dim: Number of raw input features per track (default: 7).
        enrichment_kwargs: Configuration for the enrichment stage.
            node_dim: Initial node embedding dimension (default: 32).
            edge_dim: Encoded pairwise LV feature dimension (default: 8).
            num_neighbors: k for static kNN graph (default: 32).
            edge_aggregation: Neighbor aggregation mode (default: 'attn8').
            layer_params: List of (k, out_dim, reduction_dilation, message_dim)
                tuples for MultiScaleEdgeConv layers.
        compaction_kwargs: Configuration for the compaction stage.
            stage_output_points: List of output point counts per stage.
            stage_output_channels: List of channel widths per stage.
            stage_num_neighbors: List of kNN K per stage.
    """

    def __init__(
        self,
        input_dim: int = 7,
        enrichment_kwargs: dict | None = None,
        compaction_kwargs: dict | None = None,
    ):
        super().__init__()

        if enrichment_kwargs is None:
            enrichment_kwargs = {}
        if compaction_kwargs is None:
            compaction_kwargs = {}

        # ---- Enrichment stage configuration ----
        node_dim = enrichment_kwargs.get('node_dim', 32)
        edge_dim = enrichment_kwargs.get('edge_dim', 8)
        num_neighbors = enrichment_kwargs.get('num_neighbors', 32)
        edge_aggregation = enrichment_kwargs.get('edge_aggregation', 'attn8')
        layer_params = enrichment_kwargs.get('layer_params', [
            (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
            (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
            (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
        ])

        # Node encoder: BN2d + Conv2d(input_dim → node_dim)
        # Uses Conv2d because MultiScaleEdgeConv operates on (B, C, P, 1) tensors
        self.node_encode = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, node_dim, kernel_size=1, bias=False),
        )

        # Edge encoder: pairwise Lorentz-vector features (4 channels) → edge_dim
        # 4 channels = ln kT, ln z, ln ΔR, ln m² from pairwise_lv_fts()
        pairwise_lv_feature_dim = 4
        self.edge_encode = nn.Sequential(
            nn.BatchNorm2d(pairwise_lv_feature_dim),
            nn.Conv2d(pairwise_lv_feature_dim, edge_dim, kernel_size=1, bias=False),
        )

        # Build MultiScaleEdgeConv enrichment layers
        self.enrichment_layers = nn.ModuleList()
        current_dim = node_dim
        enrichment_neighbor_counts = []

        for param in layer_params:
            layer_k, out_dim, reduction_dilation, message_dim = param
            self.enrichment_layers.append(
                MultiScaleEdgeConv(
                    node_dim=current_dim,
                    edge_dim=edge_dim,
                    num_neighbors=layer_k,
                    out_dim=out_dim,
                    reduction_dilation=reduction_dilation,
                    message_dim=message_dim,
                    edge_aggregation=edge_aggregation,
                    use_rel_lv_fts=True,
                    use_rel_fts=False,
                    use_rel_dist=False,
                    update_coords=False,
                    lv_aggregation=False,
                    use_node_se=True,
                    use_edge_se=True,
                    init_scale=1e-5,
                )
            )
            enrichment_neighbor_counts.append(layer_k)
            current_dim = out_dim

        # Static graph: all enrichment layers share one kNN (computed once)
        self.enrichment_num_neighbors = max(enrichment_neighbor_counts)
        self.enrichment_knn = partial(knn, k=self.enrichment_num_neighbors)
        self.enrichment_get_graph_feature = partial(
            get_graph_feature,
            k=self.enrichment_num_neighbors,
            use_rel_fts=False,
            use_rel_coords=False,
            use_rel_dist=False,
            use_rel_lv_fts=True,
            use_polarization_angle=False,
        )
        # Tell each layer the shared graph uses max K neighbors
        for layer in self.enrichment_layers:
            layer.num_neighbors_in = self.enrichment_num_neighbors

        # Post-enrichment normalization (matches ParticleNeXt.post)
        self.enrichment_post = nn.Sequential(
            nn.BatchNorm2d(current_dim),
            nn.ReLU(),
        )

        self.enrichment_output_dim = current_dim

        # ---- Compaction stage configuration ----
        compaction_output_points = compaction_kwargs.get(
            'stage_output_points', [256, 128]
        )
        compaction_output_channels = compaction_kwargs.get(
            'stage_output_channels', [256, 256]
        )
        compaction_num_neighbors = compaction_kwargs.get(
            'stage_num_neighbors', [16, 16]
        )

        num_compaction_stages = len(compaction_output_points)
        assert len(compaction_output_channels) == num_compaction_stages
        assert len(compaction_num_neighbors) == num_compaction_stages

        self.compaction_stages = nn.ModuleList()
        compaction_input_dim = current_dim

        for stage_index in range(num_compaction_stages):
            self.compaction_stages.append(
                CompactionStage(
                    input_channels=compaction_input_dim,
                    output_channels=compaction_output_channels[stage_index],
                    num_output_points=compaction_output_points[stage_index],
                    num_neighbors=compaction_num_neighbors[stage_index],
                )
            )
            compaction_input_dim = compaction_output_channels[stage_index]

        self.output_dim = compaction_output_channels[-1]

    def enrich(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 1: Enrich ALL tracks with neighbor context.

        Computes static kNN graph once under no_grad(), then passes through
        all MultiScaleEdgeConv layers with gradients. All tracks participate
        — no masking at this stage.

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) raw per-track features.
            lorentz_vectors: (B, 4, P) raw 4-vectors (px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            enriched_features: (B, enrichment_output_dim, P)
                Enriched per-track features. Padded positions are zeroed.
        """
        boolean_mask = mask.bool()
        null_positions = ~boolean_mask

        # ---- Static graph computation (no gradients) ----
        # kNN indices and pairwise LV features are computed once and shared
        # across all enrichment layers. Computed under no_grad() because:
        #   1. pairwise_lv_fts() has NaN gradients from sqrt(ΔR²) near zero
        #   2. kNN topology should be fixed (not learned) for stability
        with torch.no_grad():
            # Push padded points far away so kNN never selects them.
            # Force fp32 to avoid overflow if AMP casts inputs to fp16
            # (fp16 max ≈ 65504, so 1e9 would overflow).
            points_for_knn = points.clone().float()
            points_for_knn.masked_fill_(null_positions, 1e9)
            # During training, add random shift to padded points to break
            # ties (prevents numerical coincidences among padded points)
            if self.training:
                random_shift = torch.rand_like(points_for_knn)
                random_shift.masked_fill_(boolean_mask, 0)
                points_for_knn = points_for_knn + 1e6 * random_shift

            # Compute kNN indices: (B, P, K)
            knn_indices = self.enrichment_knn(points_for_knn)

            # Compute pairwise LV features and null edge mask
            # get_graph_feature with lvs only returns:
            #   edge_inputs: (B, 4, P, K) — [ln kT, ln z, ln ΔR, ln m²]
            #   lvs_neighbors: (B, 4, P, K) — raw neighbor 4-vectors
            #   null_edge_positions: (B, 1, P, K) — True where edge is invalid
            edge_inputs, _, lvs_neighbors, null_edge_positions = (
                self.enrichment_get_graph_feature(
                    lvs=lorentz_vectors,
                    mask=boolean_mask,
                    edges=None,
                    idx=knn_indices,
                    null_edge_pos=None,
                )
            )

        # ---- Feature encoding (with gradients) ----
        # Add trailing dimension for Conv2d: (B, C, P) → (B, C, P, 1)
        features_4d = features.unsqueeze(-1)
        encoded_features = self.node_encode(features_4d)  # (B, node_dim, P, 1)

        # Encode pairwise LV edge features
        encoded_edges = self.edge_encode(edge_inputs)  # (B, edge_dim, P, K)

        # ---- MultiScaleEdgeConv enrichment layers ----
        current_features = encoded_features
        for layer in self.enrichment_layers:
            _, current_features = layer(
                points=points_for_knn,
                features=current_features,
                lorentz_vectors=lorentz_vectors,
                mask=boolean_mask,
                edges=None,
                idx=knn_indices,
                null_edge_pos=null_edge_positions,
                edge_inputs=encoded_edges,
                lvs_ngbs=lvs_neighbors,
            )
            # current_features: (B, out_dim, P, 1)

        # Post-enrichment: BN + ReLU, then squeeze trailing dimension
        enriched = self.enrichment_post(current_features).squeeze(-1)
        # (B, enrichment_output_dim, P)

        # Zero out padded positions to prevent information leakage
        enriched = enriched * mask.float()

        return enriched

    def compact(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 2: Compact enriched tracks into dense tokens.

        Progressively downsamples through CompactionStages using FPS + kNN +
        MLP + max-pool. Operates only on visible (unmasked) tracks — the
        pretrainer densely packs visible tracks before calling this.

        Args:
            coordinates: (B, 2, P_visible) coordinates of visible tracks.
            features: (B, C_enriched, P_visible) enriched features.
            mask: (B, 1, P_visible) boolean mask for visible tracks.

        Returns:
            Tuple of:
                tokens: (B, output_dim, M) dense token features.
                token_coordinates: (B, 2, M) token positions in (η, φ).
        """
        current_features = features
        current_coordinates = coordinates
        current_mask = mask

        for stage in self.compaction_stages:
            current_features, current_coordinates, current_mask = stage(
                current_coordinates, current_features, current_mask
            )

        return current_features, current_coordinates
