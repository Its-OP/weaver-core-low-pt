"""Parallel Identity + Context Backbone for per-track classification.

Two independent streams process raw track features:
    - Identity stream: MLP projection preserving per-track discriminants
      (dxy_significance, charge, pT, etc.) without neighbor contamination.
    - Context stream: Lightweight EdgeConv layers providing neighborhood
      awareness via kNN graph with physics-informed edge features.

The outputs are concatenated, giving downstream layers (GAPLayers, scoring
head) access to both the track's own signature and its local environment.

This replaces the frozen EnrichCompactBackbone which was pretrained for
masked track reconstruction — a task that overwrites per-track identity
with neighbor-averaged context, diluting signal-specific information.

Architecture:
    Raw features (B, input_dim, P)
      ├── Identity: BN → Conv1d(input_dim → identity_dim) → BN → ReLU
      │             Preserves per-track features in higher-dim space
      └── Context:  BN → Conv1d(input_dim → node_dim) → N × EdgeConv(k, pairwise LV)
                    Aggregates neighbor information via message passing
    Output: concat(identity, context) → (B, identity_dim + context_dim, P)

The context stream uses the same MultiScaleEdgeConv and pairwise
Lorentz-vector edge features (ln kT, ln z, ln ΔR, ln m²) as the original
backbone, but is lighter (1-2 layers, k=16) and trained from scratch for
classification.
"""
import torch
import torch.nn as nn
from functools import partial

from weaver.nn.model.ParticleNeXt import (
    MultiScaleEdgeConv,
    knn,
    get_graph_feature,
)


class ParallelBackbone(nn.Module):
    """Parallel identity + context backbone.

    Args:
        input_dim: Number of raw input features per track (default: 7).
        identity_dim: Output dimension of identity stream (default: 64).
        context_dim: Output dimension of context stream (default: 128).
        num_context_layers: Number of EdgeConv layers in context stream (default: 2).
        context_num_neighbors: kNN K for context stream (default: 16).
        context_edge_dim: Encoded pairwise LV edge feature dimension (default: 8).
        context_node_dim: Initial node embedding dim in context stream (default: 32).
        context_message_dim: Message dimension within MultiScaleEdgeConv (default: 64).
        context_reduction_dilation: Multi-scale dilation config (default: [(4,1),(2,1)]).
        context_edge_aggregation: Neighbor aggregation mode (default: 'attn8').
    """

    def __init__(
        self,
        input_dim: int = 7,
        identity_dim: int = 64,
        context_dim: int = 128,
        num_context_layers: int = 2,
        context_num_neighbors: int = 16,
        context_edge_dim: int = 8,
        context_node_dim: int = 32,
        context_message_dim: int = 64,
        context_reduction_dilation: list | None = None,
        context_edge_aggregation: str = 'attn8',
    ):
        super().__init__()

        if context_reduction_dilation is None:
            context_reduction_dilation = [(4, 1), (2, 1)]

        self.output_dim = identity_dim + context_dim

        # ---- Identity stream ----
        # Simple MLP that projects raw features into a higher-dim space
        # without any neighbor mixing. Preserves per-track discriminants.
        self.identity_projection = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Conv1d(input_dim, identity_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(identity_dim),
            nn.ReLU(),
        )

        # ---- Context stream ----
        # Node encoder: raw features → context_node_dim
        # Uses Conv2d because MultiScaleEdgeConv expects (B, C, P, 1)
        self.context_node_encode = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, context_node_dim, kernel_size=1, bias=False),
        )

        # Edge encoder: pairwise Lorentz-vector features (4 channels) → edge_dim
        # 4 channels = ln kT, ln z, ln ΔR, ln m² from pairwise_lv_fts()
        pairwise_lv_feature_dim = 4
        self.context_edge_encode = nn.Sequential(
            nn.BatchNorm2d(pairwise_lv_feature_dim),
            nn.Conv2d(
                pairwise_lv_feature_dim, context_edge_dim,
                kernel_size=1, bias=False,
            ),
        )

        # EdgeConv layers
        self.context_layers = nn.ModuleList()
        current_dim = context_node_dim

        for layer_index in range(num_context_layers):
            # Last layer outputs context_dim; intermediate layers output context_dim too
            out_dim = context_dim
            self.context_layers.append(
                MultiScaleEdgeConv(
                    node_dim=current_dim,
                    edge_dim=context_edge_dim,
                    num_neighbors=context_num_neighbors,
                    out_dim=out_dim,
                    reduction_dilation=context_reduction_dilation,
                    message_dim=context_message_dim,
                    edge_aggregation=context_edge_aggregation,
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
            current_dim = out_dim

        # kNN and graph feature functions
        self.context_num_neighbors = context_num_neighbors
        self.context_knn = partial(knn, k=context_num_neighbors)
        self.context_get_graph_feature = partial(
            get_graph_feature,
            k=context_num_neighbors,
            use_rel_fts=False,
            use_rel_coords=False,
            use_rel_dist=False,
            use_rel_lv_fts=True,
            use_polarization_angle=False,
        )

        # Tell each layer the shared graph uses K neighbors
        for layer in self.context_layers:
            layer.num_neighbors_in = context_num_neighbors

        # Post-context normalization
        self.context_post = nn.Sequential(
            nn.BatchNorm2d(current_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: parallel identity + context streams.

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) raw per-track features.
            lorentz_vectors: (B, 4, P) raw 4-vectors (px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            output: (B, identity_dim + context_dim, P) concatenated features.
                Padded positions are zeroed.
        """
        boolean_mask = mask.bool()
        null_positions = ~boolean_mask
        mask_float = mask.float()

        # ---- Identity stream ----
        # MLP projection without any neighbor information
        identity_output = self.identity_projection(features)  # (B, identity_dim, P)
        identity_output = identity_output * mask_float

        # ---- Context stream ----
        # Static graph computation (no gradients through kNN/pairwise_lv_fts)
        with torch.no_grad():
            # Push padded points far away so kNN never selects them
            points_for_knn = points.clone().float()
            points_for_knn.masked_fill_(null_positions, 1e9)
            if self.training:
                random_shift = torch.rand_like(points_for_knn)
                random_shift.masked_fill_(boolean_mask, 0)
                points_for_knn = points_for_knn + 1e6 * random_shift

            # kNN indices: (B, P, K)
            knn_indices = self.context_knn(points_for_knn)

            # Pairwise LV features: (B, 4, P, K) = [ln kT, ln z, ln ΔR, ln m²]
            edge_inputs, _, lvs_neighbors, null_edge_positions = (
                self.context_get_graph_feature(
                    lvs=lorentz_vectors,
                    mask=boolean_mask,
                    edges=None,
                    idx=knn_indices,
                    null_edge_pos=None,
                )
            )

        # Feature encoding (with gradients)
        features_4d = features.unsqueeze(-1)  # (B, C, P, 1)
        encoded_nodes = self.context_node_encode(features_4d)
        encoded_edges = self.context_edge_encode(edge_inputs)

        # EdgeConv layers
        current_features = encoded_nodes
        for layer in self.context_layers:
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

        # Post-context: BN + ReLU, squeeze trailing dim
        context_output = self.context_post(current_features).squeeze(-1)
        context_output = context_output * mask_float  # (B, context_dim, P)

        # ---- Concatenate streams ----
        output = torch.cat([identity_output, context_output], dim=1)

        return output
