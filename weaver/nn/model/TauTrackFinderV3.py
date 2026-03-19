"""Tau-origin pion track finder V3: ABCNet-inspired GAPLayer architecture.

Replaces V2's max-pool kNN + self-attention refinement with:
    - GAPLayer (Graph Attention Pooling): attention-weighted edge convolution
      on kNN graphs, following ABCNet (Mikuni & Canelli, EPJ Plus 2020).
    - Dual kNN: first in physical (eta, phi) space, second in learned feature
      space. The feature-space kNN allows the model to find similar tracks
      that may be far apart in (eta, phi).
    - Global context injection: event-level average pooling provides each
      track with awareness of the overall event topology.
    - Multi-scale feature concatenation: all intermediate features are
      concatenated (GAP1 + GAP2 + backbone + raw + global), matching
      ABCNet's aggregation design.

Architecture:
    1. Pretrained EnrichCompactBackbone (frozen, enrichment only) -> (B, 256, P)
    2. GAPLayer 1: kNN in (eta, phi) -> attention-weighted edge features
    3. Intermediate MLPs
    4. GAPLayer 2: kNN in learned feature space
    5. Intermediate MLPs
    6. Global context: masked average pool -> project -> tile
    7. Concatenate all features + skip-connected raw features + Lorentz vectors
    8. Per-track scoring MLP -> per_track_logits (B, P)

Loss:
    - Asymmetric Loss (ASL) on ALL ~1130 tracks — zeros easy negative gradients.
    - Pairwise ranking loss — directly optimizes recall@K.

GAPLayer attention mechanism (ABCNet, Eq. 1-2):
    x'_i = h(x_i, theta_i, F)          -- node transform
    y'_ij = h(y_ij, theta_ij, F)        -- edge transform (y_ij = x_j - x_i)
    c_ij = LeakyReLU(h(x'_i, 1) + h(y'_ij, 1))  -- attention logits
    c_ij = softmax_j(c_ij)              -- normalize over neighbors
    hat_x_i = ReLU(sum_j c_ij * y'_ij)  -- attention-weighted aggregation

References:
    ABCNet: Mikuni & Canelli, EPJ Plus 135 (2020) 463
    GAPNet: Can et al., arXiv:1905.08705
    Focal loss: Lin et al., ICCV 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.HierarchicalGraphBackbone import cross_set_knn, cross_set_gather
from weaver.nn.model.ParallelBackbone import ParallelBackbone


class GAPLayer(nn.Module):
    """Graph Attention Pooling Layer (ABCNet-style).

    Computes attention-weighted aggregation of edge features over kNN
    neighbors. Each head independently learns attention coefficients
    and the outputs are combined by taking the element-wise maximum.

    Mathematical formulation (per head h):
        x'_i = Conv1d(x_i)                           -- node encoding
        y_ij = x_j - x_i                             -- edge features (differences)
        y'_ij = Conv2d(y_ij)                          -- edge encoding
        self_coef_i = Conv2d(x'_i, output=1)          -- self-attention score
        neighbor_coef_ij = Conv2d(y'_ij, output=1)    -- neighbor attention score
        c_ij = softmax_j(LeakyReLU(self_coef_i + neighbor_coef_ij))
        hat_x_i^h = ReLU(sum_j c_ij * y'_ij)

    Multi-head combination:
        hat_x_i = max_h(hat_x_i^h)                   -- element-wise max

    Graph features (per-node max of encoded edge features):
        graph_i = max_j(y'_ij)

    Args:
        input_dim: Dimension of input features per track.
        encoding_dim: Output dimension of each head (F in ABCNet).
        num_neighbors: Number of kNN neighbors (K).
        num_heads: Number of parallel attention heads (H).
            Ignored when use_mia=True (MIA uses encoding_dim heads internally).
        use_mia: If True, use More-Interaction Attention (MIParT, Wu & Wang 2024).
            MIA replaces Q/K-based attention with high-dimensional pairwise
            embeddings: each of encoding_dim channels acts as an independent
            attention head (head_dim=1), element-wise scaling edge features.
            Only a Value projection is learned per layer; attention pattern is
            purely determined by pairwise geometry.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        num_neighbors: int,
        num_heads: int = 1,
        use_mia: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads
        self.use_mia = use_mia

        if use_mia:
            # ---- MIA mode (More-Interaction Attention) ----
            # Edge features → encoding_dim-channel attention weights via MLP.
            # Each channel = 1 attention head with head_dim=1.
            # No Q/K projections — attention purely from pairwise geometry.
            # Only a Value projection is learned.
            #
            # MIA attention: softmax_j(MLP(y_ij)) ∈ (B, encoding_dim, P, K)
            # Output: sum_j attention_ij * value_j
            self.mia_edge_embed = nn.Sequential(
                nn.Conv2d(input_dim, encoding_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(encoding_dim),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(encoding_dim, encoding_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(encoding_dim),
            )

            # Value projection: input features → encoding_dim values
            self.mia_value_projection = nn.Sequential(
                nn.Conv1d(input_dim, encoding_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(encoding_dim),
            )
        else:
            # ---- Standard ABCNet-style attention ----
            # Per-head parameter modules
            self.node_encoders = nn.ModuleList()
            self.edge_encoders = nn.ModuleList()
            self.self_attention_scorers = nn.ModuleList()
            self.neighbor_attention_scorers = nn.ModuleList()

            for _ in range(num_heads):
                # Node encoder: x_i -> x'_i of dimension encoding_dim
                self.node_encoders.append(nn.Sequential(
                    nn.Conv1d(input_dim, encoding_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(encoding_dim),
                ))

                # Edge encoder: y_ij -> y'_ij of dimension encoding_dim
                self.edge_encoders.append(nn.Sequential(
                    nn.Conv2d(input_dim, encoding_dim, kernel_size=1, bias=True),
                    nn.BatchNorm2d(encoding_dim),
                ))

                # Self-attention scorer: x'_i -> scalar
                self.self_attention_scorers.append(
                    nn.Conv1d(encoding_dim, 1, kernel_size=1, bias=True),
                )

                # Neighbor attention scorer: y'_ij -> scalar
                self.neighbor_attention_scorers.append(
                    nn.Conv2d(encoding_dim, 1, kernel_size=1, bias=True),
                )

    def compute_mia_attention_weights(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MIA high-dimensional attention weights and intermediates.

        Returns a tuple of:
            attention_weights: (B, encoding_dim, P, K) softmax attention weights
                where each of encoding_dim channels is an independent attention head.
            encoded_edges: (B, encoding_dim, P, K) edge embeddings from mia_edge_embed,
                reusable for graph feature max-pool without recomputation.
            neighbor_validity: (B, 1, P, K) mask indicating valid neighbors,
                reusable for masking in downstream aggregation.
        """
        # Gather neighbor features and compute edge features
        neighbor_features = cross_set_gather(features, neighbor_indices)
        center_expanded = features.unsqueeze(-1).expand_as(neighbor_features)
        edge_features = neighbor_features - center_expanded  # (B, input_dim, P, K)

        # MLP → (B, encoding_dim, P, K) encoded edge embeddings
        encoded_edges = self.mia_edge_embed(edge_features)

        # Mask invalid neighbors
        neighbor_validity = cross_set_gather(
            mask.float(), neighbor_indices,
        )  # (B, 1, P, K)
        attention_logits = encoded_edges.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )

        # Softmax over K independently per channel.
        # Force float32 to avoid precision loss under AMP autocast.
        with torch.amp.autocast('cuda', enabled=False):
            attention_weights = functional.softmax(
                attention_logits.float(), dim=-1,
            )
        attention_weights = attention_weights.nan_to_num(0.0)

        return attention_weights, encoded_edges, neighbor_validity

    def _forward_mia(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MIA forward: high-dimensional attention-weighted aggregation.

        Attention pattern determined purely by pairwise edge geometry.
        Only Value projection is learned per layer.

        output_i = sum_j softmax_j(MLP(f_j - f_i))[c] * V(f_j)[c]
        for each channel c independently (encoding_dim heads, head_dim=1).

        Reuses encoded_edges and neighbor_validity from attention weight
        computation to avoid redundant cross_set_gather and mia_edge_embed calls.
        """
        # Attention weights + intermediates from edge geometry.
        # encoded_edges: (B, encoding_dim, P, K) — reused for graph features.
        # neighbor_validity: (B, 1, P, K) — reused for masking graph features.
        attention_weights, encoded_edges, neighbor_validity = (
            self.compute_mia_attention_weights(
                features, neighbor_indices, mask,
            )
        )

        # Value projection: (B, input_dim, P) → (B, encoding_dim, P)
        values = self.mia_value_projection(features)

        # Gather neighbor values: (B, encoding_dim, P, K)
        neighbor_values = cross_set_gather(values, neighbor_indices)

        # Attention-weighted aggregation (element-wise per channel)
        # attention_weights: (B, encoding_dim, P, K)
        # neighbor_values: (B, encoding_dim, P, K)
        attention_output = (attention_weights * neighbor_values).sum(dim=-1)
        attention_output = functional.relu(attention_output)  # (B, encoding_dim, P)

        # Graph features: max-pool over encoded edges (reuse from attention computation)
        # Mask invalid neighbors before max-pool
        encoded_edges_masked = encoded_edges.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )
        graph_features = encoded_edges_masked.max(dim=-1)[0]
        graph_features = graph_features.masked_fill(
            graph_features == float('-inf'), 0.0,
        )

        # Zero padded
        mask_float = mask.float()
        attention_output = attention_output * mask_float
        graph_features = graph_features * mask_float

        return attention_output, graph_features

    def compute_attention_coefficients(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention coefficients for the first head only.

        Exposed for testing — returns (B, P, K) softmax attention weights.
        """
        return self._compute_head_attention(
            features, neighbor_indices, mask, head_index=0,
        )[0]

    def _compute_head_attention(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mask: torch.Tensor,
        head_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute attention-weighted output for a single head.

        Args:
            features: (B, input_dim, P) per-track features.
            neighbor_indices: (B, P, K) kNN indices.
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            attention_coefficients: (B, P, K) softmax attention weights.
            attention_output: (B, encoding_dim, P) attention-weighted features.
            graph_features: (B, encoding_dim, P) max-pooled edge features.
        """
        # ---- Node encoding ----
        # x'_i = Conv1d(x_i) -> (B, encoding_dim, P)
        encoded_nodes = self.node_encoders[head_index](features)

        # ---- Gather neighbor features and compute edge features ----
        # Gather: (B, input_dim, P) -> (B, input_dim, P, K)
        neighbor_features = cross_set_gather(features, neighbor_indices)

        # Edge features: y_ij = neighbor_j - center_i
        center_expanded = features.unsqueeze(-1).expand_as(neighbor_features)
        edge_features = neighbor_features - center_expanded  # (B, input_dim, P, K)

        # ---- Edge encoding ----
        # y'_ij = Conv2d(y_ij) -> (B, encoding_dim, P, K)
        encoded_edges = self.edge_encoders[head_index](edge_features)

        # ---- Attention coefficient computation ----
        # self_coef_i: (B, 1, P) -> unsqueeze to (B, 1, P, 1) for broadcast
        self_attention_score = self.self_attention_scorers[head_index](
            encoded_nodes,
        )  # (B, 1, P)
        self_attention_score = self_attention_score.unsqueeze(-1)  # (B, 1, P, 1)

        # neighbor_coef_ij: (B, 1, P, K)
        neighbor_attention_score = self.neighbor_attention_scorers[head_index](
            encoded_edges,
        )  # (B, 1, P, K)

        # c_ij = LeakyReLU(self + neighbor) -> (B, 1, P, K)
        attention_logits = functional.leaky_relu(
            self_attention_score + neighbor_attention_score,
            negative_slope=0.2,
        )  # (B, 1, P, K)

        # Mask invalid neighbors: set logits to -inf before softmax
        # Neighbor validity: check if neighbor points to a valid track
        neighbor_validity = cross_set_gather(
            mask.float(), neighbor_indices,
        )  # (B, 1, P, K)
        attention_logits = attention_logits.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )

        # Softmax over K neighbors: c_ij = softmax_j(logits_ij)
        # Force float32 to avoid precision loss under AMP autocast.
        with torch.amp.autocast('cuda', enabled=False):
            attention_coefficients = functional.softmax(
                attention_logits.float(), dim=-1,
            )  # (B, 1, P, K)

        # Handle all-masked case (NaN from softmax of all -inf)
        attention_coefficients = attention_coefficients.nan_to_num(0.0)

        # ---- Attention-weighted aggregation ----
        # hat_x_i = ReLU(sum_j c_ij * y'_ij)
        # attention_coefficients: (B, 1, P, K) * encoded_edges: (B, encoding_dim, P, K)
        weighted_edges = attention_coefficients * encoded_edges  # broadcast over encoding_dim
        attention_output = weighted_edges.sum(dim=-1)  # (B, encoding_dim, P)
        attention_output = functional.relu(attention_output)

        # ---- Graph features: max-pool over encoded edges ----
        # Mask invalid edges before max-pool
        encoded_edges_masked = encoded_edges.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )
        graph_features = encoded_edges_masked.max(dim=-1)[0]  # (B, encoding_dim, P)
        graph_features = graph_features.masked_fill(
            graph_features == float('-inf'), 0.0,
        )

        # Squeeze attention coefficients for return: (B, 1, P, K) -> (B, P, K)
        attention_coefficients_squeezed = attention_coefficients.squeeze(1)

        return attention_coefficients_squeezed, attention_output, graph_features

    def forward(
        self,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: multi-head attention-weighted edge convolution.

        Dispatches to MIA mode or standard ABCNet mode based on use_mia flag.

        Args:
            features: (B, input_dim, P) per-track features.
            neighbor_indices: (B, P, K) kNN indices.
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            attention_features: (B, encoding_dim, P)
            graph_features: (B, encoding_dim, P)
        """
        if self.use_mia:
            return self._forward_mia(features, neighbor_indices, mask)

        head_attention_outputs = []
        head_graph_features = []

        for head_index in range(self.num_heads):
            _, attention_output, graph_output = self._compute_head_attention(
                features, neighbor_indices, mask, head_index,
            )
            head_attention_outputs.append(attention_output)
            head_graph_features.append(graph_output)

        # Combine heads: element-wise max (ABCNet design)
        if self.num_heads == 1:
            attention_features = head_attention_outputs[0]
            graph_features = head_graph_features[0]
        else:
            # Stack: (num_heads, B, encoding_dim, P) -> max over dim 0
            attention_features = torch.stack(
                head_attention_outputs, dim=0,
            ).max(dim=0)[0]
            graph_features = torch.stack(
                head_graph_features, dim=0,
            ).max(dim=0)[0]

        # Zero out padded positions
        mask_float = mask.float()  # (B, 1, P)
        attention_features = attention_features * mask_float
        graph_features = graph_features * mask_float

        return attention_features, graph_features


class TauTrackFinderV3(nn.Module):
    """ABCNet-inspired tau track finder with dual kNN GAPLayers.

    Architecture:
        1. Frozen backbone enrichment -> (B, backbone_dim, P)
        2. GAPLayer 1 in physical (eta, phi) space
        3. Intermediate MLPs
        4. GAPLayer 2 in learned feature space
        5. Intermediate MLPs
        6. Global context injection (average pool + project + tile)
        7. Multi-scale concatenation (all features + raw + Lorentz + global)
        8. Per-track scoring head -> (B, P) logits

    No self-attention refinement stage (removed: does not help when
    particles in the receptive window originate from different sources).

    Args:
        backbone_mode: Which backbone to use. Options:
            'frozen' — Frozen pretrained EnrichCompactBackbone (default, V3 original).
            'parallel' — Trainable ParallelBackbone (identity + context streams).
        backbone_kwargs: Config for EnrichCompactBackbone (used when backbone_mode='frozen').
        parallel_backbone_kwargs: Config for ParallelBackbone (used when backbone_mode='parallel').
        gap1_encoding_dim: Encoding dimension for first GAPLayer.
        gap1_num_neighbors: kNN K for first GAPLayer (physical space).
        gap1_num_heads: Number of attention heads for first GAPLayer.
        gap2_encoding_dim: Encoding dimension for second GAPLayer.
        gap2_num_neighbors: kNN K for second GAPLayer (feature space).
        gap2_num_heads: Number of attention heads for second GAPLayer.
        intermediate_dim: Hidden dimension of intermediate MLPs.
        global_context_dim: Dimension of projected global context.
        scoring_dropout: Dropout rate in the scoring head.
        focal_alpha: Alpha for focal loss class weighting (default: 0.75).
        focal_gamma: Gamma for focal loss modulation (default: 2.0).
    """

    def __init__(
        self,
        backbone_mode: str = 'frozen',
        backbone_kwargs: dict | None = None,
        parallel_backbone_kwargs: dict | None = None,
        gap1_encoding_dim: int = 64,
        gap1_num_neighbors: int = 16,
        gap1_num_heads: int = 4,
        gap2_encoding_dim: int = 64,
        gap2_num_neighbors: int = 16,
        gap2_num_heads: int = 4,
        intermediate_dim: int = 128,
        global_context_dim: int = 32,
        scoring_dropout: float = 0.4,
        # ASL loss hyperparameters (Ben-Baruch et al., ICCV 2021)
        focal_gamma_positive: float = 1.0,
        focal_gamma_negative: float = 4.0,
        asl_clip: float = 0.05,
        # Ranking loss hyperparameters
        ranking_loss_weight: float = 0.1,
        ranking_num_samples: int = 10,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if parallel_backbone_kwargs is None:
            parallel_backbone_kwargs = {}

        self.backbone_mode = backbone_mode
        self.gap1_num_neighbors = gap1_num_neighbors
        self.gap2_num_neighbors = gap2_num_neighbors
        # ASL parameters
        self.focal_gamma_positive = focal_gamma_positive
        self.focal_gamma_negative = focal_gamma_negative
        self.asl_clip = asl_clip
        # Ranking loss parameters
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_num_samples = ranking_num_samples

        # ---- Backbone ----
        if backbone_mode == 'parallel':
            # Trainable parallel identity + context backbone
            self.backbone = ParallelBackbone(**parallel_backbone_kwargs)
            backbone_dim = self.backbone.output_dim
        else:
            # Frozen pretrained EnrichCompactBackbone (default)
            self.backbone = EnrichCompactBackbone(**backbone_kwargs)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            backbone_dim = self.backbone.enrichment_output_dim  # typically 256

        # ---- GAPLayer 1: kNN in physical (eta, phi) space ----
        self.gap_layer_physical = GAPLayer(
            input_dim=backbone_dim,
            encoding_dim=gap1_encoding_dim,
            num_neighbors=gap1_num_neighbors,
            num_heads=gap1_num_heads,
            use_mia=True,
        )

        # Intermediate MLPs after GAPLayer 1
        # Input: cat(attention_features, graph_features) = 2 * gap1_encoding_dim
        self.intermediate_mlp_1 = nn.Sequential(
            nn.Conv1d(
                2 * gap1_encoding_dim, intermediate_dim,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Conv1d(
                intermediate_dim, intermediate_dim,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
        )

        # ---- GAPLayer 2: kNN in learned feature space ----
        self.gap_layer_learned = GAPLayer(
            input_dim=intermediate_dim,
            encoding_dim=gap2_encoding_dim,
            num_neighbors=gap2_num_neighbors,
            num_heads=gap2_num_heads,
            use_mia=True,
        )

        # Intermediate MLPs after GAPLayer 2
        self.intermediate_mlp_2 = nn.Sequential(
            nn.Conv1d(
                2 * gap2_encoding_dim, intermediate_dim,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
            nn.Conv1d(
                intermediate_dim, intermediate_dim,
                kernel_size=1, bias=False,
            ),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(),
        )

        # ---- Global context ----
        # Average pool enriched features -> project to global_context_dim
        self.global_context_projection = nn.Sequential(
            nn.Linear(backbone_dim, global_context_dim),
            nn.ReLU(),
        )

        # ---- Skip-connected raw features normalization ----
        input_dim = backbone_kwargs.get('input_dim', 7)
        self.raw_feature_norm = nn.BatchNorm1d(input_dim)
        self.lorentz_vector_norm = nn.BatchNorm1d(4)

        # ---- Multi-scale concatenation dimension ----
        # GAP1_attention + GAP1_graph + GAP2_attention + GAP2_graph
        # + backbone_enriched + raw_features + lorentz_vectors + global_context
        self.combined_dim = (
            gap1_encoding_dim       # GAP1 attention features
            + gap1_encoding_dim     # GAP1 graph features (max-pooled edges)
            + gap2_encoding_dim     # GAP2 attention features
            + gap2_encoding_dim     # GAP2 graph features (max-pooled edges)
            + backbone_dim          # backbone enriched features
            + input_dim             # raw features (all 7, BN-normalized)
            + 4                     # Lorentz vectors (px, py, pz, E)
            + global_context_dim    # global context
        )

        # ---- Per-track scoring head ----
        self.per_track_head = nn.Sequential(
            nn.Conv1d(self.combined_dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(scoring_dropout),
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(scoring_dropout),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def _compute_physical_knn(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute kNN indices in physical (eta, phi) space.

        Uses cross_set_knn which handles phi wrapping.

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            mask: (B, 1, P) boolean mask.

        Returns:
            neighbor_indices: (B, P, K) kNN indices.
        """
        with torch.no_grad():
            return cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=self.gap1_num_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )

    def _compute_feature_space_knn(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Compute kNN indices in learned feature space using chunked distances.

        Avoids materializing the full (B, P, P) distance matrix by processing
        query tracks in chunks of `chunk_size`. Each chunk computes distances
        to ALL reference tracks and extracts top-K, then results are concatenated.

        Uses the expanded L2 distance formula:
            ||f_i - f_j||^2 = ||f_i||^2 + ||f_j||^2 - 2 * f_i^T f_j

        Pre-computes squared norms once, then each chunk only needs a
        (B, chunk, P) bmm instead of the full (B, P, P).

        Self-matches are excluded by setting diagonal distances to +inf,
        preventing a track from being its own neighbor.

        Args:
            features: (B, C, P) intermediate features.
            mask: (B, 1, P) boolean mask.
            chunk_size: Number of query tracks to process at once (default: 256).

        Returns:
            neighbor_indices: (B, P, K) kNN indices.
        """
        with torch.no_grad():
            # features: (B, C, P) -> transpose to (B, P, C) for distance computation
            features_transposed = features.transpose(1, 2)  # (B, P, C)
            batch_size, num_tracks, feature_dim = features_transposed.shape

            # Pre-compute squared norms: ||f_i||^2 for all tracks
            # norms_squared: (B, P, 1) — reused across all chunks
            norms_squared = features_transposed.pow(2).sum(
                dim=-1, keepdim=True,
            )  # (B, P, 1)

            # Transpose of all features for bmm: (B, C, P)
            features_for_dot_product = features_transposed.transpose(1, 2)  # (B, C, P)

            # Pre-compute invalid mask for reference points
            mask_flat = mask.squeeze(1)  # (B, P)
            invalid_mask = ~mask_flat.bool()  # (B, P)
            # Reference norms transposed: (B, 1, P) for broadcasting
            norms_squared_reference = norms_squared.transpose(1, 2)  # (B, 1, P)

            # Process query tracks in chunks
            chunk_indices_list = []

            for chunk_start in range(0, num_tracks, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_tracks)
                current_chunk_size = chunk_end - chunk_start

                # Chunk of query features: (B, chunk, C)
                chunk_features = features_transposed[
                    :, chunk_start:chunk_end, :
                ]  # (B, chunk, C)

                # Chunk norms: (B, chunk, 1)
                chunk_norms = norms_squared[
                    :, chunk_start:chunk_end, :
                ]  # (B, chunk, 1)

                # Chunk pairwise distances via expanded formula:
                # ||f_i - f_j||^2 = ||f_i||^2 + ||f_j||^2 - 2 * f_i^T f_j
                # chunk_norms: (B, chunk, 1) broadcasts with norms_ref: (B, 1, P)
                # dot_product: (B, chunk, C) @ (B, C, P) = (B, chunk, P)
                dot_product = torch.bmm(
                    chunk_features, features_for_dot_product,
                )  # (B, chunk, P)
                chunk_distances = (
                    chunk_norms + norms_squared_reference - 2.0 * dot_product
                )  # (B, chunk, P)

                # Mask invalid reference points: set distance to +inf
                chunk_distances.masked_fill_(
                    invalid_mask.unsqueeze(1), float('inf'),
                )

                # Exclude self-matches: set diagonal distance to +inf.
                # For chunk [chunk_start:chunk_end], self-match is at column
                # index == row_global_index, i.e., column chunk_start + local_row.
                if chunk_end > chunk_start:
                    batch_range = torch.arange(
                        batch_size, device=features.device,
                    )
                    local_range = torch.arange(
                        current_chunk_size, device=features.device,
                    )
                    global_indices = local_range + chunk_start
                    # Set self-match distances to inf: chunk_distances[b, i, chunk_start+i]
                    chunk_distances[
                        batch_range.unsqueeze(1),
                        local_range.unsqueeze(0),
                        global_indices.unsqueeze(0),
                    ] = float('inf')

                # kNN per chunk: select K nearest neighbors.
                # sorted=False: downstream GAPLayer operations (attention-weighted
                # sum, max-pool) are permutation-invariant over neighbors.
                _, chunk_neighbor_indices = chunk_distances.topk(
                    self.gap2_num_neighbors, dim=-1, largest=False, sorted=False,
                )  # (B, chunk, K)

                chunk_indices_list.append(chunk_neighbor_indices)

            # Concatenate all chunks along the query dimension
            neighbor_indices = torch.cat(
                chunk_indices_list, dim=1,
            )  # (B, P, K)

        return neighbor_indices

    def _compute_global_context(
        self,
        enriched_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute event-level global context and tile to per-track.

        Masked average pooling over valid tracks, then project and tile
        back to (B, global_dim, P). Each track gets the same event-level
        context vector.

        Args:
            enriched_features: (B, backbone_dim, P) from backbone.
            mask: (B, 1, P) boolean mask.

        Returns:
            global_context: (B, global_dim, P) tiled context features.
        """
        mask_float = mask.float()  # (B, 1, P)
        num_valid = mask_float.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)

        # Masked average pool: (B, backbone_dim, P) -> (B, backbone_dim)
        masked_features = enriched_features * mask_float
        pooled = masked_features.sum(dim=-1) / num_valid.squeeze(-1)  # (B, backbone_dim)

        # Project to global_context_dim
        projected = self.global_context_projection(pooled)  # (B, global_dim)

        # Tile to all track positions: (B, global_dim) -> (B, global_dim, P)
        num_tracks = enriched_features.shape[2]
        tiled = projected.unsqueeze(-1).expand(-1, -1, num_tracks)

        return tiled

    def _asymmetric_loss(
        self,
        predicted_logits: torch.Tensor,
        target_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Asymmetric Loss (ASL) over valid tracks.

        ASL (Ben-Baruch et al., ICCV 2021) uses different focusing
        parameters for positives vs negatives, plus a hard probability-shift
        threshold that zeros gradients from trivially easy negatives.

        For positives: L = -(1-p)^γ+ * log(p)
        For negatives: L = -max(p-m, 0)^γ- * log(1-p)

        where m is the clip threshold (default 0.05). When a negative has
        predicted probability p < m, its loss (and gradient) is exactly zero.

        With 0.23% positive rate and γ-=4, m=0.05:
            - ~60% of background tracks have p < 0.05 → zero gradient
            - Gradient balance shifts from 1.76 (easy neg) to 0.00
            - Positives (1.56) now dominate over hard negatives (0.69)

        Args:
            predicted_logits: (B, N) per-track logits (pre-sigmoid).
            target_labels: (B, N) binary labels (1.0 = tau track).
            valid_mask: (B, N) boolean or float, True/1.0 for valid tracks.

        Returns:
            Scalar ASL loss averaged over valid tracks.
        """
        # Clamp logits to avoid NaN from BCE on extreme values (padded tracks)
        predicted_logits = predicted_logits.clamp(min=-50.0, max=50.0)
        predicted_probabilities = torch.sigmoid(predicted_logits)

        # ---- Positive loss: -(1-p)^γ+ * log(p) ----
        # Numerically stable: use BCE then multiply by focal weight
        positive_bce = functional.binary_cross_entropy_with_logits(
            predicted_logits, torch.ones_like(predicted_logits), reduction='none',
        )
        positive_focal_weight = (1.0 - predicted_probabilities) ** self.focal_gamma_positive
        positive_loss = positive_focal_weight * positive_bce

        # ---- Negative loss: -max(p-m, 0)^γ- * log(1-p) ----
        # Hard probability shift: clamp (p - clip) to zero
        shifted_probability = (predicted_probabilities - self.asl_clip).clamp(min=0.0)
        negative_bce = functional.binary_cross_entropy_with_logits(
            predicted_logits, torch.zeros_like(predicted_logits), reduction='none',
        )
        # γ- applied to shifted probability (not original)
        negative_focal_weight = shifted_probability ** self.focal_gamma_negative
        negative_loss = negative_focal_weight * negative_bce

        # ---- Combine based on labels ----
        loss_per_track = torch.where(
            target_labels == 1.0,
            positive_loss,
            negative_loss,
        )

        # Average over valid tracks only
        valid_float = valid_mask.float()
        loss_per_track = loss_per_track * valid_float
        num_valid = valid_float.sum().clamp(min=1.0)
        return loss_per_track.sum() / num_valid

    def _ranking_loss(
        self,
        predicted_logits: torch.Tensor,
        target_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sampled pairwise ranking loss.

        For each GT pion, sample S negatives and penalize any negative
        that scores above the positive:
            L_rank = mean_i mean_j log(1 + exp(score_neg_j - score_pos_i))

        This directly optimizes for recall@K by pushing positive scores
        above negative scores. Differentiable and GPU-friendly.

        Evidence: val loss stalls at 0.00274 from epoch 15-50, but R@30
        improves 0.216→0.239 (+10.6%). BCE cannot capture ranking quality;
        this loss provides gradient signal for those improvements.

        # TODO: Vectorize this loop across the batch using padded positives
        # and a single torch.randint call. The per-event loop is kept for now
        # because it operates on tiny tensors and the main performance
        # bottlenecks are elsewhere (kNN, forward passes).

        Args:
            predicted_logits: (B, N) per-track logits.
            target_labels: (B, N) binary labels (1.0 = tau track).
            valid_mask: (B, N) boolean mask.

        Returns:
            Scalar ranking loss. Zero if no GT tracks in the batch.
        """
        batch_size = predicted_logits.shape[0]
        event_losses = []

        for event_index in range(batch_size):
            event_labels = target_labels[event_index]
            event_scores = predicted_logits[event_index]
            event_valid = valid_mask[event_index]

            # Find positive and negative positions
            positive_mask = (event_labels == 1.0) & event_valid
            negative_mask = (event_labels == 0.0) & event_valid
            positive_indices = positive_mask.nonzero(as_tuple=True)[0]
            negative_indices = negative_mask.nonzero(as_tuple=True)[0]

            num_positives = len(positive_indices)
            num_negatives = len(negative_indices)

            if num_positives == 0 or num_negatives == 0:
                continue

            # Sample S negatives per positive (with replacement if needed)
            num_samples = min(self.ranking_num_samples, num_negatives)
            sample_indices = torch.randint(
                0, num_negatives, (num_samples,),
                device=predicted_logits.device,
            )
            sampled_negative_indices = negative_indices[sample_indices]

            # Positive scores: (num_positives, 1)
            positive_scores = event_scores[positive_indices].unsqueeze(1)
            # Sampled negative scores: (1, num_samples)
            negative_scores = event_scores[sampled_negative_indices].unsqueeze(0)

            # Pairwise ranking loss: log(1 + exp(s_neg - s_pos))
            # Numerically stable via softplus (handles overflow when |Δ| > 40)
            # Shape: (num_positives, num_samples)
            pairwise_loss = functional.softplus(
                negative_scores - positive_scores,
            )

            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            return torch.tensor(
                0.0, device=predicted_logits.device,
                dtype=predicted_logits.dtype,
            )

        return torch.stack(event_losses).mean()

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: backbone -> GAPLayers -> scoring -> loss/logits.

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) standardized per-track features.
            lorentz_vectors: (B, 4, P) raw per-track 4-vectors [px, py, pz, E].
            mask: (B, 1, P) boolean mask, True for valid tracks.
            track_labels: (B, 1, P) binary labels (1.0 = tau pion).

        Returns:
            Training: {'total_loss', 'per_track_loss'}
            Inference: {'per_track_logits': (B, P)}
        """
        mask_float = mask.float()
        valid_mask = mask.squeeze(1).bool()  # (B, P)

        # ---- Step 1: Backbone feature extraction ----
        if self.backbone_mode == 'parallel':
            # Trainable parallel backbone: identity + context streams
            enriched_features = self.backbone(
                points, features, lorentz_vectors, mask,
            )  # (B, backbone_dim, P)
        else:
            # Frozen pretrained backbone enrichment
            with torch.no_grad():
                enriched_features = self.backbone.enrich(
                    points, features, lorentz_vectors, mask,
                )
            enriched_features = enriched_features.detach()  # (B, backbone_dim, P)

        # ---- Step 2: GAPLayer 1 in physical (eta, phi) space ----
        physical_knn_indices = self._compute_physical_knn(points, mask)
        gap1_attention, gap1_graph = self.gap_layer_physical(
            enriched_features, physical_knn_indices, mask,
        )  # Each: (B, gap1_encoding_dim, P)

        # Intermediate MLPs
        gap1_combined = torch.cat([gap1_attention, gap1_graph], dim=1)
        intermediate_1 = self.intermediate_mlp_1(gap1_combined)  # (B, intermediate_dim, P)
        intermediate_1 = intermediate_1 * mask_float

        # ---- Step 4: GAPLayer 2 in learned feature space ----
        feature_knn_indices = self._compute_feature_space_knn(
            intermediate_1, mask,
        )
        gap2_attention, gap2_graph = self.gap_layer_learned(
            intermediate_1, feature_knn_indices, mask,
        )  # Each: (B, gap2_encoding_dim, P)

        # Intermediate MLPs
        gap2_combined = torch.cat([gap2_attention, gap2_graph], dim=1)
        intermediate_2 = self.intermediate_mlp_2(gap2_combined)  # (B, intermediate_dim, P)
        intermediate_2 = intermediate_2 * mask_float

        # ---- Step 6: Global context ----
        global_context = self._compute_global_context(
            enriched_features, mask,
        )  # (B, global_dim, P)

        # ---- Step 7: Skip-connected raw features + Lorentz vectors ----
        raw_features_normalized = self.raw_feature_norm(features) * mask_float
        lorentz_normalized = self.lorentz_vector_norm(
            lorentz_vectors.float(),
        ).to(features.dtype) * mask_float

        # ---- Multi-scale concatenation ----
        combined = torch.cat([
            gap1_attention,           # GAPLayer 1 attention output
            gap1_graph,               # GAPLayer 1 graph features
            gap2_attention,           # GAPLayer 2 attention output
            gap2_graph,               # GAPLayer 2 graph features
            enriched_features,        # Backbone enriched features
            raw_features_normalized,  # All raw features (BN-normalized)
            lorentz_normalized,       # Lorentz 4-vectors (BN-normalized)
            global_context,           # Event-level global context
        ], dim=1)  # (B, combined_dim, P)

        # ---- Step 8: Per-track scoring ----
        per_track_logits = self.per_track_head(combined).squeeze(1)  # (B, P)
        per_track_logits = per_track_logits * valid_mask.float()  # Zero padded

        # ---- Training: compute loss ----
        if track_labels is not None:
            labels_flat = (
                track_labels.squeeze(1)[:, :per_track_logits.shape[1]]
                * valid_mask.float()
            )

            # Asymmetric Loss (replaces focal BCE)
            per_track_loss = self._asymmetric_loss(
                per_track_logits, labels_flat, valid_mask,
            )

            # Pairwise ranking loss (directly optimizes recall@K)
            ranking_loss = self._ranking_loss(
                per_track_logits, labels_flat, valid_mask,
            )

            total_loss = per_track_loss + self.ranking_loss_weight * ranking_loss

            return {
                'total_loss': total_loss,
                'per_track_loss': per_track_loss,
                'ranking_loss': ranking_loss,
                '_per_track_logits': per_track_logits,
            }

        # ---- Inference: return per-track logits ----
        return {
            'per_track_logits': per_track_logits,
        }
