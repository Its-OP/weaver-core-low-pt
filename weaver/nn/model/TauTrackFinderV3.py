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
    - Focal BCE on ALL ~1130 tracks (same formulation as V2).

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
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        num_neighbors: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.num_neighbors = num_neighbors
        self.num_heads = num_heads

        # Per-head parameter modules
        self.node_encoders = nn.ModuleList()
        self.edge_encoders = nn.ModuleList()
        self.self_attention_scorers = nn.ModuleList()
        self.neighbor_attention_scorers = nn.ModuleList()

        for _ in range(num_heads):
            # Node encoder: x_i -> x'_i of dimension encoding_dim
            # Conv1d operates on (B, C_in, P) -> (B, encoding_dim, P)
            self.node_encoders.append(nn.Sequential(
                nn.Conv1d(input_dim, encoding_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(encoding_dim),
            ))

            # Edge encoder: y_ij -> y'_ij of dimension encoding_dim
            # Conv2d operates on (B, C_in, P, K) -> (B, encoding_dim, P, K)
            self.edge_encoders.append(nn.Sequential(
                nn.Conv2d(input_dim, encoding_dim, kernel_size=1, bias=True),
                nn.BatchNorm2d(encoding_dim),
            ))

            # Self-attention scorer: x'_i -> scalar
            # Broadcasts over K dimension via unsqueeze
            self.self_attention_scorers.append(
                nn.Conv1d(encoding_dim, 1, kernel_size=1, bias=True),
            )

            # Neighbor attention scorer: y'_ij -> scalar
            self.neighbor_attention_scorers.append(
                nn.Conv2d(encoding_dim, 1, kernel_size=1, bias=True),
            )

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
        attention_coefficients = functional.softmax(
            attention_logits, dim=-1,
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

        Args:
            features: (B, input_dim, P) per-track features.
            neighbor_indices: (B, P, K) kNN indices.
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            attention_features: (B, encoding_dim, P) — max across heads of
                attention-weighted outputs.
            graph_features: (B, encoding_dim, P) — max across heads of
                max-pooled edge features.
        """
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
        backbone_kwargs: Config for EnrichCompactBackbone.
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
        backbone_kwargs: dict | None = None,
        gap1_encoding_dim: int = 64,
        gap1_num_neighbors: int = 16,
        gap1_num_heads: int = 4,
        gap2_encoding_dim: int = 64,
        gap2_num_neighbors: int = 16,
        gap2_num_heads: int = 4,
        intermediate_dim: int = 128,
        global_context_dim: int = 32,
        scoring_dropout: float = 0.4,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}

        self.gap1_num_neighbors = gap1_num_neighbors
        self.gap2_num_neighbors = gap2_num_neighbors
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # ---- Backbone (frozen) ----
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
    ) -> torch.Tensor:
        """Compute kNN indices in learned feature space.

        Uses pairwise L2 distance in the high-dimensional feature space.
        This allows finding tracks that are similar in learned representation
        even if far apart in (eta, phi).

        Args:
            features: (B, C, P) intermediate features.
            mask: (B, 1, P) boolean mask.

        Returns:
            neighbor_indices: (B, P, K) kNN indices.
        """
        with torch.no_grad():
            # Pairwise L2 distance in feature space
            # features: (B, C, P) -> transpose to (B, P, C) for distance
            features_transposed = features.transpose(1, 2)  # (B, P, C)

            # ||f_i - f_j||^2 = ||f_i||^2 + ||f_j||^2 - 2 * f_i^T f_j
            feature_norms_squared = (
                features_transposed.pow(2).sum(dim=-1, keepdim=True)
            )  # (B, P, 1)
            pairwise_distances = (
                feature_norms_squared
                + feature_norms_squared.transpose(1, 2)
                - 2.0 * torch.bmm(
                    features_transposed, features_transposed.transpose(1, 2),
                )
            )  # (B, P, P)

            # Mask invalid reference points: set distance to +inf
            mask_flat = mask.squeeze(1)  # (B, P)
            invalid_mask = ~mask_flat.bool()  # (B, P)
            pairwise_distances.masked_fill_(
                invalid_mask.unsqueeze(1), float('inf'),
            )

            # kNN: select K nearest neighbors per point
            _, neighbor_indices = pairwise_distances.topk(
                self.gap2_num_neighbors, dim=-1, largest=False,
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

    def _focal_bce_loss(
        self,
        predicted_logits: torch.Tensor,
        target_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal binary cross-entropy loss over valid tracks.

        Focal loss (Lin et al., RetinaNet, ICCV 2017):
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        With ~3 positive out of ~1130 tracks:
            - alpha = 0.75 upweights the rare positive class
            - gamma = 2.0 downweights easy negatives

        Args:
            predicted_logits: (B, N) per-track logits (pre-sigmoid).
            target_labels: (B, N) binary labels (1.0 = tau track).
            valid_mask: (B, N) boolean or float, True/1.0 for valid tracks.

        Returns:
            Scalar focal BCE loss averaged over valid tracks.
        """
        # Standard per-element BCE (numerically stable via log-sum-exp)
        bce_per_track = functional.binary_cross_entropy_with_logits(
            predicted_logits, target_labels, reduction='none',
        )

        # p_t = P(correct class)
        predicted_probabilities = torch.sigmoid(predicted_logits)
        probability_correct = torch.where(
            target_labels == 1.0,
            predicted_probabilities,
            1.0 - predicted_probabilities,
        )

        # alpha_t: class-balancing weight
        alpha_weight = torch.where(
            target_labels == 1.0,
            self.focal_alpha,
            1.0 - self.focal_alpha,
        )

        # FL(p_t) = alpha_t * (1 - p_t)^gamma * BCE(x, y)
        focal_weight = alpha_weight * (
            (1.0 - probability_correct) ** self.focal_gamma
        )
        focal_loss_per_track = focal_weight * bce_per_track

        # Average over valid tracks only
        valid_float = valid_mask.float()
        focal_loss_per_track = focal_loss_per_track * valid_float
        num_valid = valid_float.sum().clamp(min=1.0)
        return focal_loss_per_track.sum() / num_valid

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

        # ---- Step 1: Backbone enrichment (frozen) ----
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

            per_track_loss = self._focal_bce_loss(
                per_track_logits, labels_flat, valid_mask,
            )

            return {
                'total_loss': per_track_loss,
                'per_track_loss': per_track_loss,
            }

        # ---- Inference: return per-track logits ----
        return {
            'per_track_logits': per_track_logits,
        }
