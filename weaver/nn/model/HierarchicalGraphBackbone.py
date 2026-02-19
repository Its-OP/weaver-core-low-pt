"""Hierarchical Graph Convolution Backbone for Per-Track Tau Classification.

Progressively downsamples ~1130 pion tracks into M=64 dense tokens via
PointNet++ set abstraction stages with physics-informed edge features
(ParT's pairwise Lorentz-vector features: ln kT, ln z, ln ΔR, ln m²).

Architecture:
    Stage 0: Input embedding — Conv1d(7 → 64) + BN + ReLU
    Stage 1: 1130 → 256  (FPS + kNN + EdgeConv + LV propagation)
    Stage 2: 256 → 128
    Stage 3: 128 → 64

Each stage uses Farthest Point Sampling in (η, φ) with phi wrapping,
cross-set kNN for neighbor lookup, and attention-weighted aggregation
with 4-vector propagation.

kNN always operates on 2D (η, φ) coordinates regardless of feature
dimensionality — spatial proximity is geometric (PointNet++ design).
The high-dimensional features only enter through EdgeConv after neighbors
are identified by spatial proximity.
"""
import math
import torch
import torch.nn as nn

from weaver.nn.model.ParticleTransformer import pairwise_lv_fts


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _delta_phi(phi_a: torch.Tensor, phi_b: torch.Tensor) -> torch.Tensor:
    """Compute phi difference with proper [-π, π] wrapping.

    Formula: Δφ = (φ_a − φ_b + π) mod 2π − π
    """
    return (phi_a - phi_b + math.pi) % (2 * math.pi) - math.pi


def _delta_r_squared(
    eta_a: torch.Tensor,
    phi_a: torch.Tensor,
    eta_b: torch.Tensor,
    phi_b: torch.Tensor,
) -> torch.Tensor:
    """Compute ΔR² with phi wrapping.

    Formula: ΔR² = (η_a − η_b)² + Δφ(φ_a, φ_b)²
    """
    return (eta_a - eta_b).square() + _delta_phi(phi_a, phi_b).square()


def farthest_point_sampling(
    coordinates: torch.Tensor,
    mask: torch.Tensor,
    num_centroids: int,
) -> torch.Tensor:
    """Farthest Point Sampling in (η, φ) space with phi wrapping.

    Greedy algorithm:
        S₀ = {random valid point}
        For t = 1, ..., M−1:
            d_i = min_{s ∈ S_{t-1}} ΔR²(p_i, s)   ∀ valid i ∉ S
            S_t = S_{t-1} ∪ {argmax_i d_i}

    The loop over M iterations is sequential (each depends on the previous)
    but each iteration is vectorized over the batch and point dimensions.

    Args:
        coordinates: (B, 2, P) where dim 1 is (η, φ).
        mask: (B, 1, P) boolean mask, True for valid points.
        num_centroids: M — number of centroids to select.

    Returns:
        centroid_indices: (B, M) indices into the P dimension, dtype=long.
    """
    batch_size, _, num_points = coordinates.shape
    device = coordinates.device

    # (B, P)
    point_mask = mask.squeeze(1).bool()

    eta = coordinates[:, 0, :]  # (B, P)
    phi = coordinates[:, 1, :]  # (B, P)

    # Minimum distance from each point to any already-selected centroid.
    # Initialize to +∞ for valid points, −∞ for padded points.
    min_distances = torch.full(
        (batch_size, num_points), float('inf'), device=device, dtype=coordinates.dtype
    )
    min_distances.masked_fill_(~point_mask, float('-inf'))

    centroid_indices = torch.zeros(
        batch_size, num_centroids, device=device, dtype=torch.long
    )

    # Select the first centroid: pick a random valid point per event.
    # Use the first valid point for deterministic behavior during inference.
    # During training, randomness comes from data shuffling already.
    for batch_idx in range(batch_size):
        valid_positions = point_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        if valid_positions.numel() > 0:
            random_pos = torch.randint(valid_positions.numel(), (1,), device=device)
            centroid_indices[batch_idx, 0] = valid_positions[random_pos]

    for step in range(num_centroids):
        # Get coordinates of the newly selected centroid
        current_idx = centroid_indices[:, step]  # (B,)
        centroid_eta = eta.gather(1, current_idx.unsqueeze(1)).squeeze(1)  # (B,)
        centroid_phi = phi.gather(1, current_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # ΔR² from every point to the new centroid: (B, P)
        distance_to_centroid = _delta_r_squared(
            eta, phi,
            centroid_eta.unsqueeze(1), centroid_phi.unsqueeze(1),
        )

        # Update minimum distances: d_i = min(d_i, ΔR²(p_i, new_centroid))
        min_distances = torch.minimum(min_distances, distance_to_centroid)

        # Mask out padded points
        min_distances.masked_fill_(~point_mask, float('-inf'))

        if step + 1 < num_centroids:
            # Select the point with the largest minimum distance
            next_centroid = min_distances.argmax(dim=1)  # (B,)
            centroid_indices[:, step + 1] = next_centroid

    return centroid_indices


def cross_set_knn(
    query_coordinates: torch.Tensor,
    reference_coordinates: torch.Tensor,
    num_neighbors: int,
    reference_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-set k-Nearest Neighbors in (η, φ) space with phi wrapping.

    Computes distances between M query points and P reference points,
    returns the K nearest reference points for each query.

    Args:
        query_coordinates: (B, 2, M) query centroids in (η, φ).
        reference_coordinates: (B, 2, P) all reference points in (η, φ).
        num_neighbors: K — neighbors per query.
        reference_mask: (B, 1, P) boolean mask for reference points,
            True for valid. If None, all points are considered valid.

    Returns:
        neighbor_indices: (B, M, K) indices into the P dimension, dtype=long.
    """
    # Extract η and φ for queries and references
    query_eta = query_coordinates[:, 0:1, :]    # (B, 1, M)
    query_phi = query_coordinates[:, 1:2, :]    # (B, 1, M)
    reference_eta = reference_coordinates[:, 0:1, :]  # (B, 1, P)
    reference_phi = reference_coordinates[:, 1:2, :]  # (B, 1, P)

    # Distance matrix: ΔR²(query_m, ref_p) → (B, M, P)
    # Broadcast: (B, 1, M, 1) vs (B, 1, 1, P) → (B, 1, M, P)
    delta_eta = query_eta.unsqueeze(-1) - reference_eta.unsqueeze(-2)
    delta_phi_val = _delta_phi(
        query_phi.unsqueeze(-1), reference_phi.unsqueeze(-2)
    )
    distances = (delta_eta.square() + delta_phi_val.square()).squeeze(1)  # (B, M, P)

    # Mask out invalid reference points with large distance
    if reference_mask is not None:
        invalid_mask = ~reference_mask.bool()  # (B, 1, P)
        distances = distances.masked_fill(invalid_mask, float('inf'))

    # Select K nearest neighbors (smallest distances)
    neighbor_indices = distances.topk(
        k=num_neighbors, dim=-1, largest=False, sorted=True
    )[1]  # (B, M, K)

    return neighbor_indices


def cross_set_gather(
    reference_features: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather features from reference set using cross-set neighbor indices.

    Args:
        reference_features: (B, C, P) features of all reference points.
        neighbor_indices: (B, M, K) indices into the P dimension.

    Returns:
        gathered_features: (B, C, M, K) neighbor features for each query.
    """
    batch_size, num_channels, num_reference_points = reference_features.shape
    _, num_queries, num_neighbors = neighbor_indices.shape

    # Flatten batch indexing: offset indices by batch * P
    batch_offset = (
        torch.arange(batch_size, device=reference_features.device)
        .view(-1, 1, 1) * num_reference_points
    )  # (B, 1, 1)
    flat_indices = (neighbor_indices + batch_offset).reshape(-1)  # (B*M*K,)

    # (B, C, P) → (B, P, C) → (B*P, C) → index → (B*M*K, C) → reshape
    flat_features = reference_features.transpose(1, 2).reshape(-1, num_channels)
    gathered = flat_features[flat_indices]  # (B*M*K, C)
    gathered = gathered.view(batch_size, num_queries, num_neighbors, num_channels)
    gathered = gathered.permute(0, 3, 1, 2).contiguous()  # (B, C, M, K)

    return gathered


def build_cross_set_edge_features(
    center_features: torch.Tensor,
    neighbor_features: torch.Tensor,
    center_lorentz_vectors: torch.Tensor,
    neighbor_lorentz_vectors: torch.Tensor,
) -> torch.Tensor:
    """Build edge features for cross-set EdgeConv with pairwise LV physics.

    Concatenates [center, neighbor − center, pairwise_lv_fts] along the
    channel dimension.

    Pairwise LV features (from ParticleTransformer.py:79):
        ln kT, ln z, ln ΔR, ln m² — 4 physics features encoding QCD
        splitting kinematics between center and neighbor 4-vectors.

    Args:
        center_features: (B, C, M) centroid feature vectors.
        neighbor_features: (B, C, M, K) gathered neighbor features.
        center_lorentz_vectors: (B, 4, M) centroid 4-vectors (px, py, pz, E).
        neighbor_lorentz_vectors: (B, 4, M, K) gathered neighbor 4-vectors.

    Returns:
        edge_features: (B, 2*C + 4, M, K) concatenated edge tensor.
    """
    # Expand center features to (B, C, M, K) for per-neighbor comparison
    center_expanded = center_features.unsqueeze(-1).expand_as(neighbor_features)

    # Relative features: neighbor − center
    relative_features = neighbor_features - center_expanded

    # Pairwise Lorentz-vector features: (B, 4, M, K)
    # pairwise_lv_fts expects (B, 4, ...) and works on any trailing dims.
    # Computes: ln kT, ln z, ln ΔR, ln m²
    center_lv_expanded = center_lorentz_vectors.unsqueeze(-1).expand_as(
        neighbor_lorentz_vectors
    )
    # pairwise_lv_fts internally computes ΔR = sqrt(ΔR²), which has
    # undefined gradient when ΔR²=0 (centroid is its own kNN neighbor).
    # Add tiny noise to neighbor LVs to break exact equality and prevent
    # sqrt(0) in the backward pass. The noise magnitude (1e-6) is negligible
    # compared to physical momenta (O(0.1-10 GeV)).
    neighbor_lv_perturbed = neighbor_lorentz_vectors + torch.randn_like(
        neighbor_lorentz_vectors
    ) * 1e-6

    lv_features = pairwise_lv_fts(
        center_lv_expanded, neighbor_lv_perturbed, num_outputs=4
    )  # (B, 4, M, K)

    # Concatenate: [center, neighbor − center, pairwise_lv_fts]
    # → (B, 2*C + 4, M, K)
    edge_features = torch.cat(
        [center_expanded, relative_features, lv_features], dim=1
    )

    return edge_features


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class SetAbstractionStage(nn.Module):
    """PointNet++ set abstraction with physics-informed edge features.

    Reduces point count: P_in → P_out via FPS + kNN + EdgeConv.
    Propagates Lorentz 4-vectors via attention-weighted aggregation.

    Steps:
        1. FPS selects P_out centroids from P_in points in (η, φ) space
        2. kNN finds K nearest input points per centroid in (η, φ) space
        3. Edge features: [center_feat, neighbor_feat − center_feat,
                           pairwise_lv_fts(center_lv, neighbor_lv)]
        4. EdgeConv MLP processes edges → messages  (B, C_out, P_out, K)
        5. Attention aggregation over K neighbors → (B, C_out, P_out)
        6. 4-vector propagation: attention_weights @ neighbor_lvs → (B, 4, P_out)

    No within-stage residual: P_in ≠ P_out and C_in ≠ C_out, so a shortcut
    connection is not applicable. This matches PointNet++'s original design.

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

        # Edge feature dimension: 2 * C_in (center + relative) + 4 (pairwise LV)
        edge_feature_dim = 2 * input_channels + 4

        # EdgeConv MLP: two Conv2d layers operating on (B, C, M, K)
        self.edge_convolution = nn.Sequential(
            nn.Conv2d(edge_feature_dim, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

        # Attention scoring: Conv2d(C_out → 1, kernel=1)
        # Produces per-neighbor attention logits for weighted aggregation
        self.attention_scorer = nn.Conv2d(
            output_channels, 1, kernel_size=1, bias=False
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: downsample P_in → P_out.

        Args:
            coordinates: (B, 2, P_in) in (η, φ) space.
            features: (B, C_in, P_in) per-point features.
            lorentz_vectors: (B, 4, P_in) per-point 4-vectors (px, py, pz, E).
            mask: (B, 1, P_in) boolean mask, True for valid.

        Returns:
            Tuple of:
                output_features: (B, C_out, P_out)
                output_lorentz_vectors: (B, 4, P_out)
                output_coordinates: (B, 2, P_out)
                output_mask: (B, 1, P_out) — all True (FPS only selects valid)
        """
        batch_size = coordinates.shape[0]

        # Step 1: Farthest Point Sampling → select P_out centroid indices
        centroid_indices = farthest_point_sampling(
            coordinates, mask, self.num_output_points
        )  # (B, P_out)

        # Gather centroid data using the selected indices
        # idx_expanded: (B, 1, P_out) for use with torch.gather on dim=2
        idx_expanded = centroid_indices.unsqueeze(1)  # (B, 1, P_out)

        centroid_coordinates = coordinates.gather(
            2, idx_expanded.expand(-1, 2, -1)
        )  # (B, 2, P_out)
        centroid_features = features.gather(
            2, idx_expanded.expand(-1, features.shape[1], -1)
        )  # (B, C_in, P_out)
        centroid_lorentz_vectors = lorentz_vectors.gather(
            2, idx_expanded.expand(-1, 4, -1)
        )  # (B, 4, P_out)

        # Step 2: Cross-set kNN — find K nearest reference points per centroid
        # kNN operates in 2D (η, φ) space, not in feature space
        neighbor_indices = cross_set_knn(
            centroid_coordinates, coordinates, self.num_neighbors, mask
        )  # (B, P_out, K)

        # Step 3: Gather neighbor features and 4-vectors from input tensors
        neighbor_features = cross_set_gather(
            features, neighbor_indices
        )  # (B, C_in, P_out, K)
        neighbor_lorentz_vectors = cross_set_gather(
            lorentz_vectors, neighbor_indices
        )  # (B, 4, P_out, K)

        # Build edge features: [center, neighbor − center, pairwise_lv_fts]
        # → (B, 2*C_in + 4, P_out, K)
        edge_features = build_cross_set_edge_features(
            centroid_features, neighbor_features,
            centroid_lorentz_vectors, neighbor_lorentz_vectors,
        )

        # Step 4: EdgeConv MLP → messages (B, C_out, P_out, K)
        messages = self.edge_convolution(edge_features)

        # Step 5: Attention aggregation over K neighbors
        # attention_logits: (B, 1, P_out, K)
        attention_logits = self.attention_scorer(messages)

        # Mask invalid neighbors: gather reference mask for each neighbor
        neighbor_mask = cross_set_gather(
            mask.float(), neighbor_indices
        )  # (B, 1, P_out, K)
        attention_logits = attention_logits.masked_fill(
            neighbor_mask == 0, float('-inf')
        )

        # Softmax over K neighbors: (B, 1, P_out, K)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        # Handle all-masked case: replace NaN from softmax(-inf) with 0
        attention_weights = attention_weights.nan_to_num(0.0)

        # Weighted sum: (B, C_out, P_out, K) × (B, 1, P_out, K) → sum → (B, C_out, P_out)
        output_features = (attention_weights * messages).sum(dim=-1)

        # Step 6: 4-vector propagation via attention weights
        # (B, 1, P_out, K) × (B, 4, P_out, K) → sum → (B, 4, P_out)
        output_lorentz_vectors = (
            attention_weights * neighbor_lorentz_vectors
        ).sum(dim=-1)

        # Output mask: all centroids are valid (FPS only selects valid points)
        output_mask = torch.ones(
            batch_size, 1, self.num_output_points,
            device=features.device, dtype=mask.dtype,
        )

        return output_features, output_lorentz_vectors, centroid_coordinates, output_mask


class HierarchicalGraphBackbone(nn.Module):
    """Hierarchical graph convolution backbone: ~1130 tracks → 64 dense tokens.

    DETR-style backbone that progressively downsamples a particle cloud
    through 3 set abstraction stages, each combining FPS + kNN + EdgeConv
    with physics-informed pairwise Lorentz-vector features.

    Stage 0: Input embedding (per-track Conv1d, no downsampling)
    Stage 1: P → N₁ (default 1130 → 256)
    Stage 2: N₁ → N₂ (default 256 → 128)
    Stage 3: N₂ → M  (default 128 → 64)

    Channel widths grow as resolution shrinks: 64 → 128 → 192 → 256.
    This is the standard CNN trade-off: fewer spatial points, richer features.

    Args:
        input_dim: Number of input features per track (default: 7).
        embed_dim: Embedding dimension after Stage 0 (default: 64).
        stage_output_points: List of 3 ints — output point counts per stage
            (default: [256, 128, 64]).
        stage_output_channels: List of 3 ints — output channel widths per stage
            (default: [128, 192, 256]).
        stage_num_neighbors: List of 3 ints — kNN K per stage
            (default: [32, 24, 16]).
    """

    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = 64,
        stage_output_points: list[int] | None = None,
        stage_output_channels: list[int] | None = None,
        stage_num_neighbors: list[int] | None = None,
    ):
        super().__init__()

        if stage_output_points is None:
            stage_output_points = [256, 128, 64]
        if stage_output_channels is None:
            stage_output_channels = [128, 192, 256]
        if stage_num_neighbors is None:
            stage_num_neighbors = [32, 24, 16]

        assert len(stage_output_points) == 3
        assert len(stage_output_channels) == 3
        assert len(stage_num_neighbors) == 3

        # Stage 0: Input embedding — Conv1d(input_dim → embed_dim) + BN + ReLU
        self.input_embedding = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # Stage 1: embed_dim → stage_output_channels[0]
        self.stage1 = SetAbstractionStage(
            input_channels=embed_dim,
            output_channels=stage_output_channels[0],
            num_output_points=stage_output_points[0],
            num_neighbors=stage_num_neighbors[0],
        )

        # Stage 2: stage_output_channels[0] → stage_output_channels[1]
        self.stage2 = SetAbstractionStage(
            input_channels=stage_output_channels[0],
            output_channels=stage_output_channels[1],
            num_output_points=stage_output_points[1],
            num_neighbors=stage_num_neighbors[1],
        )

        # Stage 3: stage_output_channels[1] → stage_output_channels[2]
        self.stage3 = SetAbstractionStage(
            input_channels=stage_output_channels[1],
            output_channels=stage_output_channels[2],
            num_output_points=stage_output_points[2],
            num_neighbors=stage_num_neighbors[2],
        )

        self.output_dim = stage_output_channels[-1]

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: embed + 3 hierarchical stages.

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) per-track features.
            lorentz_vectors: (B, 4, P) per-track 4-vectors (px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            Tuple of:
                tokens: (B, C_out, M) dense token features.
                token_lorentz_vectors: (B, 4, M) token 4-vectors.
                token_coordinates: (B, 2, M) token positions in (η, φ).
        """
        # Stage 0: Input embedding (no downsampling)
        features = self.input_embedding(features) * mask  # (B, embed_dim, P)

        # Stage 1: P → N₁
        features, lorentz_vectors, points, mask = self.stage1(
            points, features, lorentz_vectors, mask
        )

        # Stage 2: N₁ → N₂
        features, lorentz_vectors, points, mask = self.stage2(
            points, features, lorentz_vectors, mask
        )

        # Stage 3: N₂ → M
        features, lorentz_vectors, points, mask = self.stage3(
            points, features, lorentz_vectors, mask
        )

        return features, lorentz_vectors, points
