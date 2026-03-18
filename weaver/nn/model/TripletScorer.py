"""Anchor-based triplet enumeration and scoring.

For each anchor track (top-200 by pre-filter score), enumerates candidate
triplets from the anchor's N=512 nearest neighbors using physics cuts
(invariant mass < 1.777 GeV, charge sum = ±1, max ΔR < 3.0). Scores
each candidate triplet with a small MLP and propagates the triplet scores
back to per-track boosts.

Integrated into TrackPreFilter as an optional post-processing step.
Trained jointly — gradients flow through triplet scoring and boost.

Physics constraints (validated on data):
    - Signal triplet invariant mass < 1.777 GeV: 99% of GT triplets pass
    - Charge sum = ±1: 100% of GT triplets (tau → π+π−π+)
    - Max pairwise ΔR < 3.0: 94% of GT triplets
    - Random triplets: only ~16% pass mass cut → strong rejection
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.HierarchicalGraphBackbone import cross_set_knn


class TripletScorer(nn.Module):
    """Score candidate triplets and propagate boosts to per-track scores.

    Args:
        num_anchors: Number of top-scoring tracks to use as anchors (default: 200).
        num_neighbors: kNN neighborhood size per anchor (default: 512).
        mass_threshold: Maximum invariant mass for triplet candidates (default: 1.777 GeV).
        max_delta_r: Maximum pairwise ΔR (default: 3.0).
        hidden_dim: Hidden dimension of triplet scoring MLP (default: 32).
    """

    # Feature indices in triplet_features tensor
    MASS_IDX = 0
    MAX_DR_IDX = 1
    CHARGE_SUM_IDX = 2
    SCORE_SUM_IDX = 3
    TRIPLET_PT_IDX = 4
    MAX_DDXY_IDX = 5

    num_triplet_features = 6

    def __init__(
        self,
        num_anchors: int = 200,
        num_neighbors: int = 512,
        mass_threshold: float = 1.777,
        max_delta_r: float = 3.0,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_neighbors = num_neighbors
        self.mass_threshold = mass_threshold
        self.max_delta_r = max_delta_r

        self.triplet_mlp = nn.Sequential(
            nn.Linear(self.num_triplet_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def compute_invariant_mass(
        self,
        lorentz_vectors: torch.Tensor,
        indices_a: list | torch.Tensor,
        indices_b: list | torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise invariant mass m(a, b).

        m² = (E_a + E_b)² - (px_a + px_b)² - (py_a + py_b)² - (pz_a + pz_b)²

        Args:
            lorentz_vectors: (B, 4, P) raw [px, py, pz, E].
            indices_a, indices_b: Track indices (lists or tensors).

        Returns:
            Invariant mass scalar or tensor.
        """
        lv = lorentz_vectors[0]  # (4, P) — single event
        px_sum = lv[0, indices_a] + lv[0, indices_b]
        py_sum = lv[1, indices_a] + lv[1, indices_b]
        pz_sum = lv[2, indices_a] + lv[2, indices_b]
        e_sum = lv[3, indices_a] + lv[3, indices_b]
        m2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
        return torch.sqrt(m2.clamp(min=1e-8))

    def compute_triplet_mass(
        self,
        lorentz_vectors: torch.Tensor,
        indices_a: list | torch.Tensor,
        indices_b: list | torch.Tensor,
        indices_c: list | torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet invariant mass m(a, b, c)."""
        lv = lorentz_vectors[0]
        px_sum = lv[0, indices_a] + lv[0, indices_b] + lv[0, indices_c]
        py_sum = lv[1, indices_a] + lv[1, indices_b] + lv[1, indices_c]
        pz_sum = lv[2, indices_a] + lv[2, indices_b] + lv[2, indices_c]
        e_sum = lv[3, indices_a] + lv[3, indices_b] + lv[3, indices_c]
        m2 = e_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2
        return torch.sqrt(m2.clamp(min=1e-8))

    def enumerate_triplets(
        self,
        scores: torch.Tensor,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Enumerate candidate triplets using vectorized physics cuts.

        For each anchor (top-A by score), find N nearest neighbors,
        then enumerate all C(N_partners, 2) triplets and filter by
        mass, charge, and ΔR constraints — fully vectorized, no Python loops
        over individual triplets.

        Args:
            scores: (B, P) per-track scores from pre-filter.
            points: (B, 2, P) eta, phi coordinates.
            features: (B, 7, P) raw features (charge at index 5).
            lorentz_vectors: (B, 4, P) raw 4-vectors.
            mask: (B, 1, P) boolean mask.

        Returns:
            all_triplet_indices: (total_triplets, 3) global track indices, or None
            all_triplet_features: (total_triplets, num_triplet_features) physics features, or None
        """
        batch_size = scores.shape[0]
        device = scores.device
        valid_mask = mask.squeeze(1).bool()

        all_triplets = []
        all_features_list = []

        with torch.no_grad():
            for event_idx in range(batch_size):
                ev_scores = scores[event_idx]  # (P,)
                ev_valid = valid_mask[event_idx]  # (P,)
                ev_lv = lorentz_vectors[event_idx]  # (4, P)
                ev_charge = features[event_idx, 5, :]  # (P,)
                ev_eta = points[event_idx, 0, :]  # (P,)
                ev_phi = points[event_idx, 1, :]  # (P,)
                ev_dxy = features[event_idx, 6, :]  # (P,)

                # Select anchors: top-A by score
                masked_scores = ev_scores.clone()
                masked_scores[~ev_valid] = float('-inf')
                num_valid = ev_valid.sum().item()
                actual_anchors = min(self.num_anchors, num_valid)
                if actual_anchors < 1:
                    continue
                _, anchor_indices = masked_scores.topk(actual_anchors)

                for anchor_idx in anchor_indices:
                    anchor = anchor_idx.item()

                    # Find N nearest neighbors in (eta, phi) — vectorized
                    d_eta = ev_eta - ev_eta[anchor]
                    d_phi = (ev_phi - ev_phi[anchor] + math.pi) % (2 * math.pi) - math.pi
                    dr_sq = d_eta**2 + d_phi**2
                    dr_sq[anchor] = float('inf')
                    dr_sq[~ev_valid] = float('inf')

                    actual_neighbors = min(self.num_neighbors, num_valid - 1)
                    if actual_neighbors < 2:
                        continue
                    _, neighbor_indices = dr_sq.topk(actual_neighbors, largest=False)

                    # Pairwise mass: anchor with each neighbor — vectorized
                    a_lv = ev_lv[:, anchor]  # (4,)
                    n_lv = ev_lv[:, neighbor_indices]  # (4, N)

                    pair_e = a_lv[3] + n_lv[3]
                    pair_px = a_lv[0] + n_lv[0]
                    pair_py = a_lv[1] + n_lv[1]
                    pair_pz = a_lv[2] + n_lv[2]
                    pair_m2 = pair_e**2 - pair_px**2 - pair_py**2 - pair_pz**2
                    pair_mass = torch.sqrt(pair_m2.clamp(min=1e-8))

                    # Filter partners by pairwise mass
                    partner_mask = pair_mass < self.mass_threshold
                    partner_local = partner_mask.nonzero(as_tuple=True)[0]
                    num_partners = len(partner_local)

                    if num_partners < 2:
                        continue

                    partner_global = neighbor_indices[partner_local]  # (M,)
                    p_lv = ev_lv[:, partner_global]  # (4, M)
                    p_charge = ev_charge[partner_global]  # (M,)
                    p_eta = ev_eta[partner_global]  # (M,)
                    p_phi = ev_phi[partner_global]  # (M,)
                    p_dxy = ev_dxy[partner_global]  # (M,)
                    p_scores = ev_scores[partner_global]  # (M,)

                    # Generate all C(M, 2) pair combinations — vectorized
                    # triu_indices gives upper triangle indices
                    idx_i, idx_j = torch.triu_indices(num_partners, num_partners, offset=1, device=device)
                    # idx_i, idx_j are local indices into partner_global

                    if len(idx_i) == 0:
                        continue

                    # Triplet 4-vectors: anchor + partner[i] + partner[j]
                    tri_px = a_lv[0] + p_lv[0, idx_i] + p_lv[0, idx_j]
                    tri_py = a_lv[1] + p_lv[1, idx_i] + p_lv[1, idx_j]
                    tri_pz = a_lv[2] + p_lv[2, idx_i] + p_lv[2, idx_j]
                    tri_e = a_lv[3] + p_lv[3, idx_i] + p_lv[3, idx_j]

                    # Triplet mass
                    tri_m2 = tri_e**2 - tri_px**2 - tri_py**2 - tri_pz**2
                    tri_mass = torch.sqrt(tri_m2.clamp(min=1e-8))

                    # Charge sum
                    q_sum = ev_charge[anchor] + p_charge[idx_i] + p_charge[idx_j]

                    # Max pairwise ΔR (3 pairs: a-i, a-j, i-j)
                    def _delta_r(eta_a, phi_a, eta_b, phi_b):
                        de = eta_a - eta_b
                        dp = (phi_a - phi_b + math.pi) % (2 * math.pi) - math.pi
                        return torch.sqrt(de**2 + dp**2 + 1e-8)

                    dr_ai = _delta_r(ev_eta[anchor], ev_phi[anchor], p_eta[idx_i], p_phi[idx_i])
                    dr_aj = _delta_r(ev_eta[anchor], ev_phi[anchor], p_eta[idx_j], p_phi[idx_j])
                    dr_ij = _delta_r(p_eta[idx_i], p_phi[idx_i], p_eta[idx_j], p_phi[idx_j])
                    max_dr = torch.max(torch.max(dr_ai, dr_aj), dr_ij)

                    # Apply all physics cuts — vectorized boolean mask
                    valid_triplets = (
                        (tri_mass < self.mass_threshold)
                        & ((q_sum.abs() - 1.0).abs() < 0.5)  # charge = ±1
                        & (max_dr < self.max_delta_r)
                    )

                    valid_idx = valid_triplets.nonzero(as_tuple=True)[0]
                    if len(valid_idx) == 0:
                        continue

                    # Gather valid triplet indices (global)
                    g_i = partner_global[idx_i[valid_idx]]  # (V,)
                    g_j = partner_global[idx_j[valid_idx]]  # (V,)
                    anchor_col = torch.full_like(g_i, anchor)  # (V,)

                    triplet_idx = torch.stack([anchor_col, g_i, g_j], dim=1)  # (V, 3)

                    # Compute triplet features — vectorized
                    tri_pt = torch.sqrt(tri_px[valid_idx]**2 + tri_py[valid_idx]**2 + 1e-8)
                    score_sum = ev_scores[anchor] + p_scores[idx_i[valid_idx]] + p_scores[idx_j[valid_idx]]

                    ddxy_ai = (ev_dxy[anchor] - p_dxy[idx_i[valid_idx]]).abs()
                    ddxy_aj = (ev_dxy[anchor] - p_dxy[idx_j[valid_idx]]).abs()
                    ddxy_ij = (p_dxy[idx_i[valid_idx]] - p_dxy[idx_j[valid_idx]]).abs()
                    max_ddxy = torch.max(torch.max(ddxy_ai, ddxy_aj), ddxy_ij)

                    triplet_feats = torch.stack([
                        tri_mass[valid_idx],
                        max_dr[valid_idx],
                        q_sum[valid_idx],
                        score_sum,
                        tri_pt,
                        max_ddxy,
                    ], dim=1)  # (V, 6)

                    all_triplets.append(triplet_idx)
                    all_features_list.append(triplet_feats)

        if len(all_triplets) == 0:
            return None, None

        triplet_indices = torch.cat(all_triplets, dim=0)  # (T, 3)
        triplet_features = torch.cat(all_features_list, dim=0)  # (T, 6)

        return triplet_indices, triplet_features

    def score_triplets(
        self,
        triplet_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score triplets with MLP.

        Args:
            triplet_features: (num_triplets, 6) physics features.

        Returns:
            scores: (num_triplets,) triplet scores.
        """
        return self.triplet_mlp(triplet_features).squeeze(-1)

    def propagate_boost(
        self,
        original_scores: torch.Tensor,
        triplet_indices: torch.Tensor | None,
        triplet_scores: torch.Tensor | None,
        boost_weight: float = 1.0,
    ) -> torch.Tensor:
        """Propagate triplet scores back to per-track boosts.

        For each track, boost = max triplet score across all triplets
        containing that track.

        Args:
            original_scores: (B, P) original per-track scores.
            triplet_indices: (num_triplets, 3) track indices, or None.
            triplet_scores: (num_triplets,) scores, or None.
            boost_weight: Scaling factor for boost.

        Returns:
            boosted_scores: (B, P) original + boost.
        """
        if triplet_indices is None or triplet_scores is None:
            return original_scores.clone()

        num_tracks = original_scores.shape[1]

        # For each track, find max triplet score among triplets containing it
        # Build a (num_tracks,) tensor of max triplet scores — out-of-place
        boost = torch.full(
            (num_tracks,), float('-inf'),
            device=original_scores.device, dtype=original_scores.dtype,
        )

        for col in range(3):
            track_ids = triplet_indices[:, col]
            # Out-of-place: create new tensor each iteration to avoid in-place mutation
            new_boost = boost.clone()
            new_boost.scatter_reduce_(
                0, track_ids, triplet_scores, reduce='amax',
            )
            boost = new_boost

        # Replace -inf (tracks in no triplet) with 0
        boost = boost.masked_fill(boost == float('-inf'), 0.0)

        # Apply boost (only to batch 0 for now — extend for multi-batch)
        padded_mask = original_scores[0] == float('-inf')
        boosted = original_scores.clone()
        boosted[0] = original_scores[0] + boost_weight * boost
        boosted[0] = boosted[0].masked_fill(padded_mask, float('-inf'))

        return boosted
