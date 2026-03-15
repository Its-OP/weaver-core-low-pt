"""Tau-origin pion track finder using object condensation.

Top-level module combining:
    1. Pretrained EnrichCompactBackbone (frozen, enrichment only — no compaction)
    2. ObjectCondensationHead (beta + clustering predictions per track)
    3. Object condensation loss (attractive/repulsive potential + beta loss)

The task: find up to 6 pion tracks originating from tau decay among ~1130
tracks per event. The model predicts per-track scores and clustering
coordinates. At inference, tracks are ranked by beta score and the top-K
are returned as tau pion candidates (recall@K evaluation).

Loss components (Kieseler, Eur. Phys. J. C 80, 886, 2020):
    - Attractive potential: pulls GT pion tracks toward the condensation
      point in the learned clustering space.
    - Repulsive potential: pushes non-GT tracks away from condensation point.
    - Beta loss: maximizes condensation point beta, suppresses background beta.

No Hungarian matching needed — all losses are per-track.

Reference:
    https://arxiv.org/abs/2002.03605
"""

import torch
import torch.nn as nn

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.ObjectCondensationHead import ObjectCondensationHead


class TauTrackFinderOC(nn.Module):
    """Object condensation tau-origin pion track finder.

    Forward pass flow:
        1. Backbone enrichment (frozen): all tracks → enriched features (B, 256, P)
        2. OC head: enriched features → beta (B, P) + clustering coords (B, D, P)
        3. Loss: attractive/repulsive potential + beta loss (training only)

    Training mode returns loss dict. Eval mode returns beta scores for ranking.

    Args:
        backbone_kwargs: Keyword arguments for EnrichCompactBackbone.
        head_kwargs: Keyword arguments for ObjectCondensationHead.
        potential_loss_weight: Weight for combined attractive + repulsive
            potential loss (default: 1.0).
        beta_loss_weight: Weight for beta suppression/maximization loss
            (default: 1.0).
        q_min: Minimum charge offset to prevent zero gradients when beta → 0
            (default: 0.1). Charge formula: q_i = arctanh²(β_i) + q_min.
        suppression_weight: Weight for background beta suppression term in
            beta loss (default: 1.0). Controls how aggressively background
            track betas are pushed toward zero.
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        potential_loss_weight: float = 1.0,
        beta_loss_weight: float = 1.0,
        q_min: float = 0.1,
        suppression_weight: float = 1.0,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if head_kwargs is None:
            head_kwargs = {}

        self.potential_loss_weight = potential_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.q_min = q_min
        self.suppression_weight = suppression_weight

        # Build backbone (pretrained weights loaded externally)
        self.backbone = EnrichCompactBackbone(**backbone_kwargs)

        # Freeze backbone — only the head is trained
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Set head's input_dim to match backbone enrichment output
        head_kwargs.setdefault('input_dim', self.backbone.enrichment_output_dim)

        # Build object condensation head
        self.head = ObjectCondensationHead(**head_kwargs)

    def _compute_charge(self, beta: torch.Tensor) -> torch.Tensor:
        """Compute charge from beta scores.

        Charge formula (Kieseler 2020, Eq. 3):
            q_i = arctanh²(β_i) + q_min

        The arctanh² mapping amplifies high-beta tracks (condensation points)
        while keeping low-beta tracks at baseline q_min. This gives the
        condensation point much higher influence in the potential losses.

        Args:
            beta: (B, P) beta scores ∈ (0, 1).

        Returns:
            charge: (B, P) charge values ∈ [q_min, ∞).
        """
        # Clamp beta to (0, 1) exclusive to avoid arctanh(0) = 0 and arctanh(1) = inf
        clamped_beta = beta.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.arctanh(clamped_beta) ** 2 + self.q_min

    def _compute_losses(
        self,
        beta: torch.Tensor,
        clustering_coordinates: torch.Tensor,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute object condensation losses.

        Steps:
            1. Compute charge from beta: q_i = arctanh²(β_i) + q_min
            2. Find condensation point α (highest-charge GT track per event)
            3. Attractive potential: pull GT tracks toward α in clustering space
            4. Repulsive potential: push non-GT tracks away from α
            5. Beta loss: maximize β_α, suppress background β

        All operations are vectorized over the batch and track dimensions.

        Args:
            beta: (B, P) beta scores ∈ (0, 1).
            clustering_coordinates: (B, D, P) learned embedding coordinates.
            track_labels: (B, 1, P) binary labels (1.0 = tau pion).
            mask: (B, 1, P) boolean mask (True = valid track).

        Returns:
            Dict with 'attractive_loss', 'repulsive_loss', 'beta_loss',
            'potential_loss', 'total_loss' (all scalar tensors).
        """
        device = beta.device
        labels_flat = track_labels.squeeze(1)  # (B, P)
        mask_flat = mask.squeeze(1).float()    # (B, P)

        # Apply mask to labels (padded positions are not GT)
        labels_flat = labels_flat * mask_flat

        # Compute charge: q_i = arctanh²(β_i) + q_min
        charge = self._compute_charge(beta)  # (B, P)

        # Separate GT (signal) and non-GT (background) masks
        signal_mask = labels_flat * mask_flat       # (B, P) — 1 for GT pions
        background_mask = (1 - labels_flat) * mask_flat  # (B, P) — 1 for background

        # Count signal and background tracks per event
        num_signal = signal_mask.sum(dim=1).clamp(min=1)          # (B,)
        num_background = background_mask.sum(dim=1).clamp(min=1)  # (B,)
        num_valid = mask_flat.sum(dim=1).clamp(min=1)             # (B,)

        # ---- Find condensation point α per event ----
        # α_k = argmax_i(q_i × M_ik) — highest-charge GT track
        # Since K=1 (one tau per event), we find the single condensation point.
        signal_charge = charge * signal_mask  # (B, P) — zero for non-GT
        condensation_index = signal_charge.argmax(dim=1)  # (B,)

        # Gather condensation point coordinates: x_α
        # clustering_coordinates: (B, D, P)
        condensation_coords = clustering_coordinates.gather(
            2, condensation_index.unsqueeze(1).unsqueeze(2).expand(
                -1, clustering_coordinates.shape[1], -1,
            ),
        ).squeeze(2)  # (B, D)

        # ---- Compute distances in clustering space ----
        # ||x_i - x_α||² for all tracks
        # clustering_coordinates: (B, D, P), condensation_coords: (B, D, 1)
        distance_squared = (
            clustering_coordinates - condensation_coords.unsqueeze(2)
        ).pow(2).sum(dim=1)  # (B, P)

        # ---- Attractive potential ----
        # L_attract = (1/N) × Σ_{i ∈ GT} q_i × ||x_i - x_α||²
        # Pulls GT pion tracks toward the condensation point.
        attractive_per_track = charge * distance_squared * signal_mask  # (B, P)
        attractive_loss = (
            attractive_per_track.sum(dim=1) / num_valid
        ).mean()  # scalar

        # ---- Repulsive potential ----
        # L_repel = (1/N) × Σ_{i ∉ GT} q_i × max(0, 1 - ||x_i - x_α||)²
        # Pushes non-GT tracks away from the condensation point.
        # Hinge at distance 1.0: no gradient once track is far enough.
        distance = distance_squared.sqrt().clamp(min=1e-6)  # (B, P)
        repulsive_hinge = torch.clamp(1.0 - distance, min=0.0) ** 2  # (B, P)
        repulsive_per_track = charge * repulsive_hinge * background_mask  # (B, P)
        repulsive_loss = (
            repulsive_per_track.sum(dim=1) / num_valid
        ).mean()  # scalar

        # ---- Beta loss ----
        # L_beta = (1 - β_α) + s_B × (1/N_bg) × Σ_{i ∉ GT} β_i
        # First term: maximize condensation point beta.
        # Second term: suppress background betas.
        condensation_beta = beta.gather(
            1, condensation_index.unsqueeze(1),
        ).squeeze(1)  # (B,)
        condensation_term = (1.0 - condensation_beta).mean()  # scalar

        background_beta_sum = (beta * background_mask).sum(dim=1)  # (B,)
        suppression_term = (
            background_beta_sum / num_background
        ).mean()  # scalar

        beta_loss = condensation_term + self.suppression_weight * suppression_term

        # ---- Combined losses ----
        potential_loss = attractive_loss + repulsive_loss
        total_loss = (
            self.potential_loss_weight * potential_loss
            + self.beta_loss_weight * beta_loss
        )

        return {
            'attractive_loss': attractive_loss,
            'repulsive_loss': repulsive_loss,
            'potential_loss': potential_loss,
            'beta_loss': beta_loss,
            'total_loss': total_loss,
        }

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: backbone enrichment → OC head → loss or scores.

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) per-track features (standardized).
            lorentz_vectors: (B, 4, P) per-track 4-vectors (raw px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.
            track_labels: (B, 1, P) binary labels. Required for training.
                1.0 = tau-origin pion, 0.0 = background/padding.

        Returns:
            Training: dict with 'total_loss', 'attractive_loss',
                'repulsive_loss', 'potential_loss', 'beta_loss'.
            Inference: dict with 'beta_scores' (B, P).
        """
        # Step 1: Backbone enrichment (frozen, no gradients)
        with torch.no_grad():
            enriched_features = self.backbone.enrich(
                points, features, lorentz_vectors, mask,
            )  # (B, enrichment_output_dim, P)

        # Detach to ensure no gradient computation for backbone
        enriched_features = enriched_features.detach()

        # Step 2: OC head (trainable)
        beta, clustering_coordinates = self.head(enriched_features)

        # Training: compute loss
        if track_labels is not None:
            return self._compute_losses(
                beta, clustering_coordinates, track_labels, mask,
            )

        # Inference: return beta scores for ranking
        return {
            'beta_scores': beta,
        }
