"""Tau-origin pion track finder using object condensation.

Top-level module combining:
    1. Pretrained EnrichCompactBackbone (frozen, enrichment only — no compaction)
    2. ObjectCondensationHead (beta + clustering predictions per track)
    3. Object condensation loss (attractive/repulsive potential + beta loss)
       with auxiliary focal BCE for direct per-track classification signal.

The task: find up to 6 pion tracks originating from tau decay among ~1130
tracks per event. The model predicts per-track scores and clustering
coordinates. At inference, tracks are ranked by beta score and the top-K
are returned as tau pion candidates (recall@K evaluation).

Loss components:
    - Focal BCE (Lin et al., RetinaNet, ICCV 2017): direct per-track binary
      classification loss on beta logits. Provides immediate gradient signal
      that tells each track whether it should have high or low beta.
    - Attractive potential (Kieseler 2020): pulls GT pion tracks toward the
      condensation point in the learned clustering space.
    - Repulsive potential (Kieseler 2020): pushes non-GT tracks away from
      condensation point (hinge at configurable radius).
    - Beta loss: maximizes condensation point beta, suppresses background beta.

Events with 0 GT tracks are handled by skipping OC-specific losses
(potential + beta) and only computing focal BCE (all targets = 0).

No Hungarian matching needed — all losses are per-track.

References:
    Kieseler 2020: https://arxiv.org/abs/2002.03605
    RetinaNet: https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.ObjectCondensationHead import ObjectCondensationHead


class TauTrackFinderOC(nn.Module):
    """Object condensation tau-origin pion track finder.

    Forward pass flow:
        1. Backbone enrichment (frozen): all tracks → enriched features (B, 256, P)
        2. OC head: enriched features → beta (B, P) + clustering coords (B, D, P)
        3. Loss: focal BCE + attractive/repulsive potential + beta loss

    Training mode returns loss dict. Eval mode returns beta scores for ranking.

    Args:
        backbone_kwargs: Keyword arguments for EnrichCompactBackbone.
        head_kwargs: Keyword arguments for ObjectCondensationHead.
        focal_bce_weight: Weight for focal BCE classification loss
            (default: 1.0). This is the primary training signal.
        potential_loss_weight: Weight for combined attractive + repulsive
            potential loss (default: 0.01). Kept low because potential
            magnitudes are small relative to classification loss.
        beta_loss_weight: Weight for condensation + suppression beta loss
            (default: 0.01). Kept low to prevent suppression gradient from
            overwhelming the classification signal.
        focal_alpha: Alpha parameter for focal loss class weighting
            (default: 0.75). Higher alpha weights positive (GT) tracks more,
            compensating for ~3/1130 = 0.27% positive rate.
        focal_gamma: Gamma parameter for focal loss modulation
            (default: 2.0). Downweights easy examples.
        q_min: Minimum charge offset to prevent zero gradients when beta → 0
            (default: 0.1). Charge formula: q_i = arctanh²(β_i) + q_min.
        suppression_weight: Weight for background beta suppression term in
            beta loss (default: 0.01). Reduced from 1.0 to prevent ~1127
            background gradients from overwhelming ~3 signal gradients.
        repulsive_hinge_radius: Distance threshold for repulsive hinge loss
            (default: 3.0). Increased from 1.0 because random 8D coordinates
            have mean distance ~2.8, so radius=1.0 produces zero repulsive
            gradient at initialization.
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        focal_bce_weight: float = 1.0,
        potential_loss_weight: float = 0.01,
        beta_loss_weight: float = 0.01,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        q_min: float = 0.1,
        suppression_weight: float = 0.01,
        repulsive_hinge_radius: float = 3.0,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if head_kwargs is None:
            head_kwargs = {}

        self.focal_bce_weight = focal_bce_weight
        self.potential_loss_weight = potential_loss_weight
        self.beta_loss_weight = beta_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.q_min = q_min
        self.suppression_weight = suppression_weight
        self.repulsive_hinge_radius = repulsive_hinge_radius

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

    def _focal_bce_loss(
        self,
        beta_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal binary cross-entropy over all tracks.

        Focal loss (Lin et al., RetinaNet, ICCV 2017) applied per-track:
            FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

        For each track:
            - Signal tracks (label=1): α_t = focal_alpha, p_t = sigmoid(logit)
            - Background tracks (label=0): α_t = 1 - focal_alpha, p_t = 1 - sigmoid(logit)

        This provides the PRIMARY training signal — direct per-track
        classification that tells each track whether it should have
        high or low beta.

        Args:
            beta_logits: (B, P) pre-sigmoid logits from beta head.
            labels: (B, P) binary labels (1.0 = tau pion, 0.0 = background).
            mask: (B, P) float mask (1.0 = valid track, 0.0 = padding).

        Returns:
            Scalar focal BCE loss (averaged over valid tracks).
        """
        # Per-track BCE (no reduction)
        # BCE = -[y × log(σ(x)) + (1-y) × log(1 - σ(x))]
        bce_per_track = functional.binary_cross_entropy_with_logits(
            beta_logits, labels, reduction='none',
        )  # (B, P)

        # p_t = probability of correct class
        probabilities = torch.sigmoid(beta_logits)
        probability_correct = torch.where(
            labels == 1.0, probabilities, 1.0 - probabilities,
        )  # (B, P)

        # α_t = class-balancing weight
        alpha_weight = torch.where(
            labels == 1.0,
            self.focal_alpha,
            1.0 - self.focal_alpha,
        )  # (B, P)

        # FL(p_t) = α_t × (1 - p_t)^γ × BCE
        focal_modulation = alpha_weight * (
            (1.0 - probability_correct) ** self.focal_gamma
        )
        focal_loss_per_track = focal_modulation * bce_per_track  # (B, P)

        # Average over valid tracks only (exclude padding)
        masked_loss = (focal_loss_per_track * mask).sum()
        num_valid = mask.sum().clamp(min=1.0)
        return masked_loss / num_valid

    def _compute_losses(
        self,
        beta: torch.Tensor,
        beta_logits: torch.Tensor,
        clustering_coordinates: torch.Tensor,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses: focal BCE + OC potential + beta loss.

        The focal BCE is always computed (even for 0-GT events).
        The OC losses (potential + beta) are only computed for events
        with at least 1 GT track. Events with 0 GT are skipped for OC
        losses to avoid the condensation point selecting a random track.

        Args:
            beta: (B, P) beta scores ∈ (0, 1) after sigmoid.
            beta_logits: (B, P) pre-sigmoid logits from beta head.
            clustering_coordinates: (B, D, P) learned embedding coordinates.
            track_labels: (B, 1, P) binary labels (1.0 = tau pion).
            mask: (B, 1, P) boolean mask (True = valid track).

        Returns:
            Dict with 'focal_bce_loss', 'attractive_loss', 'repulsive_loss',
            'potential_loss', 'beta_loss', 'total_loss' (all scalar tensors).
        """
        device = beta.device
        labels_flat = track_labels.squeeze(1)  # (B, P)
        mask_flat = mask.squeeze(1).float()    # (B, P)

        # Apply mask to labels (padded positions are not GT)
        labels_flat = labels_flat * mask_flat

        # ---- Focal BCE loss (always computed, primary signal) ----
        focal_bce_loss = self._focal_bce_loss(
            beta_logits, labels_flat, mask_flat,
        )

        # ---- Identify events with GT tracks ----
        signal_mask = labels_flat * mask_flat       # (B, P) — 1 for GT pions
        background_mask = (1 - labels_flat) * mask_flat  # (B, P) — 1 for background
        num_signal_per_event = signal_mask.sum(dim=1)  # (B,)
        has_gt = num_signal_per_event > 0  # (B,) boolean

        # If no events have GT tracks, skip OC losses entirely
        if not has_gt.any():
            zero = torch.tensor(0.0, device=device)
            total_loss = self.focal_bce_weight * focal_bce_loss
            return {
                'focal_bce_loss': focal_bce_loss,
                'attractive_loss': zero,
                'repulsive_loss': zero,
                'potential_loss': zero,
                'beta_loss': zero,
                'total_loss': total_loss,
            }

        # ---- Compute OC losses only for events with GT ----
        # Compute charge: q_i = arctanh²(β_i) + q_min
        charge = self._compute_charge(beta)  # (B, P)

        # Find condensation point α per event (highest-charge GT track)
        # α = argmax_i(q_i × signal_mask_i)
        signal_charge = charge * signal_mask  # (B, P)
        condensation_index = signal_charge.argmax(dim=1)  # (B,)

        # Gather condensation point coordinates: x_α ∈ R^D
        # clustering_coordinates: (B, D, P)
        condensation_coords = clustering_coordinates.gather(
            2, condensation_index.unsqueeze(1).unsqueeze(2).expand(
                -1, clustering_coordinates.shape[1], -1,
            ),
        ).squeeze(2)  # (B, D)

        # Compute distances in clustering space: ||x_i - x_α||
        # clustering_coordinates: (B, D, P), condensation_coords: (B, D, 1)
        distance_squared = (
            clustering_coordinates - condensation_coords.unsqueeze(2)
        ).pow(2).sum(dim=1)  # (B, P)

        # Safe distance: add epsilon BEFORE sqrt to avoid NaN gradient.
        # d/dx sqrt(x) = 1/(2*sqrt(x)) → inf when x→0.
        # Adding epsilon inside avoids this: sqrt(x + eps) has bounded gradient.
        distance = (distance_squared + 1e-6).sqrt()  # (B, P)

        # Event mask for averaging: only count events with GT
        # has_gt: (B,) boolean → (B, 1) float for broadcasting
        event_weight = has_gt.float()  # (B,)
        num_gt_events = event_weight.sum().clamp(min=1.0)

        num_valid = mask_flat.sum(dim=1).clamp(min=1)  # (B,)

        # ---- Attractive potential ----
        # L_attract = (1/N) × Σ_{i ∈ GT} q_i × ||x_i - x_α||²
        # Pulls GT pion tracks toward the condensation point.
        attractive_per_track = charge * distance_squared * signal_mask  # (B, P)
        attractive_per_event = attractive_per_track.sum(dim=1) / num_valid  # (B,)
        attractive_loss = (attractive_per_event * event_weight).sum() / num_gt_events

        # ---- Repulsive potential ----
        # L_repel = (1/N) × Σ_{i ∉ GT} q_i × max(0, R - ||x_i - x_α||)²
        # Pushes non-GT tracks away from the condensation point.
        # Hinge at configurable radius R (default 3.0, increased from 1.0
        # because random 8D coordinates have mean distance ~2.8).
        repulsive_hinge = torch.clamp(
            self.repulsive_hinge_radius - distance, min=0.0,
        ) ** 2  # (B, P)
        repulsive_per_track = charge * repulsive_hinge * background_mask  # (B, P)
        repulsive_per_event = repulsive_per_track.sum(dim=1) / num_valid  # (B,)
        repulsive_loss = (repulsive_per_event * event_weight).sum() / num_gt_events

        # ---- Beta loss ----
        # L_beta = (1 - β_α) + s_B × (1/N_bg) × Σ_{i ∉ GT} β_i
        # First term: maximize condensation point beta (only for GT events).
        # Second term: suppress background betas (weighted by s_B).
        condensation_beta = beta.gather(
            1, condensation_index.unsqueeze(1),
        ).squeeze(1)  # (B,)
        condensation_term = (
            ((1.0 - condensation_beta) * event_weight).sum() / num_gt_events
        )

        num_background = background_mask.sum(dim=1).clamp(min=1)  # (B,)
        background_mean_beta = (beta * background_mask).sum(dim=1) / num_background
        suppression_term = (
            (background_mean_beta * event_weight).sum() / num_gt_events
        )

        beta_loss = condensation_term + self.suppression_weight * suppression_term

        # ---- Combined losses ----
        potential_loss = attractive_loss + repulsive_loss
        total_loss = (
            self.focal_bce_weight * focal_bce_loss
            + self.potential_loss_weight * potential_loss
            + self.beta_loss_weight * beta_loss
        )

        return {
            'focal_bce_loss': focal_bce_loss,
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
            Training: dict with 'total_loss', 'focal_bce_loss',
                'attractive_loss', 'repulsive_loss', 'potential_loss',
                'beta_loss'.
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
        beta, beta_logits, clustering_coordinates = self.head(enriched_features)

        # Training: compute loss
        if track_labels is not None:
            return self._compute_losses(
                beta, beta_logits, clustering_coordinates, track_labels, mask,
            )

        # Inference: return beta scores for ranking
        return {
            'beta_scores': beta,
        }
