"""Tau-origin pion track finder using DETR-style mask prediction.

Minimal top-level module combining:
    1. Pretrained EnrichCompactBackbone (frozen, enrichment only)
    2. TauTrackFinderHead (decoder + mask scoring + confidence)
    3. Loss: cross-entropy mask + confidence BCE (last layer only)

The task: find up to 6 pion tracks originating from tau decay among ~1130
tracks per event. Each query predicts a soft mask over all tracks and a
confidence score. At inference, queries are ranked by confidence.

Loss components:
    - Cross-entropy mask loss:
        CE = -log(softmax(mask_logits)[gt_track_index])
        Treats "select 1 track from ~1130" as multi-class classification.
    - Confidence BCE:
        Binary cross-entropy for exists/empty on all queries.
        no_object_weight downweights the ~90% empty targets.

References:
    DETR: https://arxiv.org/abs/2005.12872
    Hungarian: Kuhn (1955), Naval Research Logistics Quarterly
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.TauTrackFinderHead import TauTrackFinderHead
from weaver.nn.model.hungarian_matcher import hungarian_matcher


class TauTrackFinder(nn.Module):
    """DETR-style tau-origin pion track finder.

    Forward pass:
        1. Backbone enrichment (frozen): tracks → enriched features (B, 256, P)
        2. Head: decoder cross-attention → mask_logits + confidence_logits
        3. (Training) Hungarian matching → CE mask + confidence BCE

    Args:
        backbone_kwargs: Keyword arguments for EnrichCompactBackbone.
        decoder_kwargs: Keyword arguments for TauTrackFinderHead.
            Must include 'max_gt_tracks' (default: 6).
        mask_ce_loss_weight: Weight for cross-entropy mask loss (default: 2.0).
        confidence_loss_weight: Weight for confidence BCE (default: 2.0).
        per_track_loss_weight: Weight for per-track focal BCE auxiliary loss
            (default: 1.0). This OC-style loss provides direct binary
            classification signal to every track (1130 gradients per event),
            closing the 377× gradient density gap vs query-only training.
        no_object_weight: Weight for empty targets in confidence BCE
            (default: 0.4). Downweights the ~90% of queries that match
            no GT track.
        focal_alpha: Alpha for per-track focal loss (default: 0.75).
            Higher values weight positive (tau) tracks more, compensating
            for ~3/1130 = 0.3% positive rate.
        focal_gamma: Gamma for per-track focal loss modulation (default: 2.0).
            Downweights easy examples where the model is already confident.
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        mask_ce_loss_weight: float = 2.0,
        confidence_loss_weight: float = 2.0,
        per_track_loss_weight: float = 1.0,
        no_object_weight: float = 0.4,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.mask_ce_loss_weight = mask_ce_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        self.per_track_loss_weight = per_track_loss_weight
        self.no_object_weight = no_object_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Maximum number of GT tracks per event
        self.max_gt_tracks = decoder_kwargs.pop('max_gt_tracks', 6)

        # Build backbone (pretrained weights loaded externally)
        self.backbone = EnrichCompactBackbone(**backbone_kwargs)

        # Freeze backbone — only the head is trained
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Set head's backbone_dim to match backbone output
        decoder_kwargs.setdefault('backbone_dim', self.backbone.output_dim)

        # Build decoder head
        self.head = TauTrackFinderHead(**decoder_kwargs)

        # ---- Per-track classification head ----
        # OC-style auxiliary head providing direct per-track supervision.
        # Operates on frozen enriched features via Conv1d MLPs.
        # Produces a per-track logit: sigmoid(logit) = P(tau pion | track).
        # Trained with focal BCE on ALL ~1130 tracks, giving 377× more
        # gradient signal than the 3 Hungarian-matched queries alone.
        backbone_dim = self.backbone.output_dim
        self.per_track_head = nn.Sequential(
            nn.Conv1d(backbone_dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def _focal_bce_loss(
        self,
        predicted_logits: torch.Tensor,
        target_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal binary cross-entropy loss over all valid tracks.

        Focal loss (Lin et al., RetinaNet, ICCV 2017):
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        Focuses training on hard tracks by downweighting easy examples.
        With ~3 positive out of ~1130 tracks, alpha=0.75 upweights positives.

        Args:
            predicted_logits: (B, P) per-track logits (pre-sigmoid).
            target_labels: (B, P) binary labels (1.0 = tau track).
            valid_mask: (B, P) boolean, True for valid tracks.

        Returns:
            Scalar focal BCE loss averaged over valid tracks.
        """
        # Standard per-element BCE (no reduction)
        bce_per_track = functional.binary_cross_entropy_with_logits(
            predicted_logits, target_labels, reduction='none',
        )  # (B, P)

        # p_t = P(correct class): p for positives, 1-p for negatives
        predicted_probabilities = torch.sigmoid(predicted_logits)
        probability_correct = torch.where(
            target_labels == 1.0,
            predicted_probabilities,
            1.0 - predicted_probabilities,
        )

        # alpha_t: class balancing weight
        alpha_weight = torch.where(
            target_labels == 1.0,
            self.focal_alpha,
            1.0 - self.focal_alpha,
        )

        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        # = alpha_t * (1 - p_t)^gamma * BCE
        focal_weight = alpha_weight * ((1.0 - probability_correct) ** self.focal_gamma)
        focal_loss_per_track = focal_weight * bce_per_track  # (B, P)

        # Average over valid tracks only
        focal_loss_per_track = focal_loss_per_track * valid_mask.float()
        num_valid = valid_mask.float().sum().clamp(min=1.0)
        return focal_loss_per_track.sum() / num_valid

    def _extract_ground_truth_indices(
        self,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract up to max_gt_tracks tau-track indices from per-track labels.

        Args:
            track_labels: (B, 1, P) binary labels (1.0 = tau track).
            mask: (B, 1, P) boolean mask (True = valid track).

        Returns:
            ground_truth_indices: (B, max_gt_tracks) — invalid positions are -1.
            ground_truth_count: (B,) number of valid GT tracks per event.
        """
        labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()

        ground_truth_count = labels_flat.sum(dim=1).long().clamp(
            max=self.max_gt_tracks,
        )

        sorted_result = labels_flat.sort(descending=True, stable=True)
        ground_truth_indices = sorted_result.indices[:, :self.max_gt_tracks]

        # Mark positions beyond ground_truth_count as invalid (-1)
        position_range = torch.arange(
            self.max_gt_tracks, device=ground_truth_indices.device,
        ).unsqueeze(0)
        invalid_mask = position_range >= ground_truth_count.unsqueeze(1)
        ground_truth_indices = ground_truth_indices.masked_fill(invalid_mask, -1)

        return ground_truth_indices, ground_truth_count

    def _compute_cost_matrix(
        self,
        mask_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        ground_truth_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Build cost matrix (B, Q, max_gt) for Hungarian matching.

        cost = -log_softmax(mask_logits)[gt_index] + (-log_sigmoid(confidence))
        Invalid GT slots get cost = 1e6 so Hungarian never selects them.
        """
        num_queries = mask_logits.shape[1]

        # Pointer cost: -log P(gt_track | query)
        log_softmax_mask = functional.log_softmax(mask_logits, dim=-1)
        gt_indices_clamped = ground_truth_indices.clamp(min=0)
        gt_indices_expanded = gt_indices_clamped.unsqueeze(1).expand(
            -1, num_queries, -1,
        )
        pointer_cost = -log_softmax_mask.gather(2, gt_indices_expanded)

        # Confidence cost: -log P(exists | query)
        confidence_cost = -functional.logsigmoid(confidence_logits)
        confidence_cost = confidence_cost.unsqueeze(2).expand(
            -1, -1, self.max_gt_tracks,
        )

        # Clamp to finite range
        pointer_cost = torch.nan_to_num(
            pointer_cost, nan=50.0, posinf=100.0,
        ).clamp(max=100.0)
        confidence_cost = torch.nan_to_num(
            confidence_cost, nan=50.0, posinf=100.0,
        ).clamp(max=100.0)

        cost_matrix = pointer_cost + confidence_cost

        # Invalid GT slots → large cost
        invalid_gt_mask = (ground_truth_indices == -1).unsqueeze(1).expand(
            -1, num_queries, -1,
        )
        cost_matrix = cost_matrix.masked_fill(invalid_gt_mask, 1e6)

        return cost_matrix

    def _compute_losses(
        self,
        mask_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        ground_truth_indices: torch.Tensor,
        ground_truth_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute CE mask loss + confidence BCE for the last decoder layer.

        Args:
            mask_logits: (B, num_queries, P) — padded positions are -inf.
            confidence_logits: (B, num_queries).
            ground_truth_indices: (B, max_gt_tracks) — -1 for invalid.
            ground_truth_count: (B,).

        Returns:
            Dict with 'mask_ce_loss' and 'confidence_loss'.
        """
        batch_size = mask_logits.shape[0]
        num_queries = mask_logits.shape[1]
        device = mask_logits.device

        # ---- Hungarian matching ----
        cost_matrix = self._compute_cost_matrix(
            mask_logits, confidence_logits, ground_truth_indices,
        )
        match_indices = hungarian_matcher(cost_matrix.detach())
        matched_query_indices = match_indices[:, 0, :]
        matched_gt_slot_indices = match_indices[:, 1, :]
        matched_gt_track_indices = ground_truth_indices.gather(
            1, matched_gt_slot_indices,
        )
        match_is_valid = matched_gt_track_indices != -1

        # ---- Cross-entropy mask loss ----
        # CE = -log(softmax(mask_logits[query])[gt_track_index])
        num_matched = match_is_valid.sum().item()
        mask_ce_loss = torch.tensor(0.0, device=device)

        if num_matched > 0:
            valid_batch, valid_slot = match_is_valid.nonzero(as_tuple=True)
            valid_queries = matched_query_indices[valid_batch, valid_slot]
            valid_gt_tracks = matched_gt_track_indices[valid_batch, valid_slot]

            matched_logits = mask_logits[valid_batch, valid_queries]  # (M, P)
            mask_ce_loss = functional.cross_entropy(
                matched_logits, valid_gt_tracks.long(),
            )

        # ---- Confidence BCE ----
        confidence_targets = torch.zeros(batch_size, num_queries, device=device)
        if num_matched > 0:
            valid_batch_conf, valid_slot_conf = match_is_valid.nonzero(as_tuple=True)
            valid_queries_conf = matched_query_indices[valid_batch_conf, valid_slot_conf]
            confidence_targets[valid_batch_conf, valid_queries_conf] = 1.0

        confidence_weights = torch.where(
            confidence_targets == 1.0,
            torch.ones_like(confidence_targets),
            torch.full_like(confidence_targets, self.no_object_weight),
        )
        confidence_loss = functional.binary_cross_entropy_with_logits(
            confidence_logits, confidence_targets, weight=confidence_weights,
        )

        return {
            'mask_ce_loss': mask_ce_loss,
            'confidence_loss': confidence_loss,
        }

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: backbone → head → loss (training) or logits (inference).

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) per-track features.
            lorentz_vectors: (B, 4, P) per-track 4-vectors.
            mask: (B, 1, P) boolean mask, True for valid tracks.
            track_labels: (B, 1, P) binary labels. Required for training.

        Returns:
            Training: {'total_loss', 'mask_ce_loss', 'confidence_loss', 'per_track_loss'}
            Inference: {'mask_logits', 'confidence_logits', 'per_track_logits'}
        """
        # Backbone enrichment (frozen)
        with torch.no_grad():
            enriched_features = self.backbone.enrich(
                points, features, lorentz_vectors, mask,
            )
        enriched_features = enriched_features.detach()

        # DETR decoder head — returns (mask_logits, confidence_logits)
        mask_logits, confidence_logits = self.head(
            enriched_features, mask, points,
        )

        # Per-track classification head — direct per-track tau scores
        # per_track_logits: (B, 1, P) → squeeze → (B, P)
        per_track_logits = self.per_track_head(enriched_features).squeeze(1)

        # ---- Training ----
        if track_labels is not None:
            ground_truth_indices, ground_truth_count = (
                self._extract_ground_truth_indices(track_labels, mask)
            )

            # DETR losses (query-based, Hungarian matching)
            detr_losses = self._compute_losses(
                mask_logits, confidence_logits,
                ground_truth_indices, ground_truth_count,
            )

            # Per-track focal BCE loss (OC-style, all 1130 tracks)
            # Provides 377× more gradient signal than query-only training.
            labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()
            valid_mask = mask.squeeze(1).bool()
            per_track_loss = self._focal_bce_loss(
                per_track_logits, labels_flat, valid_mask,
            )

            total_loss = (
                self.mask_ce_loss_weight * detr_losses['mask_ce_loss']
                + self.confidence_loss_weight * detr_losses['confidence_loss']
                + self.per_track_loss_weight * per_track_loss
            )

            return {
                'total_loss': total_loss,
                'mask_ce_loss': detr_losses['mask_ce_loss'],
                'confidence_loss': detr_losses['confidence_loss'],
                'per_track_loss': per_track_loss,
            }

        # ---- Inference ----
        return {
            'mask_logits': mask_logits,
            'confidence_logits': confidence_logits,
            'per_track_logits': per_track_logits,
        }
