"""Tau-origin pion track finder using DETR-style over-prediction.

Top-level module combining:
    1. Pretrained EnrichCompactBackbone (frozen, no gradients)
    2. TauTrackFinderHead (encoder + decoder + pointer + confidence)
    3. Loss computation (Hungarian matching + focal pointer + confidence BCE)

The task: find up to 6 pion tracks originating from tau decay among ~1130
tracks per event. Uses 15 learned queries (over-prediction) — excess queries
learn to predict empty (∅). At inference, queries are ranked by confidence.

Loss components:
    - Pointer focal loss (Lin et al., ICCV 2017, RetinaNet):
        For each matched query, focal cross-entropy over ~1130 tracks.
        FL(p_t) = -α (1 - p_t)^γ log(p_t), α=0.25, γ=2.0
    - Confidence BCE loss (Carion et al., ECCV 2020, DETR):
        Binary cross-entropy for exists/∅ on all 15 queries.
    - Hungarian matching (Kuhn, 1955):
        Optimal 1-to-1 assignment of queries to GT tracks (no gradients).
        Reuses existing hungarian_matcher from weaver.

References:
    DETR: https://arxiv.org/abs/2005.12872
    RetinaNet: https://arxiv.org/abs/1708.02002
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

    Forward pass flow:
        1. Backbone enrichment (frozen): all tracks → enriched features (B, 256, P)
        2. Backbone compaction (frozen): enriched → compact tokens (B, 256, 128)
        3. Head forward: pointer logits (B, 15, P) + confidence logits (B, 15)
        4. Loss: Hungarian matching → focal pointer + confidence BCE

    Training mode returns loss dict. Eval mode returns logits dict.

    Args:
        backbone_kwargs: Keyword arguments for EnrichCompactBackbone.
        decoder_kwargs: Keyword arguments for TauTrackFinderHead.
            Must include 'max_gt_tracks' (default: 6).
        pointer_loss_weight: Weight for pointer focal loss (default: 5.0).
        confidence_loss_weight: Weight for confidence BCE loss (default: 1.0).
        focal_alpha: Alpha parameter for focal loss (default: 0.25).
        focal_gamma: Gamma parameter for focal loss (default: 2.0).
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        pointer_loss_weight: float = 5.0,
        confidence_loss_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.pointer_loss_weight = pointer_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Maximum number of GT tracks per event
        self.max_gt_tracks = decoder_kwargs.pop('max_gt_tracks', 6)

        # Build backbone (pretrained weights loaded externally)
        self.backbone = EnrichCompactBackbone(**backbone_kwargs)

        # Freeze backbone — only the head is trained
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Set head's backbone_dim to match backbone output
        decoder_kwargs.setdefault('backbone_dim', self.backbone.output_dim)

        # Build encoder-decoder head
        self.head = TauTrackFinderHead(**decoder_kwargs)

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
            Tuple of:
                gt_indices: (B, max_gt_tracks) indices of GT tracks.
                    Positions beyond gt_count are set to -1.
                gt_count: (B,) number of valid GT tracks per event.
        """
        # Mask out padding positions so they aren't counted as GT tracks
        labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()  # (B, P)

        gt_count = labels_flat.sum(dim=1).long().clamp(
            max=self.max_gt_tracks,
        )  # (B,)

        # Sort descending so positive labels come first, then take top-k
        sorted_result = labels_flat.sort(descending=True, stable=True)
        gt_indices = sorted_result.indices[:, :self.max_gt_tracks]  # (B, max_gt)

        # Mark positions beyond per-event gt_count as invalid (-1)
        position_range = torch.arange(
            self.max_gt_tracks, device=gt_indices.device,
        ).unsqueeze(0)  # (1, max_gt)
        invalid_mask = position_range >= gt_count.unsqueeze(1)  # (B, max_gt)
        gt_indices = gt_indices.masked_fill(invalid_mask, -1)

        return gt_indices, gt_count

    def _compute_cost_matrix(
        self,
        pointer_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        gt_indices: torch.Tensor,
        gt_count: torch.Tensor,
    ) -> torch.Tensor:
        """Build cost matrix (B, num_queries, max_gt_tracks) for Hungarian matching.

        Cost = λ_ptr × (-log P(query → GT track)) + λ_conf × (-log P(exists))

        Args:
            pointer_logits: (B, num_queries, P) pointer scores.
            confidence_logits: (B, num_queries) confidence scores.
            gt_indices: (B, max_gt_tracks) indices of GT tracks (-1 = invalid).
            gt_count: (B,) number of valid GT tracks per event.

        Returns:
            cost_matrix: (B, num_queries, max_gt_tracks) with large cost for
                invalid GT slots.
        """
        batch_size = pointer_logits.shape[0]
        num_queries = pointer_logits.shape[1]

        # Softmax over tracks for each query → P(query → track)
        pointer_probabilities = functional.softmax(
            pointer_logits, dim=2,
        )  # (B, num_queries, P)

        # Gather probability of each query pointing to each GT track
        # gt_indices: (B, max_gt) → expand to (B, num_queries, max_gt)
        gt_indices_clamped = gt_indices.clamp(min=0)  # Replace -1 with 0 for gather
        gt_indices_expanded = gt_indices_clamped.unsqueeze(1).expand(
            -1, num_queries, -1,
        )  # (B, num_queries, max_gt)

        pointer_cost = -torch.log(
            pointer_probabilities.gather(2, gt_indices_expanded).clamp(min=1e-8)
        )  # (B, num_queries, max_gt): -log P(query q → GT track g)

        # Confidence cost: -log P(query exists)
        # σ(confidence_logits) = P(exists)
        confidence_probability = torch.sigmoid(confidence_logits)  # (B, num_queries)
        confidence_cost = -torch.log(
            confidence_probability.clamp(min=1e-8)
        ).unsqueeze(2).expand(
            -1, -1, self.max_gt_tracks,
        )  # (B, num_queries, max_gt)

        # Combined cost
        cost_matrix = pointer_cost + confidence_cost

        # Set cost to large value for invalid GT slots (gt_index == -1)
        # so Hungarian never matches them
        invalid_gt_mask = (gt_indices == -1).unsqueeze(1).expand(
            -1, num_queries, -1,
        )  # (B, num_queries, max_gt)
        cost_matrix = cost_matrix.masked_fill(invalid_gt_mask, 1e6)

        return cost_matrix

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal cross-entropy loss over tracks for one query-GT pair.

        Focal loss (Lin et al., RetinaNet, ICCV 2017):
            FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

        where p_t = softmax(logits)[target_index] and α_t is the class weight.
        The (1 - p_t)^γ modulating factor downweights easy examples (where the
        model is already confident), focusing training on hard examples.

        Args:
            logits: (P,) raw pointer logits for one query over P tracks.
            targets: scalar index of the correct track.

        Returns:
            Scalar focal loss value.
        """
        # log_softmax for numerical stability: log P(track_i) for all tracks
        log_probabilities = functional.log_softmax(logits, dim=0)  # (P,)
        probabilities = log_probabilities.exp()  # (P,)

        # p_t = P(correct track)
        log_probability_target = log_probabilities[targets]
        probability_target = probabilities[targets]

        # FL = -α (1 - p_t)^γ log(p_t)
        focal_weight = self.focal_alpha * (
            (1.0 - probability_target) ** self.focal_gamma
        )
        return -focal_weight * log_probability_target

    def _compute_losses(
        self,
        pointer_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        gt_indices: torch.Tensor,
        gt_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute pointer focal loss and confidence BCE loss.

        Steps:
            1. Build cost matrix (B, num_queries, max_gt_tracks)
            2. Hungarian matching → optimal query-to-GT assignment
            3. Pointer focal loss on matched queries only
            4. Confidence BCE loss on all queries

        Args:
            pointer_logits: (B, num_queries, P) pointer scores.
            confidence_logits: (B, num_queries) confidence scores.
            gt_indices: (B, max_gt_tracks) GT track indices (-1 = invalid).
            gt_count: (B,) number of valid GT tracks per event.

        Returns:
            Dict with 'pointer_focal_loss', 'confidence_bce_loss', 'total_loss'.
        """
        batch_size = pointer_logits.shape[0]
        num_queries = pointer_logits.shape[1]
        device = pointer_logits.device

        # Step 1: Build cost matrix and run Hungarian matching
        cost_matrix = self._compute_cost_matrix(
            pointer_logits, confidence_logits, gt_indices, gt_count,
        )  # (B, num_queries, max_gt_tracks)

        # hungarian_matcher: (B, num_queries, max_gt) → (B, 2, max_gt)
        # indices[:, 0] = matched query indices
        # indices[:, 1] = matched GT slot indices
        match_indices = hungarian_matcher(cost_matrix.detach())
        matched_query_indices = match_indices[:, 0, :]   # (B, max_gt)
        matched_gt_slot_indices = match_indices[:, 1, :]  # (B, max_gt)

        # Step 2: Identify which matched pairs are real (not empty GT slots)
        # A match is "real" if gt_indices[b, matched_gt_slot[b, k]] != -1
        matched_gt_track_indices = gt_indices.gather(
            1, matched_gt_slot_indices,
        )  # (B, max_gt): actual track index for each match
        match_is_valid = matched_gt_track_indices != -1  # (B, max_gt)

        # Step 3: Pointer focal loss — only on validly matched queries
        total_pointer_loss = torch.tensor(0.0, device=device)
        num_matched_total = 0

        for batch_index in range(batch_size):
            for match_slot in range(self.max_gt_tracks):
                if not match_is_valid[batch_index, match_slot]:
                    continue

                query_index = matched_query_indices[batch_index, match_slot]
                target_track = matched_gt_track_indices[batch_index, match_slot]

                # Focal loss for this query pointing to this GT track
                query_logits = pointer_logits[batch_index, query_index]  # (P,)
                total_pointer_loss = total_pointer_loss + self._focal_loss(
                    query_logits, target_track,
                )
                num_matched_total += 1

        # Average over matched pairs (avoid division by zero)
        if num_matched_total > 0:
            pointer_focal_loss = total_pointer_loss / num_matched_total
        else:
            pointer_focal_loss = torch.tensor(0.0, device=device)

        # Step 4: Confidence BCE loss — all queries
        # Build confidence targets: 1 for queries matched to real GT, 0 otherwise
        confidence_targets = torch.zeros(
            batch_size, num_queries, device=device,
        )
        for batch_index in range(batch_size):
            for match_slot in range(self.max_gt_tracks):
                if match_is_valid[batch_index, match_slot]:
                    query_index = matched_query_indices[batch_index, match_slot]
                    confidence_targets[batch_index, query_index] = 1.0

        confidence_bce_loss = functional.binary_cross_entropy_with_logits(
            confidence_logits, confidence_targets,
        )

        # Step 5: Weighted total loss
        total_loss = (
            self.pointer_loss_weight * pointer_focal_loss
            + self.confidence_loss_weight * confidence_bce_loss
        )

        return {
            'pointer_focal_loss': pointer_focal_loss,
            'confidence_bce_loss': confidence_bce_loss,
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
        """Forward pass: backbone → head → loss (training) or logits (inference).

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) per-track features (standardized).
            lorentz_vectors: (B, 4, P) per-track 4-vectors (raw px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.
            track_labels: (B, 1, P) binary labels. Required for training.
                1.0 = tau-origin pion, 0.0 = background/padding.

        Returns:
            Training: dict with 'total_loss', 'pointer_focal_loss',
                'confidence_bce_loss' (all scalar tensors).
            Inference: dict with 'pointer_logits' (B, num_queries, P) and
                'confidence_logits' (B, num_queries).
        """
        # Step 1: Backbone enrichment (frozen, no gradients)
        with torch.no_grad():
            enriched_features = self.backbone.enrich(
                points, features, lorentz_vectors, mask,
            )  # (B, 256, P)

        # Detach to ensure no gradient computation for backbone
        enriched_features = enriched_features.detach()

        # Step 2: Backbone compaction (frozen, no gradients)
        # ALL tracks enter compaction (no masking — unlike pretraining)
        with torch.no_grad():
            compact_tokens, _ = self.backbone.compact(
                points, enriched_features, mask,
            )  # (B, 256, 128)

        compact_tokens = compact_tokens.detach()

        # Step 3: Head forward (trainable)
        pointer_logits, confidence_logits = self.head(
            enriched_features, compact_tokens, mask,
        )

        # Training: compute loss
        if track_labels is not None:
            gt_indices, gt_count = self._extract_ground_truth_indices(
                track_labels, mask,
            )
            return self._compute_losses(
                pointer_logits, confidence_logits, gt_indices, gt_count,
            )

        # Inference: return logits
        return {
            'pointer_logits': pointer_logits,
            'confidence_logits': confidence_logits,
        }
