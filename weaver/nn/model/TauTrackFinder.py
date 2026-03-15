"""Tau-origin pion track finder using DETR-style mask prediction with DN-DETR
denoising training.

Top-level module combining:
    1. Pretrained EnrichCompactBackbone (frozen, no gradients)
    2. TauTrackFinderHead (encoder + dual-cross-attention decoder + mask + confidence)
    3. DN-DETR denoising training (training only)
    4. Auxiliary losses at every decoder layer
    5. Loss computation: dice + focal BCE for masks, BCE for confidence

The task: find up to 6 pion tracks originating from tau decay among ~1130
tracks per event. Uses learned queries (over-prediction) with mask-based
output. Each query predicts a binary mask over all tracks and a confidence
score. At inference, queries are ranked by confidence.

Loss components:
    - Dice loss (Milletari et al., V-Net, 3DV 2016):
        dice_loss = 1 - (2 * sum(p * g) + eps) / (sum(p) + sum(g) + eps)
        Measures overlap between predicted and ground-truth binary masks.
    - Focal binary cross-entropy (Lin et al., RetinaNet, ICCV 2017):
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Applied per-track, then averaged. Focuses training on hard tracks.
    - Confidence BCE (Carion et al., DETR, ECCV 2020):
        Binary cross-entropy for exists/empty on all queries.
        Uses no-object coefficient to downweight empty targets.
    - DN-DETR denoising (Li et al., CVPR 2022):
        Noised ground-truth features as additional decoder queries with
        known targets. Stabilizes bipartite matching and accelerates
        convergence. No Hungarian matching needed for denoising queries.
    - Auxiliary losses (Carion et al., DETR, ECCV 2020):
        Losses computed at every decoder layer, not just the last.
        Provides direct supervision to intermediate representations.

References:
    DETR: https://arxiv.org/abs/2005.12872
    RetinaNet: https://arxiv.org/abs/1708.02002
    DN-DETR: https://arxiv.org/abs/2203.01305
    V-Net: https://arxiv.org/abs/1606.04797
    Hungarian: Kuhn (1955), Naval Research Logistics Quarterly
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.TauTrackFinderHead import TauTrackFinderHead
from weaver.nn.model.hungarian_matcher import hungarian_matcher


class TauTrackFinder(nn.Module):
    """DETR-style tau-origin pion track finder with mask prediction, DN-DETR
    denoising, and auxiliary per-layer losses.

    Forward pass flow:
        1. Backbone enrichment (frozen): all tracks -> enriched features (B, 256, P)
        2. Backbone compaction (frozen): enriched -> compact tokens (B, 256, 128)
        3. (Training) Build denoising queries from noised GT track features
        4. Head forward: per-layer mask_logits (B, Q, P) + confidence_logits (B, Q)
        5. (Training) Hungarian matching + auxiliary losses at each decoder layer
        6. (Training) Denoising losses (no matching needed)

    Training mode returns loss dict. Inference mode returns logits dict.

    Args:
        backbone_kwargs: Keyword arguments for EnrichCompactBackbone.
        decoder_kwargs: Keyword arguments for TauTrackFinderHead.
            Must include 'max_gt_tracks' (default: 6).
        dice_loss_weight: Weight for dice loss component (default: 2.0).
        focal_bce_loss_weight: Weight for focal BCE loss component (default: 5.0).
        confidence_loss_weight: Weight for confidence BCE loss (default: 2.0).
        focal_alpha: Alpha parameter for focal loss (default: 0.25).
            Balances positive vs negative class contributions.
        focal_gamma: Gamma parameter for focal loss (default: 2.0).
            Controls focusing strength on hard examples.
        no_object_weight: Weight for empty/no-object targets in confidence BCE
            (default: 0.4). With 15 queries and ~3 GT tracks, 80% of targets
            are empty. Downweighting prevents the model from trivially minimizing
            loss by always predicting "no object".
        num_denoising_groups: Number of denoising groups G for DN-DETR training
            (default: 5). Each group contains max_gt_tracks noised queries.
        denoising_noise_scale: Standard deviation of Gaussian noise added to
            GT track features for denoising queries (default: 0.5).
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
        dice_loss_weight: float = 2.0,
        focal_bce_loss_weight: float = 5.0,
        confidence_loss_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        no_object_weight: float = 0.4,
        num_denoising_groups: int = 5,
        denoising_noise_scale: float = 0.5,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.dice_loss_weight = dice_loss_weight
        self.focal_bce_loss_weight = focal_bce_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.no_object_weight = no_object_weight
        self.num_denoising_groups = num_denoising_groups
        self.denoising_noise_scale = denoising_noise_scale

        # Maximum number of GT tracks per event
        self.max_gt_tracks = decoder_kwargs.pop('max_gt_tracks', 6)

        # Build backbone (pretrained weights loaded externally)
        self.backbone = EnrichCompactBackbone(**backbone_kwargs)

        # Freeze backbone — only the head is trained
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Set head's backbone_dim to match backbone output
        decoder_kwargs.setdefault('backbone_dim', self.backbone.output_dim)

        # Build encoder-decoder head
        self.head = TauTrackFinderHead(**decoder_kwargs)

        # ---- Denoising projection ----
        # Projects GT track features (backbone_dim) + Gaussian noise into
        # the decoder query space (decoder_dim) for DN-DETR denoising training.
        # The projection allows noised GT features to serve as decoder queries
        # with known targets, bypassing the need for Hungarian matching.
        backbone_dim = self.backbone.output_dim
        decoder_dim = self.head.decoder_dim
        self.denoising_projection = nn.Linear(backbone_dim, decoder_dim)

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
                ground_truth_indices: (B, max_gt_tracks) indices of GT tracks.
                    Positions beyond ground_truth_count are set to -1.
                ground_truth_count: (B,) number of valid GT tracks per event.
        """
        # Mask out padding positions so they are not counted as GT tracks
        labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()  # (B, P)

        ground_truth_count = labels_flat.sum(dim=1).long().clamp(
            max=self.max_gt_tracks,
        )  # (B,)

        # Sort descending so positive labels come first, then take top-k
        sorted_result = labels_flat.sort(descending=True, stable=True)
        ground_truth_indices = sorted_result.indices[
            :, :self.max_gt_tracks
        ]  # (B, max_gt)

        # Mark positions beyond per-event ground_truth_count as invalid (-1)
        position_range = torch.arange(
            self.max_gt_tracks, device=ground_truth_indices.device,
        ).unsqueeze(0)  # (1, max_gt)
        invalid_mask = position_range >= ground_truth_count.unsqueeze(1)  # (B, max_gt)
        ground_truth_indices = ground_truth_indices.masked_fill(invalid_mask, -1)

        return ground_truth_indices, ground_truth_count

    def _build_denoising_queries(
        self,
        enriched_features: torch.Tensor,
        ground_truth_indices: torch.Tensor,
        ground_truth_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create DN-DETR denoising queries from GT track features + Gaussian noise.

        DN-DETR (Li et al., CVPR 2022) adds noised GT features as extra decoder
        queries during training. Since their target assignments are known a priori,
        no Hungarian matching is needed. This stabilizes the matching-based loss
        and accelerates convergence.

        Steps:
            1. Gather enriched features at GT track indices: (B, max_gt, backbone_dim)
            2. Repeat for G denoising groups: (B, G * max_gt, backbone_dim)
            3. Add Gaussian noise: noise ~ N(0, sigma^2) where sigma = denoising_noise_scale
            4. Project through denoising_projection: (B, G * max_gt, decoder_dim)
            5. Create denoising targets:
               - mask_targets: binary mask over P tracks with 1 at the GT track index
               - confidence_targets: 1.0 for valid GT, 0.0 for padding
            6. Build attention mask preventing cross-contamination between:
               - Learnable queries and denoising queries (bidirectional)
               - Different denoising groups (prevents information leakage)

        Args:
            enriched_features: (B, backbone_dim, P) enriched per-track features
                from the backbone.
            ground_truth_indices: (B, max_gt_tracks) GT track indices (-1 = invalid).
            ground_truth_count: (B,) number of valid GT tracks per event.

        Returns:
            Tuple of:
                denoising_queries: (B, G * max_gt, decoder_dim)
                denoising_mask_targets: (B, G * max_gt, P) binary masks
                denoising_confidence_targets: (B, G * max_gt) binary
                denoising_valid_mask: (B, G * max_gt) which queries are valid
                attention_mask: (num_learnable + G*max_gt, num_learnable + G*max_gt)
                    boolean self-attention mask (True = blocked position)
        """
        batch_size = enriched_features.shape[0]
        backbone_dim = enriched_features.shape[1]
        num_tracks = enriched_features.shape[2]
        device = enriched_features.device
        num_groups = self.num_denoising_groups
        max_gt = self.max_gt_tracks
        num_learnable = self.head.num_queries

        # Step 1: Gather enriched features at GT track indices
        # Replace -1 (invalid) with 0 for safe gather; these will be masked later
        safe_indices = ground_truth_indices.clamp(min=0)  # (B, max_gt)

        # Expand for gathering from (B, backbone_dim, P):
        # indices shape: (B, backbone_dim, max_gt)
        gather_indices = safe_indices.unsqueeze(1).expand(
            -1, backbone_dim, -1,
        )  # (B, backbone_dim, max_gt)
        ground_truth_features = enriched_features.gather(
            2, gather_indices,
        )  # (B, backbone_dim, max_gt)

        # Transpose to (B, max_gt, backbone_dim) for processing
        ground_truth_features = ground_truth_features.transpose(1, 2)  # (B, max_gt, backbone_dim)

        # Step 2: Repeat for G denoising groups
        # (B, max_gt, backbone_dim) -> (B, G * max_gt, backbone_dim)
        repeated_features = ground_truth_features.repeat(
            1, num_groups, 1,
        )  # (B, G * max_gt, backbone_dim)

        # Step 3: Add Gaussian noise
        # noise ~ N(0, sigma^2) where sigma = denoising_noise_scale
        gaussian_noise = torch.randn_like(repeated_features) * self.denoising_noise_scale
        noised_features = repeated_features + gaussian_noise

        # Step 4: Project into decoder query space
        # denoising_projection: (B, G * max_gt, backbone_dim) -> (B, G * max_gt, decoder_dim)
        denoising_queries = self.denoising_projection(noised_features)

        # Step 5a: Create denoising mask targets
        # For each denoising query, build a binary mask over P tracks
        # with 1 at the corresponding GT track index and 0 elsewhere
        # Repeat GT indices for all groups: (B, G * max_gt)
        repeated_gt_indices = ground_truth_indices.repeat(
            1, num_groups,
        )  # (B, G * max_gt)

        denoising_mask_targets = torch.zeros(
            batch_size, num_groups * max_gt, num_tracks,
            device=device, dtype=enriched_features.dtype,
        )  # (B, G * max_gt, P)

        # Build valid mask: a denoising query is valid if its GT index is not -1
        denoising_valid_mask = repeated_gt_indices != -1  # (B, G * max_gt)

        # Set mask target to 1 at GT track positions for valid queries
        valid_gt_positions = repeated_gt_indices.clamp(min=0)  # (B, G * max_gt)
        # scatter_ requires (B, G*max_gt, 1) index tensor
        denoising_mask_targets.scatter_(
            2,
            valid_gt_positions.unsqueeze(2),
            denoising_valid_mask.unsqueeze(2).float(),
        )

        # Step 5b: Create denoising confidence targets
        # 1.0 for valid GT queries, 0.0 for padding queries
        denoising_confidence_targets = denoising_valid_mask.float()  # (B, G * max_gt)

        # Step 6: Build attention mask for self-attention
        # The attention mask prevents information leakage between:
        # (a) Learnable queries and denoising queries (bidirectional block)
        # (b) Different denoising groups (each group is independent)
        #
        # Layout: [learnable queries | denoising group 0 | group 1 | ... | group G-1]
        # Total sequence length = num_learnable + G * max_gt
        total_sequence_length = num_learnable + num_groups * max_gt

        # Initialize attention mask: True = blocked (cannot attend)
        # PyTorch nn.TransformerDecoder uses additive mask where True = -inf
        attention_mask = torch.zeros(
            total_sequence_length, total_sequence_length,
            device=device, dtype=torch.bool,
        )

        # Block learnable queries from seeing denoising queries
        attention_mask[
            :num_learnable, num_learnable:
        ] = True

        # Block denoising queries from seeing learnable queries
        attention_mask[
            num_learnable:, :num_learnable
        ] = True

        # Block different denoising groups from seeing each other
        # Each denoising group can only attend to queries within the same group
        for group_index in range(num_groups):
            group_start = num_learnable + group_index * max_gt
            group_end = group_start + max_gt

            # Block this group from seeing all denoising queries outside its range
            # Before this group
            if group_start > num_learnable:
                attention_mask[
                    group_start:group_end, num_learnable:group_start
                ] = True
            # After this group
            if group_end < total_sequence_length:
                attention_mask[
                    group_start:group_end, group_end:
                ] = True

        return (
            denoising_queries,
            denoising_mask_targets,
            denoising_confidence_targets,
            denoising_valid_mask,
            attention_mask,
        )

    def _dice_loss(
        self,
        predicted_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dice loss for a single query-GT pair.

        Dice loss (Milletari et al., V-Net, 3DV 2016) measures set overlap:
            dice_loss = 1 - (2 * sum(p * g) + epsilon) / (sum(p) + sum(g) + epsilon)

        where:
            p = sigmoid(predicted_logits) in [0, 1]
            g = binary target mask in {0, 1}
            epsilon = 1e-6 for numerical stability

        The loss is 0 when prediction perfectly matches the target (full overlap)
        and approaches 1 when there is no overlap.

        Args:
            predicted_mask: (P,) logits (pre-sigmoid) for one query over P tracks.
            target_mask: (P,) binary {0, 1} ground-truth mask.

        Returns:
            Scalar dice loss value.
        """
        epsilon = 1e-6

        # Apply sigmoid to convert logits to probabilities
        predicted_probabilities = torch.sigmoid(predicted_mask)  # (P,)

        # dice_loss = 1 - (2 * sum(p * g) + eps) / (sum(p) + sum(g) + eps)
        intersection = (predicted_probabilities * target_mask).sum()
        denominator = predicted_probabilities.sum() + target_mask.sum()

        dice_score = (2.0 * intersection + epsilon) / (denominator + epsilon)
        return 1.0 - dice_score

    def _focal_bce_loss(
        self,
        predicted_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal binary cross-entropy averaged over tracks.

        Focal loss (Lin et al., RetinaNet, ICCV 2017) applied per-track:
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        For each track position:
            - Positive tracks (g=1): alpha_t = focal_alpha, p_t = sigmoid(logit)
            - Negative tracks (g=0): alpha_t = 1 - focal_alpha, p_t = 1 - sigmoid(logit)

        The (1 - p_t)^gamma modulating factor downweights easy examples where
        the model is already confident, focusing training on hard tracks.

        Args:
            predicted_mask: (P,) logits (pre-sigmoid) for one query over P tracks.
            target_mask: (P,) binary {0, 1} ground-truth mask.

        Returns:
            Scalar focal BCE loss value (averaged over tracks).
        """
        # Compute per-track probabilities
        predicted_probabilities = torch.sigmoid(predicted_mask)  # (P,)

        # Compute per-track binary cross-entropy (no reduction)
        # BCE = -[g * log(p) + (1-g) * log(1-p)]
        bce_per_track = functional.binary_cross_entropy_with_logits(
            predicted_mask, target_mask, reduction='none',
        )  # (P,)

        # Compute p_t: probability of the correct class
        # For positive tracks (g=1): p_t = p
        # For negative tracks (g=0): p_t = 1 - p
        probability_correct = torch.where(
            target_mask == 1.0,
            predicted_probabilities,
            1.0 - predicted_probabilities,
        )  # (P,)

        # Compute alpha_t: class-balancing weight
        # For positive tracks (g=1): alpha_t = focal_alpha
        # For negative tracks (g=0): alpha_t = 1 - focal_alpha
        alpha_weight = torch.where(
            target_mask == 1.0,
            torch.tensor(self.focal_alpha, device=predicted_mask.device, dtype=predicted_mask.dtype),
            torch.tensor(1.0 - self.focal_alpha, device=predicted_mask.device, dtype=predicted_mask.dtype),
        )  # (P,)

        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        # Since bce_per_track = -log(p_t), focal_loss = alpha_t * (1 - p_t)^gamma * bce
        focal_weight = alpha_weight * ((1.0 - probability_correct) ** self.focal_gamma)
        focal_loss_per_track = focal_weight * bce_per_track  # (P,)

        # Average over tracks
        return focal_loss_per_track.mean()

    def _compute_cost_matrix(
        self,
        mask_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        ground_truth_indices: torch.Tensor,
        ground_truth_count: torch.Tensor,
    ) -> torch.Tensor:
        """Build cost matrix (B, num_queries, max_gt_tracks) for Hungarian matching.

        For each query q and GT slot g:
            - Build GT mask: 1 at ground_truth_indices[g], 0 elsewhere
            - Pointer cost: -sum(gt_mask * log_sigmoid(mask_logits))
              (negative log-probability only at the GT track position)
            - Confidence cost: -log(sigmoid(confidence_logits[q]))

        Cost = pointer_cost + confidence_cost

        Invalid GT slots (index == -1) receive cost = 1e6 so Hungarian
        matching never selects them.

        Args:
            mask_logits: (B, num_queries, P) mask prediction logits.
            confidence_logits: (B, num_queries) confidence scores.
            ground_truth_indices: (B, max_gt_tracks) indices of GT tracks (-1 = invalid).
            ground_truth_count: (B,) number of valid GT tracks per event.

        Returns:
            cost_matrix: (B, num_queries, max_gt_tracks) with large cost for
                invalid GT slots.
        """
        batch_size = mask_logits.shape[0]
        num_queries = mask_logits.shape[1]

        # Compute log-sigmoid of mask logits for pointer cost
        # log_sigmoid(x) = log(sigmoid(x)), numerically stable via PyTorch
        log_sigmoid_mask = functional.logsigmoid(mask_logits)  # (B, num_queries, P)

        # Gather log-sigmoid probability at each GT track position
        # gt_indices: (B, max_gt) -> expand to (B, num_queries, max_gt)
        gt_indices_clamped = ground_truth_indices.clamp(min=0)  # Replace -1 with 0 for gather
        gt_indices_expanded = gt_indices_clamped.unsqueeze(1).expand(
            -1, num_queries, -1,
        )  # (B, num_queries, max_gt)

        # Pointer cost: -sum(gt_mask * log_sigmoid(mask_logits))
        # Since gt_mask is 1 only at one position, this reduces to:
        # -log_sigmoid(mask_logits[q, gt_index])
        pointer_cost = -log_sigmoid_mask.gather(
            2, gt_indices_expanded,
        )  # (B, num_queries, max_gt)

        # Confidence cost: -log(sigmoid(confidence_logits[q]))
        confidence_log_probability = functional.logsigmoid(
            confidence_logits,
        )  # (B, num_queries)
        confidence_cost = -confidence_log_probability.unsqueeze(2).expand(
            -1, -1, self.max_gt_tracks,
        )  # (B, num_queries, max_gt)

        # Clamp costs to prevent inf/NaN in the cost matrix.
        # logsigmoid can produce very negative values (e.g., -100 for logits of -100),
        # and -logsigmoid can be very large. Clamp to a finite maximum.
        pointer_cost = pointer_cost.clamp(max=100.0)
        confidence_cost = confidence_cost.clamp(max=100.0)

        # Combined cost
        cost_matrix = pointer_cost + confidence_cost

        # Set cost to large value for invalid GT slots (gt_index == -1)
        # so Hungarian never matches them.
        # Use a value that is large but finite to avoid scipy errors.
        invalid_gt_mask = (ground_truth_indices == -1).unsqueeze(1).expand(
            -1, num_queries, -1,
        )  # (B, num_queries, max_gt)
        cost_matrix = cost_matrix.masked_fill(invalid_gt_mask, 1e6)

        return cost_matrix

    def _compute_losses_single_layer(
        self,
        mask_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        ground_truth_indices: torch.Tensor,
        ground_truth_count: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute dice + focal BCE + confidence losses for one decoder layer's output.

        Steps:
            1. Build cost matrix, run Hungarian matching (bipartite assignment)
            2. For each matched query-GT pair:
               - Build GT binary mask (1 at GT track index, 0 elsewhere)
               - Compute dice_loss(query_mask_logits, gt_mask)
               - Compute focal_bce_loss(query_mask_logits, gt_mask)
            3. Confidence BCE with no_object_weight for unmatched queries

        Args:
            mask_logits: (B, num_queries, P) mask prediction logits.
            confidence_logits: (B, num_queries) confidence scores.
            ground_truth_indices: (B, max_gt_tracks) GT track indices (-1 = invalid).
            ground_truth_count: (B,) number of valid GT tracks per event.
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            Dict with 'mask_dice_loss', 'mask_focal_bce_loss', 'confidence_loss'.
        """
        batch_size = mask_logits.shape[0]
        num_queries = mask_logits.shape[1]
        num_tracks = mask_logits.shape[2]
        device = mask_logits.device

        # Step 1: Build cost matrix and run Hungarian matching
        cost_matrix = self._compute_cost_matrix(
            mask_logits, confidence_logits, ground_truth_indices, ground_truth_count,
        )  # (B, num_queries, max_gt_tracks)

        # hungarian_matcher: (B, num_queries, max_gt) -> (B, 2, max_gt)
        # indices[:, 0] = matched query indices
        # indices[:, 1] = matched GT slot indices
        match_indices = hungarian_matcher(cost_matrix.detach())
        matched_query_indices = match_indices[:, 0, :]   # (B, max_gt)
        matched_gt_slot_indices = match_indices[:, 1, :]  # (B, max_gt)

        # Identify which matched pairs are real (not empty GT slots)
        matched_gt_track_indices = ground_truth_indices.gather(
            1, matched_gt_slot_indices,
        )  # (B, max_gt): actual track index for each match
        match_is_valid = matched_gt_track_indices != -1  # (B, max_gt)

        # Step 2: Mask losses (dice + focal BCE) — only on validly matched queries
        total_dice_loss = torch.tensor(0.0, device=device)
        total_focal_bce_loss = torch.tensor(0.0, device=device)
        num_matched_total = 0

        for batch_index in range(batch_size):
            for match_slot in range(self.max_gt_tracks):
                if not match_is_valid[batch_index, match_slot]:
                    continue

                query_index = matched_query_indices[batch_index, match_slot]
                target_track_index = matched_gt_track_indices[batch_index, match_slot]

                # Build GT binary mask: 1 at GT track position, 0 elsewhere
                gt_binary_mask = torch.zeros(
                    num_tracks, device=device, dtype=mask_logits.dtype,
                )
                gt_binary_mask[target_track_index] = 1.0

                # Get this query's mask logits
                query_mask_logits = mask_logits[batch_index, query_index]  # (P,)

                # Compute dice loss for this query-GT pair
                total_dice_loss = total_dice_loss + self._dice_loss(
                    query_mask_logits, gt_binary_mask,
                )

                # Compute focal BCE loss for this query-GT pair
                total_focal_bce_loss = total_focal_bce_loss + self._focal_bce_loss(
                    query_mask_logits, gt_binary_mask,
                )

                num_matched_total += 1

        # Average over matched pairs (avoid division by zero)
        if num_matched_total > 0:
            mask_dice_loss = total_dice_loss / num_matched_total
            mask_focal_bce_loss = total_focal_bce_loss / num_matched_total
        else:
            mask_dice_loss = torch.tensor(0.0, device=device)
            mask_focal_bce_loss = torch.tensor(0.0, device=device)

        # Step 3: Confidence BCE loss — all queries
        # Build confidence targets: 1 for queries matched to real GT, 0 otherwise
        confidence_targets = torch.zeros(
            batch_size, num_queries, device=device,
        )
        for batch_index in range(batch_size):
            for match_slot in range(self.max_gt_tracks):
                if match_is_valid[batch_index, match_slot]:
                    query_index = matched_query_indices[batch_index, match_slot]
                    confidence_targets[batch_index, query_index] = 1.0

        # No-object coefficient (Carion et al., DETR, ECCV 2020):
        # Downweight the "empty" class in confidence BCE to prevent the model
        # from trivially minimizing loss by always predicting "no object".
        #
        # Weight scheme:
        #   - Matched queries (exists):  weight = 1.0
        #   - Unmatched queries (empty): weight = no_object_weight
        confidence_weights = torch.where(
            confidence_targets == 1.0,
            torch.ones_like(confidence_targets),
            torch.full_like(confidence_targets, self.no_object_weight),
        )
        confidence_loss = functional.binary_cross_entropy_with_logits(
            confidence_logits, confidence_targets, weight=confidence_weights,
        )

        return {
            'mask_dice_loss': mask_dice_loss,
            'mask_focal_bce_loss': mask_focal_bce_loss,
            'confidence_loss': confidence_loss,
        }

    def _compute_denoising_losses(
        self,
        denoising_mask_logits: torch.Tensor,
        denoising_confidence_logits: torch.Tensor,
        denoising_mask_targets: torch.Tensor,
        denoising_confidence_targets: torch.Tensor,
        denoising_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for denoising queries (no Hungarian matching needed).

        DN-DETR denoising queries have known target assignments, so losses are
        computed directly without bipartite matching.

        For each valid denoising query:
            - dice_loss(predicted_mask, target_mask)
            - focal_bce_loss(predicted_mask, target_mask)
            - BCE for confidence (target = 1.0 for valid, 0.0 for padding)

        Args:
            denoising_mask_logits: (B, G * max_gt, P) mask logits for denoising queries.
            denoising_confidence_logits: (B, G * max_gt) confidence logits.
            denoising_mask_targets: (B, G * max_gt, P) binary mask targets.
            denoising_confidence_targets: (B, G * max_gt) binary confidence targets.
            denoising_valid_mask: (B, G * max_gt) which queries are valid.

        Returns:
            Dict with 'denoising_dice_loss', 'denoising_focal_bce_loss',
            'denoising_confidence_loss'.
        """
        batch_size = denoising_mask_logits.shape[0]
        num_denoising_queries = denoising_mask_logits.shape[1]
        device = denoising_mask_logits.device

        # Mask losses: only for valid denoising queries
        total_dice_loss = torch.tensor(0.0, device=device)
        total_focal_bce_loss = torch.tensor(0.0, device=device)
        num_valid_total = 0

        for batch_index in range(batch_size):
            for query_index in range(num_denoising_queries):
                if not denoising_valid_mask[batch_index, query_index]:
                    continue

                query_mask_logits = denoising_mask_logits[
                    batch_index, query_index
                ]  # (P,)
                query_mask_targets = denoising_mask_targets[
                    batch_index, query_index
                ]  # (P,)

                total_dice_loss = total_dice_loss + self._dice_loss(
                    query_mask_logits, query_mask_targets,
                )
                total_focal_bce_loss = total_focal_bce_loss + self._focal_bce_loss(
                    query_mask_logits, query_mask_targets,
                )
                num_valid_total += 1

        # Average over valid denoising queries
        if num_valid_total > 0:
            denoising_dice_loss = total_dice_loss / num_valid_total
            denoising_focal_bce_loss = total_focal_bce_loss / num_valid_total
        else:
            denoising_dice_loss = torch.tensor(0.0, device=device)
            denoising_focal_bce_loss = torch.tensor(0.0, device=device)

        # Confidence BCE for all denoising queries (valid and padding)
        # Valid queries have target = 1.0, padding queries have target = 0.0
        # Use no_object_weight for padding queries to match the learnable query behavior
        confidence_weights = torch.where(
            denoising_confidence_targets == 1.0,
            torch.ones_like(denoising_confidence_targets),
            torch.full_like(denoising_confidence_targets, self.no_object_weight),
        )
        denoising_confidence_loss = functional.binary_cross_entropy_with_logits(
            denoising_confidence_logits,
            denoising_confidence_targets,
            weight=confidence_weights,
        )

        return {
            'denoising_dice_loss': denoising_dice_loss,
            'denoising_focal_bce_loss': denoising_focal_bce_loss,
            'denoising_confidence_loss': denoising_confidence_loss,
        }

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: backbone -> head -> loss (training) or logits (inference).

        Training flow:
            1. Backbone enrich + compact (frozen, no_grad, detach)
            2. Build denoising queries from GT track features + noise
            3. Head forward with denoising queries and attention mask:
               returns per-layer mask_logits and confidence_logits
            4. For each decoder layer: compute matching losses (auxiliary losses)
            5. Sum all layer losses equally (uniform layer weighting)
            6. Add denoising losses
            7. Return loss dict

        Inference flow:
            1. Backbone enrich + compact
            2. Head forward (no denoising)
            3. Return last layer's mask_logits and confidence_logits

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) per-track features (standardized).
            lorentz_vectors: (B, 4, P) per-track 4-vectors (raw px, py, pz, E).
            mask: (B, 1, P) boolean mask, True for valid tracks.
            track_labels: (B, 1, P) binary labels. Required for training.
                1.0 = tau-origin pion, 0.0 = background/padding.

        Returns:
            Training: dict with scalar tensors:
                - 'total_loss': weighted sum of all components
                - 'mask_dice_loss': dice loss averaged across layers
                - 'mask_focal_bce_loss': focal BCE loss averaged across layers
                - 'confidence_loss': confidence loss averaged across layers
                - 'denoising_loss': total denoising contribution (or 0)
            Inference: dict with:
                - 'mask_logits': (B, num_queries, P)
                - 'confidence_logits': (B, num_queries)
        """
        # Step 1: Backbone enrichment (frozen, no gradients)
        with torch.no_grad():
            enriched_features = self.backbone.enrich(
                points, features, lorentz_vectors, mask,
            )  # (B, backbone_dim, P)

        # Detach to ensure no gradient computation for backbone
        enriched_features = enriched_features.detach()

        # Step 2: Backbone compaction (frozen, no gradients)
        # ALL tracks enter compaction (no masking -- unlike pretraining)
        with torch.no_grad():
            compact_tokens, _ = self.backbone.compact(
                points, enriched_features, mask,
            )  # (B, backbone_dim, 128)

        compact_tokens = compact_tokens.detach()

        # ---- Training mode ----
        if track_labels is not None:
            # Extract GT track indices
            ground_truth_indices, ground_truth_count = self._extract_ground_truth_indices(
                track_labels, mask,
            )

            # Build DN-DETR denoising queries from noised GT features
            (
                denoising_queries,
                denoising_mask_targets,
                denoising_confidence_targets,
                denoising_valid_mask,
                attention_mask,
            ) = self._build_denoising_queries(
                enriched_features, ground_truth_indices, ground_truth_count,
            )

            # Head forward with denoising queries and attention mask
            # Returns per-layer outputs for auxiliary losses
            num_denoising = self.num_denoising_groups * self.max_gt_tracks
            head_output = self.head(
                enriched_features, compact_tokens, mask, points,
                num_denoising_queries=num_denoising,
                denoising_query_features=denoising_queries,
                denoising_attention_mask=attention_mask,
            )

            # head returns (all_layer_mask_logits, all_layer_confidence_logits)
            # all_layer_mask_logits: list of (B, num_learnable + G*max_gt, P)
            # all_layer_confidence_logits: list of (B, num_learnable + G*max_gt)
            all_layer_mask_logits, all_layer_confidence_logits = head_output
            num_learnable = self.head.num_queries
            num_layers = len(all_layer_mask_logits)

            # Accumulate losses across all decoder layers (auxiliary losses)
            accumulated_dice_loss = torch.tensor(0.0, device=enriched_features.device)
            accumulated_focal_bce_loss = torch.tensor(0.0, device=enriched_features.device)
            accumulated_confidence_loss = torch.tensor(0.0, device=enriched_features.device)

            for layer_index in range(num_layers):
                layer_mask_logits = all_layer_mask_logits[layer_index]
                layer_confidence_logits = all_layer_confidence_logits[layer_index]

                # Split learnable and denoising outputs
                learnable_mask_logits = layer_mask_logits[:, :num_learnable, :]
                learnable_confidence_logits = layer_confidence_logits[:, :num_learnable]

                # Compute losses for learnable queries (with Hungarian matching)
                layer_losses = self._compute_losses_single_layer(
                    learnable_mask_logits,
                    learnable_confidence_logits,
                    ground_truth_indices,
                    ground_truth_count,
                    mask,
                )

                accumulated_dice_loss = accumulated_dice_loss + layer_losses['mask_dice_loss']
                accumulated_focal_bce_loss = accumulated_focal_bce_loss + layer_losses['mask_focal_bce_loss']
                accumulated_confidence_loss = accumulated_confidence_loss + layer_losses['confidence_loss']

            # Average across layers (uniform layer weighting)
            averaged_dice_loss = accumulated_dice_loss / num_layers
            averaged_focal_bce_loss = accumulated_focal_bce_loss / num_layers
            averaged_confidence_loss = accumulated_confidence_loss / num_layers

            # Compute denoising losses using the last decoder layer's output
            last_layer_mask_logits = all_layer_mask_logits[-1]
            last_layer_confidence_logits = all_layer_confidence_logits[-1]
            denoising_mask_logits = last_layer_mask_logits[:, num_learnable:, :]
            denoising_confidence_logits = last_layer_confidence_logits[:, num_learnable:]

            denoising_losses = self._compute_denoising_losses(
                denoising_mask_logits,
                denoising_confidence_logits,
                denoising_mask_targets,
                denoising_confidence_targets,
                denoising_valid_mask,
            )

            # Weighted denoising loss
            denoising_total = (
                self.dice_loss_weight * denoising_losses['denoising_dice_loss']
                + self.focal_bce_loss_weight * denoising_losses['denoising_focal_bce_loss']
                + self.confidence_loss_weight * denoising_losses['denoising_confidence_loss']
            )

            # Total loss: weighted sum of learnable + denoising losses
            total_loss = (
                self.dice_loss_weight * averaged_dice_loss
                + self.focal_bce_loss_weight * averaged_focal_bce_loss
                + self.confidence_loss_weight * averaged_confidence_loss
                + denoising_total
            )

            return {
                'total_loss': total_loss,
                'mask_dice_loss': averaged_dice_loss,
                'mask_focal_bce_loss': averaged_focal_bce_loss,
                'confidence_loss': averaged_confidence_loss,
                'denoising_loss': denoising_total,
            }

        # ---- Inference mode ----
        # Head forward without denoising queries
        head_output = self.head(
            enriched_features, compact_tokens, mask, points,
        )

        # Head returns (all_layer_mask_logits, all_layer_confidence_logits);
        # use only the last layer's predictions for inference
        all_layer_mask_logits, all_layer_confidence_logits = head_output
        last_mask_logits = all_layer_mask_logits[-1]
        last_confidence_logits = all_layer_confidence_logits[-1]

        return {
            'mask_logits': last_mask_logits,
            'confidence_logits': last_confidence_logits,
        }
