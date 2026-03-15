"""Tau-origin pion track finder using DETR-style mask prediction with DN-DETR
denoising training.

Top-level module combining:
    1. Pretrained EnrichCompactBackbone (frozen, no gradients)
    2. TauTrackFinderHead (encoder + dual-cross-attention decoder + mask + confidence)
    3. DN-DETR denoising training (training only)
    4. Auxiliary losses at every decoder layer
    5. Loss computation: cross-entropy for masks, BCE for confidence

The task: find up to 6 pion tracks originating from tau decay among ~1130
tracks per event. Uses learned queries (over-prediction) with mask-based
output. Each query predicts a binary mask over all tracks and a confidence
score. At inference, queries are ranked by confidence.

Loss components:
    - Cross-entropy mask loss (track selection as classification):
        CE = -log(softmax(mask_logits)[gt_track_index])
        Treats "select 1 track from ~1130" as multi-class classification.
        Padded tracks have logits = -inf → softmax gives 0 probability.

        Dice loss was previously used but is degenerate for single-track-
        positive masks: with 1 positive out of ~1130 tracks, dice ≈ 0.998
        for random predictions and gradients scale as O(1/P²), providing
        nearly zero learning signal.

    - Confidence BCE (Carion et al., DETR, ECCV 2020):
        Binary cross-entropy for exists/empty on all queries.
        Uses no-object coefficient to downweight empty targets.
    - DN-DETR denoising (Li et al., CVPR 2022):
        Noised ground-truth features as additional decoder queries with
        known targets. Stabilizes bipartite matching and accelerates
        convergence. No Hungarian matching needed for denoising queries.
        Applied at every decoder layer (same as learnable losses) and
        scaled by denoising_loss_weight for balanced gradient contribution.
    - Auxiliary losses (Carion et al., DETR, ECCV 2020):
        Losses computed at every decoder layer, not just the last.
        Provides direct supervision to intermediate representations.

References:
    DETR: https://arxiv.org/abs/2005.12872
    DN-DETR: https://arxiv.org/abs/2203.01305
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
        mask_ce_loss_weight: Weight for cross-entropy mask loss (default: 2.0).
            Applied to both learnable and denoising mask losses.
        confidence_loss_weight: Weight for confidence BCE loss (default: 2.0).
        denoising_loss_weight: Global scale for all denoising loss components
            (default: 1.0). Denoising losses are computed at every decoder
            layer and averaged, then scaled by this factor relative to
            the learnable losses.
        no_object_weight: Weight for empty/no-object targets in confidence BCE
            (default: 0.4). With 30 queries and ~3 GT tracks, ~90% of targets
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
        mask_ce_loss_weight: float = 2.0,
        confidence_loss_weight: float = 2.0,
        denoising_loss_weight: float = 1.0,
        no_object_weight: float = 0.4,
        num_denoising_groups: int = 5,
        denoising_noise_scale: float = 0.5,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}

        self.mask_ce_loss_weight = mask_ce_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        self.denoising_loss_weight = denoising_loss_weight
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

    def _compute_cost_matrix(
        self,
        mask_logits: torch.Tensor,
        confidence_logits: torch.Tensor,
        ground_truth_indices: torch.Tensor,
        ground_truth_count: torch.Tensor,
    ) -> torch.Tensor:
        """Build cost matrix (B, num_queries, max_gt_tracks) for Hungarian matching.

        For each query q and GT slot g:
            - Pointer cost: -log_softmax(mask_logits)[q, gt_index]
              Uses softmax (not sigmoid) so the cost captures competition
              between tracks — a query that spreads probability evenly gets
              high cost even if it assigns some probability to the GT track.
            - Confidence cost: -log(sigmoid(confidence_logits[q]))

        Cost = pointer_cost + confidence_cost

        Invalid GT slots (index == -1) receive cost = 1e6 so Hungarian
        matching never selects them.

        Args:
            mask_logits: (B, num_queries, P) mask prediction logits.
                Padded positions have logits = -inf → softmax gives 0.
            confidence_logits: (B, num_queries) confidence scores.
            ground_truth_indices: (B, max_gt_tracks) indices of GT tracks (-1 = invalid).
            ground_truth_count: (B,) number of valid GT tracks per event.

        Returns:
            cost_matrix: (B, num_queries, max_gt_tracks) with large cost for
                invalid GT slots.
        """
        batch_size = mask_logits.shape[0]
        num_queries = mask_logits.shape[1]

        # Compute log-softmax of mask logits for pointer cost
        # log_softmax treats track selection as multi-class classification,
        # matching the cross-entropy training loss.
        # Padded tracks with logits = -inf get log_softmax = -inf (probability 0).
        log_softmax_mask = functional.log_softmax(
            mask_logits, dim=-1,
        )  # (B, num_queries, P)

        # Gather log-softmax probability at each GT track position
        # gt_indices: (B, max_gt) -> expand to (B, num_queries, max_gt)
        gt_indices_clamped = ground_truth_indices.clamp(min=0)  # Replace -1 with 0 for gather
        gt_indices_expanded = gt_indices_clamped.unsqueeze(1).expand(
            -1, num_queries, -1,
        )  # (B, num_queries, max_gt)

        # Pointer cost: -log_softmax(mask_logits)[q, gt_index]
        # Measures how much probability mass the query places on the GT track
        # relative to all other tracks (captures inter-track competition).
        pointer_cost = -log_softmax_mask.gather(
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
        pointer_cost = torch.nan_to_num(pointer_cost, nan=50.0, posinf=100.0).clamp(max=100.0)
        confidence_cost = torch.nan_to_num(confidence_cost, nan=50.0, posinf=100.0).clamp(max=100.0)

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
        precomputed_match: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute cross-entropy mask loss + confidence BCE for one decoder layer.

        Uses cross-entropy over tracks for the mask loss:
            CE = -log(softmax(mask_logits)[gt_track_index])

        This treats track selection as multi-class classification, which is
        the natural formulation when each GT mask has exactly 1 positive
        track out of ~1130. Padded tracks have logits = -inf, so softmax
        assigns them zero probability automatically.

        Accepts optional pre-computed Hungarian matching to avoid redundant
        CPU round-trips when computing auxiliary losses across decoder layers.

        Args:
            mask_logits: (B, num_queries, P) mask prediction logits.
                Padded positions have logits = -inf.
            confidence_logits: (B, num_queries) confidence scores.
            ground_truth_indices: (B, max_gt_tracks) GT track indices (-1 = invalid).
            ground_truth_count: (B,) number of valid GT tracks per event.
            mask: (B, 1, P) boolean mask, True for valid tracks.
            precomputed_match: Optional tuple of (matched_query_indices,
                matched_gt_track_indices, match_is_valid) to skip Hungarian.

        Returns:
            Tuple of (loss_dict, match_tuple):
                loss_dict: 'mask_ce_loss', 'confidence_loss'
                match_tuple: (matched_query_indices, matched_gt_track_indices,
                    match_is_valid) for reuse by other layers.
        """
        batch_size = mask_logits.shape[0]
        num_queries = mask_logits.shape[1]
        device = mask_logits.device

        # Step 1: Hungarian matching (or reuse pre-computed)
        if precomputed_match is not None:
            matched_query_indices, matched_gt_track_indices, match_is_valid = precomputed_match
        else:
            cost_matrix = self._compute_cost_matrix(
                mask_logits, confidence_logits,
                ground_truth_indices, ground_truth_count,
            )
            match_indices = hungarian_matcher(cost_matrix.detach())
            matched_query_indices = match_indices[:, 0, :]  # (B, max_gt)
            matched_gt_slot_indices = match_indices[:, 1, :]
            matched_gt_track_indices = ground_truth_indices.gather(
                1, matched_gt_slot_indices,
            )
            match_is_valid = matched_gt_track_indices != -1

        match_tuple = (matched_query_indices, matched_gt_track_indices, match_is_valid)

        # Step 2: Cross-entropy mask loss
        # For each matched (query, GT track) pair, compute:
        #   CE = -log(softmax(mask_logits[query])[gt_track_index])
        # Padded tracks have logits = -inf → softmax gives 0 (correct).
        num_matched_total = match_is_valid.sum().item()
        mask_ce_loss = torch.tensor(0.0, device=device)

        if num_matched_total > 0:
            # Flatten valid matches: collect (batch_idx, query_idx, gt_track_idx)
            valid_batch_indices, valid_slot_indices = match_is_valid.nonzero(as_tuple=True)
            valid_query_indices = matched_query_indices[valid_batch_indices, valid_slot_indices]
            valid_gt_tracks = matched_gt_track_indices[valid_batch_indices, valid_slot_indices]

            # Gather matched query logits: (num_matched, P)
            matched_logits = mask_logits[valid_batch_indices, valid_query_indices]  # (M, P)

            # Cross-entropy: softmax over P tracks, NLL at GT track position
            # F.cross_entropy handles log_softmax + NLL in a numerically stable way.
            # Padded tracks (logits = -inf) get softmax = 0 automatically.
            mask_ce_loss = functional.cross_entropy(
                matched_logits, valid_gt_tracks.long(),
            )

        # Step 3: Confidence BCE — vectorized
        confidence_targets = torch.zeros(batch_size, num_queries, device=device)
        if num_matched_total > 0:
            valid_batch_indices_conf, valid_slot_indices_conf = match_is_valid.nonzero(as_tuple=True)
            valid_query_indices_conf = matched_query_indices[valid_batch_indices_conf, valid_slot_indices_conf]
            confidence_targets[valid_batch_indices_conf, valid_query_indices_conf] = 1.0

        confidence_weights = torch.where(
            confidence_targets == 1.0,
            torch.ones_like(confidence_targets),
            torch.full_like(confidence_targets, self.no_object_weight),
        )
        confidence_loss = functional.binary_cross_entropy_with_logits(
            confidence_logits, confidence_targets, weight=confidence_weights,
        )

        loss_dict = {
            'mask_ce_loss': mask_ce_loss,
            'confidence_loss': confidence_loss,
        }
        return loss_dict, match_tuple

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
            - Cross-entropy: -log(softmax(mask_logits)[gt_track_index])
            - BCE for confidence (target = 1.0 for valid, 0.0 for padding)

        Args:
            denoising_mask_logits: (B, G * max_gt, P) mask logits for denoising
                queries. Padded track positions have logits = -inf.
            denoising_confidence_logits: (B, G * max_gt) confidence logits.
            denoising_mask_targets: (B, G * max_gt, P) one-hot binary mask targets.
            denoising_confidence_targets: (B, G * max_gt) binary confidence targets.
            denoising_valid_mask: (B, G * max_gt) which queries are valid.

        Returns:
            Dict with 'denoising_mask_ce_loss', 'denoising_confidence_loss'.
        """
        device = denoising_mask_logits.device
        num_valid_total = denoising_valid_mask.sum().item()

        if num_valid_total > 0:
            # Flatten valid denoising queries: (num_valid, P)
            valid_batch_indices, valid_query_indices = denoising_valid_mask.nonzero(as_tuple=True)
            valid_logits = denoising_mask_logits[valid_batch_indices, valid_query_indices]  # (V, P)
            valid_targets = denoising_mask_targets[valid_batch_indices, valid_query_indices]  # (V, P)

            # Extract GT track index from one-hot target mask
            # Each target has exactly 1 positive track → argmax gives the index
            gt_track_indices = valid_targets.argmax(dim=1)  # (V,)

            # Cross-entropy: softmax over P tracks, NLL at GT track position
            # Padded tracks (logits = -inf) get softmax = 0 automatically.
            denoising_mask_ce_loss = functional.cross_entropy(
                valid_logits, gt_track_indices,
            )
        else:
            denoising_mask_ce_loss = torch.tensor(0.0, device=device)

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
            'denoising_mask_ce_loss': denoising_mask_ce_loss,
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
                - 'mask_ce_loss': cross-entropy mask loss averaged across layers
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
            accumulated_mask_ce_loss = torch.tensor(0.0, device=enriched_features.device)
            accumulated_confidence_loss = torch.tensor(0.0, device=enriched_features.device)
            accumulated_denoising_mask_ce = torch.tensor(0.0, device=enriched_features.device)
            accumulated_denoising_confidence = torch.tensor(0.0, device=enriched_features.device)

            # Run Hungarian matching ONCE on the last layer's output,
            # then reuse the assignment for all layers' auxiliary losses.
            # This follows standard practice from Mask2Former (Cheng et al.,
            # CVPR 2022) and DN-DETR (Li et al., CVPR 2022): match on the
            # final decoder layer, reuse for auxiliary layers.
            last_learnable_mask = all_layer_mask_logits[-1][:, :num_learnable, :]
            last_learnable_conf = all_layer_confidence_logits[-1][:, :num_learnable]
            _, cached_match = self._compute_losses_single_layer(
                last_learnable_mask, last_learnable_conf,
                ground_truth_indices, ground_truth_count, mask,
            )

            for layer_index in range(num_layers):
                layer_mask_logits = all_layer_mask_logits[layer_index]
                layer_confidence_logits = all_layer_confidence_logits[layer_index]

                # ---- Learnable query losses ----
                learnable_mask_logits = layer_mask_logits[:, :num_learnable, :]
                learnable_confidence_logits = layer_confidence_logits[:, :num_learnable]

                layer_losses, _ = self._compute_losses_single_layer(
                    learnable_mask_logits,
                    learnable_confidence_logits,
                    ground_truth_indices,
                    ground_truth_count,
                    mask,
                    precomputed_match=cached_match,
                )

                accumulated_mask_ce_loss = accumulated_mask_ce_loss + layer_losses['mask_ce_loss']
                accumulated_confidence_loss = accumulated_confidence_loss + layer_losses['confidence_loss']

                # ---- Denoising query losses (at every layer, not just the last) ----
                # Computing denoising losses at all layers ensures balanced
                # gradient contribution with the learnable losses, which are
                # also computed at every layer and averaged.
                denoising_layer_mask = layer_mask_logits[:, num_learnable:, :]
                denoising_layer_conf = layer_confidence_logits[:, num_learnable:]

                denoising_layer_losses = self._compute_denoising_losses(
                    denoising_layer_mask,
                    denoising_layer_conf,
                    denoising_mask_targets,
                    denoising_confidence_targets,
                    denoising_valid_mask,
                )

                accumulated_denoising_mask_ce = (
                    accumulated_denoising_mask_ce
                    + denoising_layer_losses['denoising_mask_ce_loss']
                )
                accumulated_denoising_confidence = (
                    accumulated_denoising_confidence
                    + denoising_layer_losses['denoising_confidence_loss']
                )

            # Average across layers (uniform layer weighting)
            averaged_mask_ce_loss = accumulated_mask_ce_loss / num_layers
            averaged_confidence_loss = accumulated_confidence_loss / num_layers
            averaged_denoising_mask_ce = accumulated_denoising_mask_ce / num_layers
            averaged_denoising_confidence = accumulated_denoising_confidence / num_layers

            # Weighted denoising loss — scaled by denoising_loss_weight to
            # control the balance between learnable and denoising gradients.
            denoising_total = self.denoising_loss_weight * (
                self.mask_ce_loss_weight * averaged_denoising_mask_ce
                + self.confidence_loss_weight * averaged_denoising_confidence
            )

            # Total loss: weighted sum of learnable + denoising losses
            # total = w_mask * CE_mask + w_conf * BCE_conf + w_dn * (w_mask * CE_dn_mask + w_conf * BCE_dn_conf)
            total_loss = (
                self.mask_ce_loss_weight * averaged_mask_ce_loss
                + self.confidence_loss_weight * averaged_confidence_loss
                + denoising_total
            )

            return {
                'total_loss': total_loss,
                'mask_ce_loss': averaged_mask_ce_loss,
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
