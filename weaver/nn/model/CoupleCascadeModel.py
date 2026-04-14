"""Stage 3 cascade model: frozen 2-stage cascade + trainable CoupleReranker.

Glues a frozen ``CascadeModel`` (TrackPreFilter + CascadeReranker) with a
trainable ``CoupleReranker`` to form the post-ParT direction-A pipeline
described in ``reports/triplet_reranking/triplet_research_plan_20260408.md``.

Forward pass:
    1. Run frozen cascade Stage 1 → top-K1 candidates
    2. Run frozen cascade Stage 2 (ParT) → per-track scores within K1
    3. Take top-K2 (default 50) by Stage 2 score
    4. Build all C(K2, 2) couple feature vectors per event (vectorized)
    5. Apply Filter A as a per-couple mask
    6. Forward through CoupleReranker → per-couple scores

Stage 1 + Stage 2 are frozen — only the CoupleReranker trains. The
training script (``part/train_couple_reranker.py``) handles batching,
loss computation through ``compute_loss``, and the per-couple metric.
"""
import torch
import torch.nn as nn


class CoupleCascadeModel(nn.Module):
    """Frozen cascade + trainable CoupleReranker.

    Args:
        cascade: Trained ``CascadeModel`` (Stage 1 + Stage 2). All
            parameters will be frozen.
        couple_reranker: Trainable ``CoupleReranker`` instance.
        top_k2: Number of top tracks (sorted by Stage 2 score) per event
            from which couples are enumerated. Default 50.
    """

    def __init__(
        self,
        cascade: nn.Module,
        couple_reranker: nn.Module,
        top_k2: int = 50,
        k_values_tracks: tuple[int, ...] = (30, 50, 75, 100, 200),
    ):
        super().__init__()
        self.cascade = cascade
        self.couple_reranker = couple_reranker
        self.top_k2 = top_k2
        # K values for the D@K_tracks metric — used inside
        # _run_cascade_to_top_k2 to count GT pions in each top-K_tracks
        # prefix of the Stage 2 score ranking. Stored as a buffer so
        # the values stick with the saved checkpoint.
        self.k_values_tracks = tuple(k_values_tracks)

        # Freeze the entire cascade (Stage 1 + Stage 2). The optimizer
        # will pick up only `couple_reranker.parameters()` because we
        # don't filter explicitly — the user must use a filter or pass
        # `filter(lambda p: p.requires_grad, model.parameters())`.
        for parameter in self.cascade.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _run_cascade_to_top_k2(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the frozen cascade and gather data at the top-K2 positions.

        Returns a dict with per-event ``(B, ..., K2)`` tensors gathered at
        the indices of the top-K2 highest-Stage-2-score tracks. Also
        includes ``n_gt_in_top_k1`` — the per-event count of GT pions in
        the Stage 1 top-K1 selection — needed for the RC@K metric in
        ``CoupleMetricsAccumulator``.
        """
        # Stage 1 → top-K1 (the existing CascadeModel pattern)
        filtered = self.cascade._run_stage1(
            points, features, lorentz_vectors, mask, track_labels,
        )

        # Stage 2 forward on the top-K1 set
        stage2_scores = self.cascade.stage2(
            filtered['points'],
            filtered['features'],
            filtered['lorentz_vectors'],
            filtered['mask'],
            filtered['stage1_scores'],
        )  # (B, K1)

        # n_gt_in_top_k1: count of GT pions surviving Stage 1's top-K1.
        # Used downstream by CoupleMetricsAccumulator to compute RC@K.
        # filtered['track_labels'] has shape (B, 1, K1); filtered['mask']
        # has the same shape.
        filtered_labels_flat = filtered['track_labels'].squeeze(1) > 0.5
        filtered_mask_flat = filtered['mask'].squeeze(1) > 0.5
        gt_in_k1_mask = filtered_labels_flat & filtered_mask_flat
        n_gt_in_top_k1 = gt_in_k1_mask.sum(dim=1)

        # n_gt_in_top_k_tracks: per-event GT pion counts at each cumulative
        # top-K of the Stage 2 score ranking, for the K values configured
        # via `k_values_tracks`. Used downstream to compute D@K_tracks.
        # We sort the K1 tracks by stage 2 score and count GT pions in
        # each top-K prefix.
        sorted_stage2_indices = torch.argsort(
            stage2_scores, dim=1, descending=True,
        )
        sorted_gt_in_k1 = gt_in_k1_mask.gather(1, sorted_stage2_indices)
        max_k = sorted_gt_in_k1.shape[1]
        n_gt_in_top_k_tracks_columns = []
        for k_tracks in self.k_values_tracks:
            effective_k = min(k_tracks, max_k)
            n_gt_in_top_k_tracks_columns.append(
                sorted_gt_in_k1[:, :effective_k].sum(dim=1),
            )
        # Shape: (B, len(k_values_tracks))
        n_gt_in_top_k_tracks = torch.stack(n_gt_in_top_k_tracks_columns, dim=1)

        # Take top-K2 by Stage 2 score (within the K1 candidates)
        top_k2_in_k1 = stage2_scores.topk(self.top_k2, dim=1).indices  # (B, K2)

        def gather_along_track_dim(tensor: torch.Tensor) -> torch.Tensor:
            """Gather along the K1 dim using top_k2_in_k1 indices.

            Args:
                tensor: ``(B, C, K1)`` per-track tensor.

            Returns:
                ``(B, C, K2)`` tensor restricted to the top-K2 positions.
            """
            num_channels = tensor.shape[1]
            expanded_indices = top_k2_in_k1.unsqueeze(1).expand(-1, num_channels, -1)
            return tensor.gather(2, expanded_indices)

        # Track validity mask: True for real tracks, False for padding.
        # Events with fewer valid tracks than K2 fill the remaining slots
        # with padding tracks whose Stage 2 scores are -inf (via
        # CascadeReranker.forward masked_fill). topk selects these because
        # -inf < all finite values, so they end up at the tail of the
        # top-K2 list. The mask lets the couple feature builder zero out
        # padding data and exclude couples involving padding from the loss.
        top_k2_stage2_scores = stage2_scores.gather(1, top_k2_in_k1)
        track_valid_mask = torch.isfinite(top_k2_stage2_scores)  # (B, K2)

        return {
            'features': gather_along_track_dim(filtered['features']),
            'points': gather_along_track_dim(filtered['points']),
            'lorentz_vectors': gather_along_track_dim(filtered['lorentz_vectors']),
            'stage1_scores': filtered['stage1_scores'].gather(1, top_k2_in_k1),
            'stage2_scores': top_k2_stage2_scores,
            'track_labels': filtered['track_labels'].squeeze(1).gather(1, top_k2_in_k1),
            'track_valid_mask': track_valid_mask,
            'n_gt_in_top_k1': n_gt_in_top_k1,
            'n_gt_in_top_k_tracks': n_gt_in_top_k_tracks,
        }

    def _build_couple_inputs(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the cascade and build the per-couple feature batch.

        Returns a dict with:
            ``couple_features``: ``(B, 51, n_couples)`` per-couple feature tensor
            ``filter_a_mask``: ``(B, n_couples)`` boolean — True for couples
                passing the loose ``m(ij) <= m_tau`` cut
            ``couple_labels``: ``(B, n_couples)`` boolean — True iff both
                tracks of the couple are GT pions
            ``n_gt_in_top_k1``: ``(B,)`` per-event count of GT pions in
                Stage 1 top-K1 — used by ``CoupleMetricsAccumulator`` for
                the RC@K metric.
        """
        # Lazy import to keep the weaver-side module dependency-free
        from utils.couple_features import build_couple_features_batched

        top_k2_data = self._run_cascade_to_top_k2(
            points, features, lorentz_vectors, mask, track_labels,
        )
        couple_inputs = build_couple_features_batched(
            top_k2_features=top_k2_data['features'],
            top_k2_points=top_k2_data['points'],
            top_k2_lorentz=top_k2_data['lorentz_vectors'],
            top_k2_stage1_scores=top_k2_data['stage1_scores'],
            top_k2_stage2_scores=top_k2_data['stage2_scores'],
            top_k2_track_labels=top_k2_data['track_labels'],
            track_valid_mask=top_k2_data['track_valid_mask'],
        )
        couple_inputs['n_gt_in_top_k1'] = top_k2_data['n_gt_in_top_k1']
        couple_inputs['n_gt_in_top_k_tracks'] = top_k2_data['n_gt_in_top_k_tracks']
        return couple_inputs

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward pass: returns ``(scores, filter_a_mask)``.

        At inference time we don't have track labels (the cascade output
        is passed through). The labels go via ``compute_loss`` instead.

        Returns:
            scores: ``(B, n_couples)`` per-couple scores
            filter_a_mask: ``(B, n_couples)`` boolean — couples passing
                the loose mass cut
        """
        dummy_track_labels = torch.zeros_like(mask)
        couple_inputs = self._build_couple_inputs(
            points, features, lorentz_vectors, mask, dummy_track_labels,
        )
        scores = self.couple_reranker(couple_inputs['couple_features'])
        return scores, couple_inputs['filter_a_mask']

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward + ranking loss + metric data.

        Returns the standard CoupleReranker loss dict (``total_loss``,
        ``ranking_loss``, ``_scores``) plus extra fields the training
        loop pops out for metrics:

            ``_couple_labels``: ``(B, n_couples)`` GT-couple labels
            ``_couple_mask``: ``(B, n_couples)`` Filter A boolean mask
            ``_n_gt_in_top_k1``: ``(B,)`` per-event GT-pion count in
                Stage 1 top-K1, used by ``CoupleMetricsAccumulator`` for
                the RC@K metric
        """
        couple_inputs = self._build_couple_inputs(
            points, features, lorentz_vectors, mask, track_labels,
        )
        couple_features = couple_inputs['couple_features']
        couple_labels = couple_inputs['couple_labels']
        filter_a_mask = couple_inputs['filter_a_mask']
        n_gt_in_top_k1 = couple_inputs['n_gt_in_top_k1']
        n_gt_in_top_k_tracks = couple_inputs['n_gt_in_top_k_tracks']

        loss_dict = self.couple_reranker.compute_loss(
            couple_features=couple_features,
            couple_labels=couple_labels.to(couple_features.dtype),
            couple_mask=filter_a_mask.to(couple_features.dtype),
        )
        loss_dict['_couple_labels'] = couple_labels
        loss_dict['_couple_mask'] = filter_a_mask
        loss_dict['_n_gt_in_top_k1'] = n_gt_in_top_k1
        loss_dict['_n_gt_in_top_k_tracks'] = n_gt_in_top_k_tracks
        return loss_dict
