"""Two-stage cascade model: Stage 1 pre-filter → top-K → Stage 2 reranker.

Lazy cascade: Stage 1 runs a forward pass each batch, selects top-K1 tracks,
and passes the filtered tensors to Stage 2. Stage 1 is frozen (no gradients).

Stage 2 interface (any nn.Module implementing these methods):
    forward(points, features, lorentz_vectors, mask, stage1_scores) -> scores
    compute_loss(points, features, lorentz_vectors, mask, track_labels,
                 stage1_scores) -> dict with 'total_loss', '_scores'
"""
import torch
import torch.nn as nn


class CascadeModel(nn.Module):
    """Two-stage cascade: frozen pre-filter → top-K1 selection → trainable reranker.

    Args:
        stage1: Trained TrackPreFilter model (will be frozen).
        stage2: Trainable Stage 2 model implementing the Stage 2 interface.
        top_k1: Number of tracks to pass from Stage 1 to Stage 2.
    """

    def __init__(
        self,
        stage1: nn.Module,
        stage2: nn.Module,
        top_k1: int = 600,
    ):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.top_k1 = top_k1

        # Freeze Stage 1: no gradients, always in eval mode
        for parameter in self.stage1.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _run_stage1(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run frozen Stage 1 and select top-K1 tracks.

        Returns dict with filtered tensors (B, C, K1), stage1_scores (B, K1),
        and selected_indices (B, K1) mapping back to full-event positions.
        """
        # Use train() mode for Stage 1: its BatchNorm running statistics are
        # stale because the pre-filter training loop ran validation in train()
        # mode (train_prefilter.py:239), corrupting running_mean/running_var.
        # With eval() mode, R@600 drops from 0.90 to 0.70, capping the cascade.
        # train() mode uses batch statistics which match training conditions.
        # @torch.no_grad() already prevents gradient computation and param updates.
        self.stage1.train()

        # Score all tracks
        scores = self.stage1(points, features, lorentz_vectors, mask)
        selected_indices = self.stage1.select_top_k(scores, mask, self.top_k1)

        # Gather filtered tensors along the track (last) dimension
        def gather_tracks(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            """Gather along P (last) dimension: (B, C, P) → (B, C, K1)."""
            num_channels = tensor.shape[1]
            expanded_indices = indices.unsqueeze(1).expand(-1, num_channels, -1)
            return tensor.gather(2, expanded_indices)

        # Gather stage1 scores for the selected tracks: (B, P) → (B, K1)
        stage1_scores = scores.gather(1, selected_indices)

        result = {
            'points': gather_tracks(points, selected_indices),
            'features': gather_tracks(features, selected_indices),
            'lorentz_vectors': gather_tracks(lorentz_vectors, selected_indices),
            'mask': gather_tracks(mask, selected_indices),
            'stage1_scores': stage1_scores,
            'selected_indices': selected_indices,
        }
        if track_labels is not None:
            result['track_labels'] = gather_tracks(track_labels, selected_indices)

        return result

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run full cascade: Stage 1 filter → Stage 2 score.

        Args:
            points: (B, 2, P) coordinates in (η, φ).
            features: (B, input_dim, P) raw per-track features.
            lorentz_vectors: (B, 4, P) raw 4-vectors.
            mask: (B, 1, P) boolean mask.

        Returns:
            scores: (B, K1) per-track scores from Stage 2 on the filtered set.
        """
        filtered = self._run_stage1(points, features, lorentz_vectors, mask)

        scores = self.stage2(
            filtered['points'],
            filtered['features'],
            filtered['lorentz_vectors'],
            filtered['mask'],
            filtered['stage1_scores'],
        )
        return scores

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        use_contrastive_denoising: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run cascade and compute Stage 2 loss on filtered tracks.

        Also computes Stage 1 recall@K1 for monitoring (how many GT tracks
        survived the filter).

        Args:
            points, features, lorentz_vectors, mask, track_labels: Full-event
                inputs before Stage 1 filtering.
            use_contrastive_denoising: Forwarded to ``stage2.compute_loss``.
                Training scripts should pass ``False`` from their validation
                loop so the denoising auxiliary term doesn't inflate val loss
                when the model is flipped to ``train()`` for BatchNorm stats.

        Returns:
            dict with 'total_loss', Stage 2 components, 'stage1_recall_at_k1',
            and '_scores' (B, K1) from Stage 2.
        """
        filtered = self._run_stage1(
            points, features, lorentz_vectors, mask, track_labels,
        )

        # Stage 2 loss on the filtered tracks. Forward the denoising kwarg
        # so the caller (training script) can disable it in the validate
        # loop without touching any instance state.
        loss_dict = self.stage2.compute_loss(
            filtered['points'],
            filtered['features'],
            filtered['lorentz_vectors'],
            filtered['mask'],
            filtered['track_labels'],
            filtered['stage1_scores'],
            use_contrastive_denoising=use_contrastive_denoising,
        )

        # Stage 1 recall@K1: fraction of GT tracks that survived the filter
        # This is a monitoring metric, not a loss term.
        filtered_labels = filtered['track_labels'].squeeze(1)
        filtered_mask = filtered['mask'].squeeze(1).bool()
        original_labels = track_labels.squeeze(1)[:, :mask.shape[2]]
        original_mask = mask.squeeze(1).bool()

        # GT count in the filtered set vs GT count in the full event
        gt_in_filtered = (
            (filtered_labels == 1.0) & filtered_mask
        ).sum(dim=1).float()  # (B,)
        gt_in_original = (
            (original_labels == 1.0) & original_mask
        ).sum(dim=1).float()  # (B,)

        # R@K1 = gt_in_filtered / gt_in_original, averaged over events with GT
        has_gt = gt_in_original > 0
        if has_gt.any():
            recall_at_k1 = (
                gt_in_filtered[has_gt] / gt_in_original[has_gt]
            ).mean()
        else:
            recall_at_k1 = torch.tensor(0.0, device=points.device)

        loss_dict['stage1_recall_at_k1'] = recall_at_k1

        return loss_dict
