"""ParticleNeXt-based pre-filter for two-stage track finding pipeline.

Uses ParticleNeXt (for_segmentation=True) as the backbone for per-track
scoring, wrapped with ranking loss and training schedule from TrackPreFilter.

ParticleNeXt provides:
  - Multi-scale EdgeConv with attention-weighted aggregation
  - Pairwise Lorentz vector edge features (ΔR, invariant mass, kT, z)
  - Squeeze-and-Excitation channel attention
  - Residual connections with learnable γ scaling
  - Built-in data augmentation (pT dropout, LV smearing)

The loss/scheduling interface matches TrackPreFilter exactly, so
train_prefilter.py requires zero changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.ParticleNeXt import ParticleNeXt


class ParticleNeXtPreFilter(nn.Module):
    """ParticleNeXt backbone with ranking loss for track pre-selection.

    Args:
        feature_input_dim: Number of raw features per track.
        node_dim: Initial node embedding dimension for ParticleNeXt.
        edge_dim: Edge feature embedding dimension.
        layer_params: MultiScaleEdgeConv layer specifications.
            Each entry: (k, out_dim, reduction_dilation, message_dim).
        fc_params: Final FC layers as [(out_dim, dropout_rate), ...].
        edge_aggregation: Edge aggregation method ('attn8', 'sum', etc.).
        use_rel_lv_fts: Include pairwise Lorentz vector features as edges.
        ranking_num_samples: Negatives sampled per positive in ranking loss.
        ranking_temperature_start: Initial ranking temperature (high = smooth).
        ranking_temperature_end: Final ranking temperature (low = sharp).
        denoising_sigma_start: Initial noise σ for contrastive denoising.
        denoising_sigma_end: Final noise σ for contrastive denoising.
        drw_warmup_fraction: Fraction of training before DRW activates.
        drw_positive_weight: Weight multiplier for positives when DRW active.
        focal_gamma: Focal weighting exponent (0 = disabled).
        contrastive_denoising_negative_sigma: σ for DINO negative copies (0 = disabled).
        input_dropout: Dropout rate on input mask (data augmentation).
        pt_dropout: pT threshold noise std for low-pT dropout (augmentation).
        lorentz_vector_scale: Scale noise std for LV augmentation.
        lorentz_vector_smear: Per-particle LV smear std (augmentation).
    """

    def __init__(
        self,
        feature_input_dim: int = 13,
        node_dim: int = 32,
        edge_dim: int = 8,
        layer_params: list | None = None,
        fc_params: list | None = None,
        edge_aggregation: str = 'attn8',
        use_rel_lv_fts: bool = True,
        # Ranking loss
        ranking_num_samples: int = 50,
        # Temperature scheduling (Kukleva et al., ICLR 2023)
        ranking_temperature_start: float = 1.0,
        ranking_temperature_end: float = 1.0,
        denoising_sigma_start: float = 0.3,
        denoising_sigma_end: float = 0.3,
        # Deferred Re-Weighting (Cao et al., NeurIPS 2019)
        drw_warmup_fraction: float = 1.0,
        drw_positive_weight: float = 1.0,
        # Equalized focal weighting (Li et al., CVPR 2022)
        focal_gamma: float = 0.0,
        # DINO contrastive denoising (Zhang et al., ICLR 2023)
        contrastive_denoising_negative_sigma: float = 0.0,
        # ParticleNeXt data augmentation
        input_dropout: float | None = None,
        pt_dropout: float | None = None,
        lorentz_vector_scale: float | None = None,
        lorentz_vector_smear: float | None = None,
    ):
        super().__init__()

        if layer_params is None:
            layer_params = [
                (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
                (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
                (32, 256, [(8, 1), (4, 1), (2, 1), (1, 1)], 64),
            ]
        if fc_params is None:
            fc_params = [(256, 0.1)]

        # ParticleNeXt backbone — per-track scoring via segmentation mode.
        # trim=False because train_prefilter.py already trims padded tracks
        # via trim_to_max_valid_tracks(), and the denoising loss calls
        # forward() multiple times (each trim would produce different lengths).
        self.backbone = ParticleNeXt(
            feature_input_dim=feature_input_dim,
            num_classes=1,
            for_segmentation=True,
            node_dim=node_dim,
            edge_dim=edge_dim,
            layer_params=layer_params,
            fc_params=fc_params,
            edge_aggregation=edge_aggregation,
            use_rel_lv_fts=use_rel_lv_fts,
            input_dropout=input_dropout,
            pt_dropout=pt_dropout,
            lorentz_vector_scale=lorentz_vector_scale,
            lorentz_vector_smear=lorentz_vector_smear,
            trim=False,
        )

        # ---- Training schedule parameters ----
        self.ranking_num_samples = ranking_num_samples

        # Temperature scheduling: σ(t) and T(t) linearly interpolate
        # between start and end values over training progress t ∈ [0,1].
        self.denoising_sigma_start = denoising_sigma_start
        self.denoising_sigma_end = denoising_sigma_end
        self.ranking_temperature_start = ranking_temperature_start
        self.ranking_temperature_end = ranking_temperature_end
        self._temperature_progress: float = 0.0

        # DRW: uniform weights for warmup fraction, then upweight positives.
        self.drw_warmup_fraction = drw_warmup_fraction
        self.drw_positive_weight = drw_positive_weight
        self._drw_active: bool = False

        # Focal weighting: (1-p)^γ modulation on pairwise loss.
        self.focal_gamma = focal_gamma

        # DINO contrastive denoising: 0 = disabled, >0 = negative copy σ.
        self.contrastive_denoising_negative_sigma = contrastive_denoising_negative_sigma

    # ---- Forward pass ----

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-track scores.

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, F, P) raw per-track features.
            lorentz_vectors: (B, 4, P) raw 4-vectors (px, py, pz, E).
            mask: (B, 1, P) boolean mask.

        Returns:
            scores: (B, P) per-track scores. Padded tracks get -inf.
        """
        # ParticleNeXt in segmentation mode returns (B, 1, P)
        raw_output = self.backbone(points, features, lorentz_vectors, mask)
        scores = raw_output.squeeze(1)  # (B, P)

        # Mask padded tracks (ParticleNeXt already masks internally,
        # but enforce -inf for consistency with downstream top-K selection)
        valid_mask = mask.squeeze(1).bool()
        scores = scores.masked_fill(~valid_mask, float('-inf'))

        return scores

    # ---- Scheduling ----

    def set_temperature_progress(self, progress: float) -> None:
        """Set curriculum progress for temperature and sigma scheduling.

        Linearly interpolates between start and end values:
            σ(t) = σ_start + t × (σ_end - σ_start)
            T(t) = T_start + t × (T_end - T_start)

        Args:
            progress: Float in [0, 1]. 0 = start of training, 1 = end.
        """
        self._temperature_progress = max(0.0, min(1.0, progress))

    @property
    def current_denoising_sigma(self) -> float:
        """Current noise sigma, interpolated by training progress."""
        return (
            self.denoising_sigma_start
            + self._temperature_progress
            * (self.denoising_sigma_end - self.denoising_sigma_start)
        )

    @property
    def current_ranking_temperature(self) -> float:
        """Current ranking loss temperature, interpolated by training progress."""
        return (
            self.ranking_temperature_start
            + self._temperature_progress
            * (self.ranking_temperature_end - self.ranking_temperature_start)
        )

    def set_drw_active(self, active: bool) -> None:
        """Activate/deactivate Deferred Re-Weighting of positive samples.

        Args:
            active: If True, multiply ranking loss by drw_positive_weight.
        """
        self._drw_active = active

    # ---- Loss functions ----

    def _ranking_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Temperature-scaled pairwise ranking loss optimized for R@K.

        For each GT pion, sample S negatives and penalize negatives
        scoring above positives:
            L = T × softplus((s_neg - s_pos) / T)

        High T smooths gradients; low T sharpens focus on hard violations.
        When DRW is active, loss is scaled by drw_positive_weight.
        """
        batch_size = scores.shape[0]
        temperature = self.current_ranking_temperature
        event_losses = []

        for event_index in range(batch_size):
            event_labels = labels[event_index]
            event_scores = scores[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels == 0.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=scores.device,
            )
            sampled_negatives = negative_indices[sample_idx]

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            # L = T × log(1 + exp((s_neg - s_pos) / T))
            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)

            # Equalized focal weighting: w = (1 - p)^γ
            # where p = σ(s_pos - s_neg) = probability of correct ordering.
            # .detach() prevents focal weights from generating own gradients.
            if self.focal_gamma > 0:
                ordering_probability = torch.sigmoid(-scaled_margin)
                focal_weight = (
                    (1.0 - ordering_probability).pow(self.focal_gamma)
                ).detach()
                pairwise_loss = focal_weight * pairwise_loss

            # DRW: upweight positive-negative pairs after warmup
            if self._drw_active:
                pairwise_loss = pairwise_loss * self.drw_positive_weight

            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        return torch.stack(event_losses).mean()

    def _contrastive_denoising_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> torch.Tensor:
        """DINO-style contrastive denoising loss.

        Creates noised GT track copies (Zhang et al., ICLR 2023):
        1. Positive copies (small noise, scheduled σ): must score ABOVE background.
        2. Negative copies (large noise, σ_neg): must score BELOW positive copies.
        """
        valid_mask = mask.squeeze(1).bool()
        labels_flat = (
            track_labels.squeeze(1)[:, :valid_mask.shape[1]] * valid_mask.float()
        )

        gt_mask = (labels_flat == 1.0) & valid_mask

        if not gt_mask.any():
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)

        gt_mask_expanded = gt_mask.unsqueeze(1)

        # --- Positive copies: small noise (scheduled sigma) ---
        positive_noise = (
            torch.randn_like(features) * self.current_denoising_sigma
        )
        positive_noised_features = torch.where(
            gt_mask_expanded, features + positive_noise, features,
        )
        positive_noised_scores = self.forward(
            points, positive_noised_features, lorentz_vectors, mask,
        )

        # --- Negative copies: large noise (if enabled) ---
        use_negative_copies = self.contrastive_denoising_negative_sigma > 0
        if use_negative_copies:
            negative_noise = (
                torch.randn_like(features)
                * self.contrastive_denoising_negative_sigma
            )
            negative_noised_features = torch.where(
                gt_mask_expanded, features + negative_noise, features,
            )
            negative_noised_scores = self.forward(
                points, negative_noised_features, lorentz_vectors, mask,
            )

        # --- Per-event loss ---
        batch_size = features.shape[0]
        temperature = self.current_ranking_temperature
        event_losses = []

        for event_index in range(batch_size):
            gt_positions = gt_mask[event_index].nonzero(as_tuple=True)[0]
            if len(gt_positions) == 0:
                continue

            pos_scores = positive_noised_scores[event_index, gt_positions]

            negative_indices = (
                (labels_flat[event_index] == 0.0) & valid_mask[event_index]
            ).nonzero(as_tuple=True)[0]
            if len(negative_indices) == 0:
                continue

            num_samples = min(20, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=features.device,
            )
            background_scores = original_scores[
                event_index, negative_indices[sample_idx]
            ]

            # Loss 1: positive copies should beat background
            positive_pairwise = temperature * functional.softplus(
                (background_scores.unsqueeze(0) - pos_scores.unsqueeze(1))
                / temperature,
            )
            event_losses.append(positive_pairwise.mean())

            # Loss 2 (DINO): negative copies should score BELOW positive copies
            if use_negative_copies:
                neg_scores = negative_noised_scores[
                    event_index, gt_positions
                ]
                negative_pairwise = temperature * functional.softplus(
                    (neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0))
                    / temperature,
                )
                event_losses.append(negative_pairwise.mean())

        if not event_losses:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        return torch.stack(event_losses).mean()

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        use_contrastive_denoising: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss.

        Returns dict with 'total_loss', 'ranking_loss', optional
        'denoising_loss', and '_scores' for metric computation.
        """
        scores = self.forward(points, features, lorentz_vectors, mask)
        valid_mask = mask.squeeze(1).bool()
        labels_flat = (
            track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()
        )

        ranking_loss = self._ranking_loss(scores, labels_flat, valid_mask)

        total_loss = ranking_loss
        loss_dict = {
            'ranking_loss': ranking_loss,
        }

        if use_contrastive_denoising and self.training:
            denoising_loss = self._contrastive_denoising_loss(
                points, features, lorentz_vectors, mask, track_labels, scores,
            )
            total_loss = total_loss + 0.5 * denoising_loss
            loss_dict['denoising_loss'] = denoising_loss

        loss_dict['total_loss'] = total_loss
        loss_dict['_scores'] = scores
        return loss_dict

    # ---- Inference utilities ----

    def select_top_k(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        top_k: int = 200,
    ) -> torch.Tensor:
        """Select top-K track indices per event.

        Returns:
            selected_indices: (B, K) indices of top-K tracks.
        """
        valid_mask = mask.squeeze(1).bool()
        masked_scores = scores.clone()
        masked_scores[~valid_mask] = float('-inf')

        num_tracks = scores.shape[1]
        actual_k = min(top_k, num_tracks)

        _, top_indices = masked_scores.topk(actual_k, dim=1)
        return top_indices

    def filter_tracks(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        top_k: int = 200,
    ) -> dict[str, torch.Tensor]:
        """Score tracks, select top-K, repack all tensors.

        Returns dict with filtered tensors, each with P dimension = top_k.
        """
        scores = self.forward(points, features, lorentz_vectors, mask)
        selected_indices = self.select_top_k(scores, mask, top_k)

        batch_size, top_k_actual = selected_indices.shape

        def gather_tracks(tensor, indices):
            """Gather along the P (last) dimension."""
            num_channels = tensor.shape[1]
            expanded_indices = indices.unsqueeze(1).expand(
                -1, num_channels, -1,
            )
            return tensor.gather(2, expanded_indices)

        return {
            'points': gather_tracks(points, selected_indices),
            'features': gather_tracks(features, selected_indices),
            'lorentz_vectors': gather_tracks(lorentz_vectors, selected_indices),
            'mask': gather_tracks(mask, selected_indices),
            'track_labels': gather_tracks(track_labels, selected_indices),
        }
