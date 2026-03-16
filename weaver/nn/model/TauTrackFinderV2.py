"""Tau-origin pion track finder V2: neighborhood-aware scoring with
top-K self-attention refinement.

Replaces the DETR decoder (4.6M params, 0% recall contribution) with:
    M4 -- No DETR decoder: all recall comes from per-track scoring.
    M1 -- kNN message passing for neighborhood-aware per-track scoring.
    M5 -- Skip-connected displacement features (dxy_significance, pT).
    M2 -- Self-attention refinement on top-K candidates.

Architecture:
    1. Pretrained EnrichCompactBackbone (frozen, enrichment only) -> (B, 256, P)
    2. kNN in (eta, phi) -> max-pool neighbor features -> MLP -> messages (B, 64, P)
    3. Skip features: dxy_significance + pT from raw inputs -> (B, 2, P)
    4. Combined = cat(enriched, messages, skip) -> (B, 322, P)
    5. Per-track scoring MLP -> per_track_logits (B, P)
    6. Top-K selection -> self-attention -> refined_logits (B, K)

Loss:
    - Per-track focal BCE on ALL ~1130 tracks (primary classification signal)
    - Refinement focal BCE on top-K combined scores (pairwise reasoning)
      Combined = per_track_logits[top-K] + refined_logits, so gradients
      from the refinement loss also reach the per-track head.

References:
    Focal loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    DGCNN/EdgeConv: Wang et al., "Dynamic Graph CNN", ACM TOG 2019
    Transformer: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.EnrichCompactBackbone import EnrichCompactBackbone
from weaver.nn.model.HierarchicalGraphBackbone import cross_set_knn, cross_set_gather


class TauTrackFinderV2(nn.Module):
    """Neighborhood-aware tau track finder with top-K self-attention refinement.

    M4: No DETR decoder -- all recall from per-track scoring (~33K params
        vs 4.6M for the dead-weight DETR decoder).
    M1: kNN + max-pool + MLP neighbor messages give each track awareness
        of its local neighborhood in (eta, phi) space.
    M5: Raw physics features (dxy_significance, pT) skip-connected to the
        scoring head, bypassing the 256-dim backbone bottleneck.
    M2: Self-attention on top-K candidates enables pairwise reasoning
        (shared displaced vertex, compatible momenta, mass constraint).

    Args:
        backbone_kwargs: Config for EnrichCompactBackbone.
        num_scoring_neighbors: K for same-set kNN in (eta, phi) (default: 16).
        message_dim: Output dim of neighbor message MLP (default: 64).
        num_refinement_candidates: Number of top candidates for self-attention
            refinement (default: 256).
        refinement_dim: Internal dimension of self-attention layers (default: 128).
        num_refinement_layers: Number of self-attention layers (default: 2).
        refinement_num_heads: Attention heads per refinement layer (default: 4).
        refinement_dropout: Dropout in self-attention layers (default: 0.1).
        per_track_loss_weight: Weight for per-track focal BCE (default: 1.0).
        refinement_loss_weight: Weight for refinement focal BCE (default: 1.0).
        focal_alpha: Alpha for focal loss class weighting (default: 0.75).
            Higher alpha upweights the rare positive class (~3/1130 = 0.3%).
        focal_gamma: Gamma for focal loss modulation (default: 2.0).
            Downweights easy examples where the model is already confident.
        dxy_significance_feature_index: Index of dxy_significance in pf_features
            (default: 6, matching YAML config order:
             0=px, 1=py, 2=pz, 3=eta, 4=phi, 5=charge, 6=dxy_significance).
    """

    def __init__(
        self,
        backbone_kwargs: dict | None = None,
        num_scoring_neighbors: int = 16,
        message_dim: int = 64,
        num_refinement_candidates: int = 256,
        refinement_dim: int = 128,
        num_refinement_layers: int = 2,
        refinement_num_heads: int = 4,
        refinement_dropout: float = 0.1,
        per_track_loss_weight: float = 1.0,
        refinement_loss_weight: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dxy_significance_feature_index: int = 6,
    ):
        super().__init__()

        if backbone_kwargs is None:
            backbone_kwargs = {}

        self.num_scoring_neighbors = num_scoring_neighbors
        self.num_refinement_candidates = num_refinement_candidates
        self.per_track_loss_weight = per_track_loss_weight
        self.refinement_loss_weight = refinement_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dxy_significance_feature_index = dxy_significance_feature_index

        # ---- Backbone (frozen) ----
        self.backbone = EnrichCompactBackbone(**backbone_kwargs)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        backbone_dim = self.backbone.enrichment_output_dim  # 256

        # ---- M1: Neighbor-aware message passing ----
        # Pool-then-transform approach for memory efficiency:
        #   1. kNN in (eta, phi) -> gather neighbor enriched features
        #   2. Max-pool over K neighbors -> (B, backbone_dim, P)
        #   3. MLP([center, max_neighbor]) -> messages (B, message_dim, P)
        #
        # Max-pooling does per-channel selection: each of backbone_dim
        # channels independently picks its strongest-activating neighbor,
        # preserving local structure (DGCNN/ParticleNet design).
        self.neighbor_message_mlp = nn.Sequential(
            # Input: cat([center, max_pooled_neighbor]) = 2 * backbone_dim
            nn.Conv1d(2 * backbone_dim, message_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(message_dim),
            nn.GELU(),
        )

        # ---- M5: Skip-connected physics features ----
        # dxy_significance (index 6 in standardized pf_features) +
        # pT (computed from raw lorentz_vectors).
        skip_feature_dim = 2
        self.skip_feature_norm = nn.BatchNorm1d(skip_feature_dim)

        # ---- Per-track scoring head ----
        # Scores each track using enriched features, neighbor context,
        # and raw physics discriminants.
        # combined_dim = backbone_dim + message_dim + skip_feature_dim
        combined_dim = backbone_dim + message_dim + skip_feature_dim
        self.combined_dim = combined_dim

        self.per_track_head = nn.Sequential(
            nn.Conv1d(combined_dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

        # ---- M2: Top-K self-attention refinement ----
        # Projects combined features into refinement space, runs
        # self-attention for pairwise reasoning among candidates,
        # then produces a per-candidate score adjustment.
        #
        # Pre-LN Transformer (norm_first=True) for training stability:
        #   x = x + SelfAttn(LayerNorm(x))
        #   x = x + FFN(LayerNorm(x))
        self.refinement_projection = nn.Linear(combined_dim, refinement_dim)
        self.refinement_norm = nn.LayerNorm(refinement_dim)
        self.refinement_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=refinement_dim,
                nhead=refinement_num_heads,
                dim_feedforward=refinement_dim * 4,
                dropout=refinement_dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_refinement_layers)
        ])
        self.refinement_scorer = nn.Sequential(
            nn.Linear(refinement_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _compute_neighbor_messages(
        self,
        enriched_features: torch.Tensor,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """M1: Compute neighborhood-aware messages via kNN + max-pool + MLP.

        For each track, finds K nearest neighbors in (eta, phi) space,
        gathers their enriched features, max-pools across neighbors, then
        applies an MLP to [center, max_neighbor] to produce messages.

        Physics motivation: In 3-prong tau -> pi+ pi- pi+ decays, the
        pions are collimated (close in eta, phi). A track surrounded by
        other tau-pion-like tracks should score higher than an isolated
        tau-like track. The OC baseline captures this through clustering
        coordinates + attractive/repulsive potential. This method provides
        an analogous neighborhood signal.

        Args:
            enriched_features: (B, backbone_dim, P) from frozen backbone.
            points: (B, 2, P) coordinates in (eta, phi).
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            neighbor_messages: (B, message_dim, P) per-track messages
                incorporating local neighborhood context.
        """
        # ---- kNN index computation (no gradients needed) ----
        # cross_set_knn handles phi wrapping: Delta_phi = (phi_a - phi_b + pi) mod 2pi - pi
        # Self-loops are allowed (query_reference_indices=None) -- harmless for max-pool
        # and provides a natural self-loop in the aggregation.
        with torch.no_grad():
            neighbor_indices = cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=self.num_scoring_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )  # (B, P, K)

        # ---- Gather and pool neighbor features ----
        # enriched_features: (B, backbone_dim, P) -> gather -> (B, backbone_dim, P, K)
        neighbor_features = cross_set_gather(
            enriched_features, neighbor_indices,
        )  # (B, backbone_dim, P, K)

        # Mask invalid neighbors (padded tracks selected by kNN)
        neighbor_validity = cross_set_gather(
            mask.float(), neighbor_indices,
        )  # (B, 1, P, K)
        neighbor_features = neighbor_features.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )

        # Max-pool over K neighbors: per-channel selection.
        # max_j in N(i) { enriched_j[c] } for each channel c independently.
        max_pooled_neighbors = neighbor_features.max(dim=-1)[0]  # (B, backbone_dim, P)

        # Handle all-masked case (events with very few tracks)
        max_pooled_neighbors = max_pooled_neighbors.masked_fill(
            max_pooled_neighbors == float('-inf'), 0.0,
        )

        # ---- MLP on [center, max_pooled_neighbor] ----
        # Captures "how does the neighborhood look alongside this track?"
        mlp_input = torch.cat(
            [enriched_features, max_pooled_neighbors], dim=1,
        )  # (B, 2 * backbone_dim, P)

        neighbor_messages = self.neighbor_message_mlp(mlp_input)  # (B, message_dim, P)

        # Zero out padded positions to prevent information leakage
        neighbor_messages = neighbor_messages * mask.float()

        return neighbor_messages

    def _compute_skip_features(
        self,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """M5: Extract displacement and momentum features as skip connections.

        Provides raw physics discriminants directly to the scoring head,
        bypassing the 256-dim backbone bottleneck where these features may
        be diluted among all dimensions.

        Physics motivation:
            - dxy_significance = |d_xy| / sigma(d_xy): tau pions originate
              from a displaced secondary vertex, so high dxy significance
              is the strongest single-variable discriminant for tau-origin
              tracks. Especially important at low pT where calorimetric
              tau ID fails.
            - pT = sqrt(px^2 + py^2): transverse momentum of the track.
              Low-pT tau decay products have characteristic momentum
              spectrum. The neutrino carrying significant energy means
              visible decay products have unusual momentum patterns.

        Args:
            features: (B, input_dim, P) standardized per-track features.
                Index 6 = dxy_significance (after standardization).
            lorentz_vectors: (B, 4, P) raw 4-vectors [px, py, pz, E].
            mask: (B, 1, P) boolean mask.

        Returns:
            skip_features: (B, 2, P) normalized [dxy_significance, pT].
        """
        # Extract dxy_significance from standardized features
        dxy_index = self.dxy_significance_feature_index
        dxy_significance = features[:, dxy_index:dxy_index + 1, :]  # (B, 1, P)

        # Compute transverse momentum: pT = sqrt(px^2 + py^2 + epsilon)
        # Force float32 to avoid precision loss in sqrt under AMP.
        # lorentz_vectors[:, 0] = px, lorentz_vectors[:, 1] = py (raw, unscaled)
        transverse_momentum = torch.sqrt(
            lorentz_vectors[:, 0:1, :].float().square()
            + lorentz_vectors[:, 1:2, :].float().square()
            + 1e-8,  # Epsilon prevents NaN in sqrt backward when pT -> 0
        ).to(features.dtype)  # (B, 1, P), cast back to input dtype

        # Concatenate and normalize via BatchNorm
        # BN normalizes pT to zero-mean unit-variance (dxy_significance
        # is already standardized, BN on it is harmless).
        skip_features = torch.cat(
            [dxy_significance, transverse_momentum], dim=1,
        )  # (B, 2, P)
        skip_features = self.skip_feature_norm(skip_features)
        skip_features = skip_features * mask.float()

        return skip_features

    def _focal_bce_loss(
        self,
        predicted_logits: torch.Tensor,
        target_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal binary cross-entropy loss over valid tracks.

        Focal loss (Lin et al., RetinaNet, ICCV 2017):
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        With ~3 positive out of ~1130 tracks, focal loss is critical:
            - alpha = 0.75 upweights the rare positive class
            - gamma = 2.0 downweights easy negatives (most background
              tracks are trivially classifiable)

        Args:
            predicted_logits: (B, N) per-track logits (pre-sigmoid).
            target_labels: (B, N) binary labels (1.0 = tau track).
            valid_mask: (B, N) boolean or float, True/1.0 for valid tracks.

        Returns:
            Scalar focal BCE loss averaged over valid tracks.
        """
        # Standard per-element BCE (numerically stable via log-sum-exp)
        # BCE(x, y) = -[y * log(sigma(x)) + (1-y) * log(1 - sigma(x))]
        bce_per_track = functional.binary_cross_entropy_with_logits(
            predicted_logits, target_labels, reduction='none',
        )  # (B, N)

        # p_t = P(correct class): sigma(x) for positives, 1-sigma(x) for negatives
        predicted_probabilities = torch.sigmoid(predicted_logits)
        probability_correct = torch.where(
            target_labels == 1.0,
            predicted_probabilities,
            1.0 - predicted_probabilities,
        )

        # alpha_t: class-balancing weight
        alpha_weight = torch.where(
            target_labels == 1.0,
            self.focal_alpha,
            1.0 - self.focal_alpha,
        )

        # FL(p_t) = alpha_t * (1 - p_t)^gamma * BCE(x, y)
        focal_weight = alpha_weight * ((1.0 - probability_correct) ** self.focal_gamma)
        focal_loss_per_track = focal_weight * bce_per_track  # (B, N)

        # Average over valid tracks only (exclude padding)
        valid_float = valid_mask.float()
        focal_loss_per_track = focal_loss_per_track * valid_float
        num_valid = valid_float.sum().clamp(min=1.0)
        return focal_loss_per_track.sum() / num_valid

    def _refine_top_candidates(
        self,
        combined_features: torch.Tensor,
        per_track_logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M2: Select top-K candidates and refine with self-attention.

        Physics motivation: The 3 tau pions have correlated properties:
            - Shared displaced vertex -> correlated dxy values
            - Invariant mass m_vis(3pi) < m_tau = 1.777 GeV
            - Collimated angular separation (close in eta, phi)
        These are pairwise/set-level features that per-track scoring
        fundamentally cannot capture. Self-attention on the top-K
        candidates enables this combinatorial reasoning.

        Signal-to-noise improvement: full track set has ~0.3% positive
        rate. Top-256 from ~1130 tracks concentrates positives to ~1.2%,
        a 4x improvement. Self-attention is applied only to these
        high-signal candidates, making it computationally tractable.

        Top-K selection uses detached scores: no gradient flows through
        the selection mechanism. Gradients from the refinement loss reach
        the feature pathway through the gathered combined_features.

        Args:
            combined_features: (B, combined_dim, P) features for all tracks.
            per_track_logits: (B, P) scores from per-track head.
            valid_mask: (B, P) boolean, True for valid tracks.

        Returns:
            refined_logits: (B, K) score adjustments for top-K candidates.
            top_indices: (B, K) indices of selected candidates in the P dim.
        """
        num_tracks = combined_features.shape[2]
        actual_k = min(self.num_refinement_candidates, num_tracks)

        # Select top-K candidates by per-track score (detached from gradient)
        # Padded positions are set to -inf so they rank last in topk.
        masked_scores = per_track_logits.clone()
        masked_scores[~valid_mask] = float('-inf')
        _, top_indices = masked_scores.detach().topk(actual_k, dim=1)  # (B, K)

        # Gather combined features at top-K positions
        # (B, combined_dim, P) -> gather -> (B, combined_dim, K)
        top_indices_expanded = top_indices.unsqueeze(1).expand(
            -1, self.combined_dim, -1,
        )  # (B, combined_dim, K)
        top_features = combined_features.gather(
            2, top_indices_expanded,
        )  # (B, combined_dim, K)
        top_features = top_features.transpose(1, 2)  # (B, K, combined_dim)

        # Project to refinement dimension + LayerNorm for stable attention
        projected = self.refinement_norm(
            self.refinement_projection(top_features),
        )  # (B, K, refinement_dim)

        # Build padding mask for self-attention: True = padded (ignored)
        # Handles the rare case where some events have fewer valid tracks than K.
        top_valid = valid_mask.gather(1, top_indices)  # (B, K)
        refinement_padding_mask = ~top_valid  # (B, K)

        # Self-attention: candidates attend to each other for pairwise reasoning.
        # Each candidate's refined score incorporates information from all other
        # candidates (e.g., consistent displacement, compatible momenta, etc.)
        for layer in self.refinement_layers:
            projected = layer(
                projected,
                src_key_padding_mask=refinement_padding_mask,
            )  # (B, K, refinement_dim)

        # Score adjustment for each candidate
        refined_logits = self.refinement_scorer(projected).squeeze(-1)  # (B, K)

        # Zero out padded positions (should not contribute to loss or ranking)
        refined_logits = refined_logits.masked_fill(~top_valid, 0.0)

        return refined_logits, top_indices

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: backbone -> neighbor scoring -> refinement -> loss/logits.

        Training returns a loss dict; inference returns per-track logits
        compatible with the existing recall@K evaluation pipeline.

        During inference, final scores combine per-track base scores with
        self-attention refinement adjustments (additive residual):
            score[i] = per_track_logits[i]                         if i not in top-K
            score[i] = per_track_logits[i] + refined_logits[i]     if i in top-K

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) standardized per-track features.
            lorentz_vectors: (B, 4, P) raw per-track 4-vectors [px, py, pz, E].
            mask: (B, 1, P) boolean mask, True for valid tracks.
            track_labels: (B, 1, P) binary labels (1.0 = tau pion).
                Required for training, None for inference.

        Returns:
            Training: {'total_loss', 'per_track_loss', 'refinement_loss'}
            Inference: {'per_track_logits': (B, P)} -- combined base + refinement.
        """
        # ---- Step 1: Backbone enrichment (frozen) ----
        with torch.no_grad():
            enriched_features = self.backbone.enrich(
                points, features, lorentz_vectors, mask,
            )  # (B, backbone_dim, P)
        enriched_features = enriched_features.detach()

        # ---- Step 2: Neighbor-aware messages (M1) ----
        neighbor_messages = self._compute_neighbor_messages(
            enriched_features, points, mask,
        )  # (B, message_dim, P)

        # ---- Step 3: Skip-connected physics features (M5) ----
        skip_features = self._compute_skip_features(
            features, lorentz_vectors, mask,
        )  # (B, 2, P)

        # ---- Step 4: Combine all feature sources ----
        # combined = [enriched(256), neighbor_messages(64), skip(2)] = (B, 322, P)
        combined_features = torch.cat(
            [enriched_features, neighbor_messages, skip_features], dim=1,
        )  # (B, combined_dim, P)

        # ---- Step 5: Per-track scoring ----
        per_track_logits = self.per_track_head(
            combined_features,
        ).squeeze(1)  # (B, P)

        # ---- Step 6: Top-K self-attention refinement (M2) ----
        valid_mask = mask.squeeze(1).bool()  # (B, P)
        refined_logits, top_indices = self._refine_top_candidates(
            combined_features, per_track_logits, valid_mask,
        )  # (B, K), (B, K)

        # ---- Training: compute losses ----
        if track_labels is not None:
            labels_flat = track_labels.squeeze(1) * mask.squeeze(1).float()  # (B, P)

            # Per-track focal BCE on ALL tracks (primary classification signal).
            # Provides gradient to every track (1130 signals per event).
            per_track_loss = self._focal_bce_loss(
                per_track_logits, labels_flat, valid_mask,
            )

            # Refinement focal BCE on top-K combined scores.
            # combined_top = per_track_logits[top-K] + refined_logits
            # This couples the two heads: gradients from refinement loss
            # flow to BOTH the refinement layers AND the per-track head,
            # providing extra training signal for the borderline candidates.
            per_track_at_top = per_track_logits.gather(1, top_indices)  # (B, K)
            combined_top_logits = per_track_at_top + refined_logits  # (B, K)

            top_labels = labels_flat.gather(1, top_indices)  # (B, K)
            top_valid = valid_mask.gather(1, top_indices)  # (B, K)
            refinement_loss = self._focal_bce_loss(
                combined_top_logits, top_labels, top_valid,
            )

            total_loss = (
                self.per_track_loss_weight * per_track_loss
                + self.refinement_loss_weight * refinement_loss
            )

            return {
                'total_loss': total_loss,
                'per_track_loss': per_track_loss,
                'refinement_loss': refinement_loss,
            }

        # ---- Inference: combine per-track + refinement scores ----
        # Additive residual: for top-K positions, add the refinement
        # adjustment. At initialization, refined_logits ~ 0, so ranking
        # starts with per_track_logits alone. As training progresses,
        # refinement provides positive adjustments for GT candidates
        # and negative adjustments for background candidates.
        final_logits = per_track_logits.clone()
        final_logits.scatter_add_(1, top_indices, refined_logits)

        return {
            'per_track_logits': final_logits,
        }
