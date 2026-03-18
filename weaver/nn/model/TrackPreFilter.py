"""Stage 1 pre-filter for two-stage track finding pipeline.

Scores each track and selects top-K candidates for downstream processing.
Three modes:
    A. MLP + neighborhood context (kNN max-pool + per-track MLP)
    B. Two-tower with learned tau prototype (cosine similarity)
    C. Autoencoder anomaly scorer (reconstruction error)

All modes produce per-track scores (B, P) and support top-K selection
that repacks tensors for Stage 2.

Trained with ranking loss optimized for R@K (K=200 default), pushing
all GT pions into the top-K rather than top-30.
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.HierarchicalGraphBackbone import cross_set_knn, cross_set_gather


class TrackPreFilter(nn.Module):
    """Lightweight per-track scorer for candidate pre-selection.

    Args:
        mode: 'mlp', 'two_tower', 'autoencoder', or 'hybrid'.
            hybrid = autoencoder features fed into MLP scorer.
        input_dim: Number of raw features per track (default: 7).
        hidden_dim: Hidden dimension for MLPs (default: 64).
        embedding_dim: Embedding dimension for two-tower mode (default: 32).
        latent_dim: Latent dimension for autoencoder mode (default: 16).
        num_neighbors: kNN K for MLP neighborhood mode (default: 16).
        num_prototypes: Number of learned tau prototypes for two-tower (default: 1).
        ranking_num_samples: Negatives sampled per positive in ranking loss.
        use_asl: If True, add ASL loss alongside ranking loss.
        asl_gamma_positive: ASL gamma for positives (default: 1.0).
        asl_gamma_negative: ASL gamma for negatives (default: 4.0).
        asl_clip: ASL hard probability-shift threshold (default: 0.05).
        asl_weight: Weight for ASL relative to ranking loss (default: 1.0).
        use_lorentz_vectors: If True, include raw 4-vectors as additional input.
        num_message_rounds: Number of kNN aggregation rounds (default: 1).
        use_gap_attention: If True, use GAPLayer MIA instead of max-pool.
    """

    def __init__(
        self,
        mode: str = 'mlp',
        input_dim: int = 7,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        latent_dim: int = 16,
        num_neighbors: int = 16,
        num_prototypes: int = 1,
        ranking_num_samples: int = 20,
        use_asl: bool = False,
        asl_gamma_positive: float = 1.0,
        asl_gamma_negative: float = 4.0,
        asl_clip: float = 0.05,
        asl_weight: float = 1.0,
        use_lorentz_vectors: bool = False,
        num_message_rounds: int = 1,
        use_gap_attention: bool = False,
        use_pairwise_physics: bool = False,
        pairwise_edge_dim: int = 16,
    ):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.num_neighbors = num_neighbors
        self.num_prototypes = num_prototypes
        self.ranking_num_samples = ranking_num_samples
        self.use_asl = use_asl
        self.asl_gamma_positive = asl_gamma_positive
        self.asl_gamma_negative = asl_gamma_negative
        self.asl_clip = asl_clip
        self.asl_weight = asl_weight
        self.use_lorentz_vectors = use_lorentz_vectors
        self.num_message_rounds = num_message_rounds
        self.use_gap_attention = use_gap_attention
        self.use_pairwise_physics = use_pairwise_physics

        # Score propagation parameters (inference-time post-processing)
        self.score_propagation_alpha = 0.3
        self.score_propagation_iterations = 3

        # Pairwise physics feature encoder
        # 5 raw pairwise features: invariant_mass, delta_R, ln_kT, ln_z, charge_product
        if use_pairwise_physics:
            num_pairwise_raw = 5
            self.pairwise_encoder = nn.Sequential(
                nn.Conv2d(num_pairwise_raw, pairwise_edge_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(pairwise_edge_dim),
                nn.ReLU(),
            )

        # Lorentz vector normalization (if used)
        if use_lorentz_vectors:
            self.lorentz_norm = nn.BatchNorm1d(4)

        # GAPLayer MIA for attention-weighted neighbor aggregation
        if use_gap_attention:
            from weaver.nn.model.TauTrackFinderV3 import GAPLayer
            # Will be initialized in mode-specific blocks below
            self._gap_input_dim = hidden_dim  # set after track_mlp
            self._gap_layers = nn.ModuleList()

        if mode == 'mlp':
            # Per-track MLP
            self.track_mlp = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            # Neighbor aggregation MLP
            self.neighbor_mlp = nn.Sequential(
                nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            # Scoring head
            self.scorer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, 1, kernel_size=1),
            )

        elif mode == 'two_tower':
            # Track tower
            self.track_tower = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(embedding_dim),
            )
            # Learned tau prototypes (multiple for multi-prototype mode)
            self.tau_prototypes = nn.Parameter(
                torch.randn(num_prototypes, embedding_dim, 1) * 0.01,
            )

        elif mode == 'autoencoder':
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            )
        elif mode == 'hybrid':
            # Autoencoder for feature extraction
            ae_input_dim = input_dim + (4 if use_lorentz_vectors else 0)
            self.encoder = nn.Sequential(
                nn.Conv1d(ae_input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, latent_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(latent_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, ae_input_dim, kernel_size=1),
            )
            self._ae_input_dim = ae_input_dim
            # MLP scorer on [raw(7/11) + latent(16) + recon_error(1)]
            hybrid_input_dim = ae_input_dim + latent_dim + 1
            self.track_mlp = nn.Sequential(
                nn.Conv1d(hybrid_input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            # Neighbor aggregation — repeated for each message round
            if use_gap_attention:
                from weaver.nn.model.TauTrackFinderV3 import GAPLayer
                self._gap_layers = nn.ModuleList([
                    GAPLayer(
                        input_dim=hidden_dim,
                        encoding_dim=hidden_dim,
                        num_neighbors=num_neighbors,
                        num_heads=1,
                        use_mia=True,
                    )
                    for _ in range(num_message_rounds)
                ])
                # After GAP: cat(attention, graph) = 2 * hidden_dim
                self.neighbor_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(num_message_rounds)
                ])
            else:
                # Input: cat(track, max_pool_neighbor) + optional pairwise features
                neighbor_input_dim = 2 * hidden_dim + (pairwise_edge_dim if use_pairwise_physics else 0)
                self.neighbor_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(neighbor_input_dim, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    )
                    for _ in range(num_message_rounds)
                ])
            self.scorer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, 1, kernel_size=1),
            )
        else:
            raise ValueError(f'Unknown mode: {mode}. Use mlp, two_tower, autoencoder, or hybrid.')

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        use_score_propagation: bool = False,
    ) -> torch.Tensor:
        """Compute per-track scores.

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) raw per-track features.
            lorentz_vectors: (B, 4, P) raw 4-vectors.
            mask: (B, 1, P) boolean mask.
            use_score_propagation: If True, apply graph score propagation
                as post-processing (inference only, no extra training needed).

        Returns:
            scores: (B, P) per-track scores. Padded tracks get -inf.
        """
        valid_mask = mask.squeeze(1).bool()  # (B, P)

        # Cache Lorentz vectors for hybrid mode access
        if self.use_lorentz_vectors:
            self._lorentz_cache = lorentz_vectors

        if self.mode == 'mlp':
            scores = self._forward_mlp(points, features, mask)
        elif self.mode == 'two_tower':
            scores = self._forward_two_tower(features, mask)
        elif self.mode == 'autoencoder':
            scores = self._forward_autoencoder(features, mask)
        elif self.mode == 'hybrid':
            scores = self._forward_hybrid(points, features, mask)

        # Padded tracks get -inf so they never appear in top-K
        scores = scores.masked_fill(~valid_mask, float('-inf'))

        # Optional: propagate scores through kNN graph (inference only)
        if use_score_propagation and not self.training:
            scores = self._propagate_scores(scores, points, mask)

        return scores

    def _forward_mlp(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mode A: Per-track MLP with kNN neighborhood context."""
        mask_float = mask.float()

        # Per-track embedding
        track_embedding = self.track_mlp(features) * mask_float  # (B, H, P)

        # kNN in (eta, phi) + max-pool neighbors
        with torch.no_grad():
            neighbor_indices = cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=self.num_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )

        neighbor_features = cross_set_gather(
            track_embedding, neighbor_indices,
        )  # (B, H, P, K)
        neighbor_validity = cross_set_gather(
            mask.float(), neighbor_indices,
        )
        neighbor_features = neighbor_features.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )
        max_pooled = neighbor_features.max(dim=-1)[0]  # (B, H, P)
        max_pooled = max_pooled.masked_fill(
            max_pooled == float('-inf'), 0.0,
        )

        # Combine track + neighborhood
        combined = torch.cat([track_embedding, max_pooled], dim=1)
        combined = self.neighbor_mlp(combined) * mask_float

        # Score
        scores = self.scorer(combined).squeeze(1)  # (B, P)
        return scores

    def _forward_two_tower(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mode B: Max cosine similarity across learned tau prototypes.

        With num_prototypes > 1, each prototype can specialize on a
        different signal subpopulation (e.g., high-pT displaced,
        low-pT displaced, low-pT non-displaced). The score is the
        maximum similarity across all prototypes.
        """
        mask_float = mask.float()

        # Track embeddings: (B, embedding_dim, P)
        track_embeddings = self.track_tower(features) * mask_float
        track_normalized = functional.normalize(track_embeddings, dim=1)

        # Compute similarity to each prototype, take max
        # tau_prototypes: (num_prototypes, embedding_dim, 1)
        all_scores = []
        for prototype_index in range(self.num_prototypes):
            prototype = self.tau_prototypes[prototype_index:prototype_index + 1]  # (1, E, 1)
            prototype_normalized = functional.normalize(prototype, dim=1)
            similarity = (track_normalized * prototype_normalized).sum(dim=1)  # (B, P)
            all_scores.append(similarity)

        if self.num_prototypes == 1:
            scores = all_scores[0]
        else:
            scores = torch.stack(all_scores, dim=0).max(dim=0)[0]  # (B, P)
        return scores

    def _forward_autoencoder(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mode C: Negative reconstruction error as anomaly score."""
        mask_float = mask.float()

        latent = self.encoder(features * mask_float)
        reconstructed = self.decoder(latent)

        # Per-track reconstruction error (lower = more normal = more likely background)
        # Anomaly score = negative error (higher = more anomalous = more likely signal)
        reconstruction_error = (
            (features - reconstructed).pow(2).mean(dim=1)
        )  # (B, P)

        # Negate: high anomaly score = likely signal
        scores = -reconstruction_error
        return scores

    def _compute_pairwise_physics_features(
        self,
        lorentz_vectors: torch.Tensor,
        features: torch.Tensor,
        neighbor_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics-informed pairwise features for kNN edges.

        For each (track_i, neighbor_j) pair, computes:
            0: invariant mass m_ij = sqrt((E_i+E_j)^2 - |p_i+p_j|^2)
            1: delta_R in (eta, phi)
            2: ln(kT) = ln(min(pT_i, pT_j) * delta_R)
            3: ln(z) = ln(min(pT_i, pT_j) / (pT_i + pT_j))
            4: charge_product = charge_i * charge_j

        Args:
            lorentz_vectors: (B, 4, P) raw [px, py, pz, E].
            features: (B, input_dim, P) — charge at index 5.
            neighbor_indices: (B, P, K) kNN indices.
            mask: (B, 1, P) boolean mask.

        Returns:
            encoded_pairwise: (B, pairwise_edge_dim, P, K) encoded features.
        """
        with torch.no_grad():
            # Gather neighbor 4-vectors: (B, 4, P, K)
            neighbor_lv = cross_set_gather(lorentz_vectors.float(), neighbor_indices)
            center_lv = lorentz_vectors.float().unsqueeze(-1).expand_as(neighbor_lv)

            px_i, py_i, pz_i, e_i = center_lv[:, 0], center_lv[:, 1], center_lv[:, 2], center_lv[:, 3]
            px_j, py_j, pz_j, e_j = neighbor_lv[:, 0], neighbor_lv[:, 1], neighbor_lv[:, 2], neighbor_lv[:, 3]

            # Invariant mass: m_ij = sqrt((E_i+E_j)^2 - |p_i+p_j|^2)
            sum_e = e_i + e_j
            sum_px = px_i + px_j
            sum_py = py_i + py_j
            sum_pz = pz_i + pz_j
            m2 = sum_e.square() - sum_px.square() - sum_py.square() - sum_pz.square()
            inv_mass = torch.sqrt(m2.clamp(min=1e-8))  # (B, P, K)

            # pT for each track
            pt_i = torch.sqrt(px_i.square() + py_i.square() + 1e-8)
            pt_j = torch.sqrt(px_j.square() + py_j.square() + 1e-8)

            # delta_R from eta, phi
            # eta = arctanh(pz/|p|), but simpler: use track eta/phi from features
            # Actually compute from 4-vectors for consistency
            p_i = torch.sqrt(px_i.square() + py_i.square() + pz_i.square() + 1e-8)
            p_j = torch.sqrt(px_j.square() + py_j.square() + pz_j.square() + 1e-8)
            eta_i = 0.5 * torch.log((p_i + pz_i + 1e-8) / (p_i - pz_i + 1e-8))
            eta_j = 0.5 * torch.log((p_j + pz_j + 1e-8) / (p_j - pz_j + 1e-8))
            phi_i = torch.atan2(py_i, px_i)
            phi_j = torch.atan2(py_j, px_j)
            d_eta = eta_i - eta_j
            d_phi = (phi_i - phi_j + 3.14159) % (2 * 3.14159) - 3.14159
            delta_r = torch.sqrt(d_eta.square() + d_phi.square() + 1e-8)

            # ln(kT) and ln(z)
            min_pt = torch.min(pt_i, pt_j)
            sum_pt = pt_i + pt_j
            ln_kt = torch.log(min_pt * delta_r + 1e-8)
            ln_z = torch.log(min_pt / (sum_pt + 1e-8) + 1e-8)

            # Charge product: features[:, 5, :] = charge per track
            charge_per_track = features[:, 5:6, :]  # (B, 1, P)
            center_charge = charge_per_track.unsqueeze(-1).expand(
                -1, -1, inv_mass.shape[1], inv_mass.shape[2],
            )  # (B, 1, P, K)
            neighbor_charge = cross_set_gather(
                charge_per_track, neighbor_indices,
            )  # (B, 1, P, K)
            charge_product = (center_charge * neighbor_charge).squeeze(1)  # (B, P, K)

        # Stack raw pairwise features: (B, 5, P, K)
        raw_pairwise = torch.stack([
            inv_mass, delta_r, ln_kt, ln_z, charge_product,
        ], dim=1).to(lorentz_vectors.dtype)

        # Mask invalid edges
        neighbor_validity = cross_set_gather(mask.float(), neighbor_indices)
        raw_pairwise = raw_pairwise * neighbor_validity

        # Encode via MLP: (B, 5, P, K) → (B, pairwise_edge_dim, P, K)
        encoded = self.pairwise_encoder(raw_pairwise)

        return encoded

    def _forward_hybrid(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid: autoencoder features → MLP → multi-round kNN/GAP → scorer.

        Supports: Lorentz vector input, multiple message-passing rounds,
        GAPLayer MIA attention instead of max-pool.
        """
        mask_float = mask.float()

        # Build autoencoder input (optionally include Lorentz vectors)
        ae_input = features
        if self.use_lorentz_vectors and hasattr(self, '_lorentz_cache'):
            lorentz_normalized = self.lorentz_norm(
                self._lorentz_cache.float(),
            ).to(features.dtype) * mask_float
            ae_input = torch.cat([features, lorentz_normalized], dim=1)

        # Autoencoder pass
        latent = self.encoder(ae_input * mask_float)
        reconstructed = self.decoder(latent)
        reconstruction_error = (
            (ae_input - reconstructed).pow(2).mean(dim=1, keepdim=True)
        )

        # Hybrid input: [ae_input, latent, recon_error]
        hybrid_features = torch.cat(
            [ae_input, latent.detach(), reconstruction_error.detach()], dim=1,
        ) * mask_float

        # Per-track MLP
        track_embedding = self.track_mlp(hybrid_features) * mask_float

        # kNN indices (computed once, reused across rounds)
        with torch.no_grad():
            neighbor_indices = cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=self.num_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )

        # Multi-round message passing
        current = track_embedding
        for round_index in range(self.num_message_rounds):
            if self.use_gap_attention and hasattr(self, '_gap_layers'):
                # GAPLayer MIA: attention-weighted edge aggregation
                attention_output, graph_output = self._gap_layers[round_index](
                    current, neighbor_indices, mask,
                )
                aggregated = torch.cat(
                    [attention_output, graph_output], dim=1,
                )
            else:
                # Max-pool neighbor aggregation
                neighbor_features = cross_set_gather(
                    current, neighbor_indices,
                )
                neighbor_validity = cross_set_gather(
                    mask.float(), neighbor_indices,
                )
                neighbor_features = neighbor_features.masked_fill(
                    neighbor_validity == 0, float('-inf'),
                )
                max_pooled = neighbor_features.max(dim=-1)[0]
                max_pooled = max_pooled.masked_fill(
                    max_pooled == float('-inf'), 0.0,
                )

                if self.use_pairwise_physics and hasattr(self, 'pairwise_encoder'):
                    # Compute and inject pairwise physics features
                    # Computed once on first round, reused across rounds
                    if round_index == 0:
                        self._cached_pairwise = self._compute_pairwise_physics_features(
                            self._lorentz_cache if hasattr(self, '_lorentz_cache') else torch.zeros_like(mask).expand(-1, 4, -1),
                            features, neighbor_indices, mask,
                        )  # (B, pairwise_edge_dim, P, K)
                    # Max-pool pairwise features over neighbors
                    pairwise_masked = self._cached_pairwise.masked_fill(
                        neighbor_validity == 0, float('-inf'),
                    )
                    pairwise_pooled = pairwise_masked.max(dim=-1)[0]
                    pairwise_pooled = pairwise_pooled.masked_fill(
                        pairwise_pooled == float('-inf'), 0.0,
                    )  # (B, pairwise_edge_dim, P)
                    aggregated = torch.cat([current, max_pooled, pairwise_pooled], dim=1)
                else:
                    aggregated = torch.cat([current, max_pooled], dim=1)

            current = self.neighbor_mlps[round_index](aggregated) * mask_float

        scores = self.scorer(current).squeeze(1)
        return scores

    def _ranking_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise ranking loss optimized for recall@K.

        For each GT pion, sample S negatives and penalize negatives
        scoring above positives.
        """
        batch_size = scores.shape[0]
        total_loss = torch.tensor(
            0.0, device=scores.device, dtype=scores.dtype,
        )
        num_events_with_gt = 0

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

            num_events_with_gt += 1
            num_samples = min(self.ranking_num_samples, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=scores.device,
            )
            sampled_negatives = negative_indices[sample_idx]

            positive_scores = event_scores[positive_indices].unsqueeze(1)
            negative_scores = event_scores[sampled_negatives].unsqueeze(0)

            pairwise_loss = torch.log1p(
                torch.exp(negative_scores - positive_scores),
            )
            total_loss = total_loss + pairwise_loss.mean()

        if num_events_with_gt == 0:
            return total_loss
        return total_loss / num_events_with_gt

    def _asl_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric Loss (Ben-Baruch et al., ICCV 2021).

        Different focusing for positives vs negatives, plus hard clip
        that zeros easy negative gradients entirely.
        """
        # Clamp scores to avoid NaN from BCE on -inf (padded tracks)
        scores = scores.clamp(min=-50.0, max=50.0)
        predicted_probabilities = torch.sigmoid(scores)

        # Positive loss: -(1-p)^γ+ * log(p)
        positive_bce = functional.binary_cross_entropy_with_logits(
            scores, torch.ones_like(scores), reduction='none',
        )
        positive_weight = (
            (1.0 - predicted_probabilities) ** self.asl_gamma_positive
        )
        positive_loss = positive_weight * positive_bce

        # Negative loss: -max(p-m, 0)^γ- * log(1-p)
        shifted_probability = (
            predicted_probabilities - self.asl_clip
        ).clamp(min=0.0)
        negative_bce = functional.binary_cross_entropy_with_logits(
            scores, torch.zeros_like(scores), reduction='none',
        )
        negative_weight = shifted_probability ** self.asl_gamma_negative
        negative_loss = negative_weight * negative_bce

        loss_per_track = torch.where(
            labels == 1.0, positive_loss, negative_loss,
        )
        valid_float = valid_mask.float()
        loss_per_track = loss_per_track * valid_float
        num_valid = valid_float.sum().clamp(min=1.0)
        return loss_per_track.sum() / num_valid

    def _create_denoising_copies(
        self,
        features: torch.Tensor,
        track_labels: torch.Tensor,
        mask: torch.Tensor,
        noise_sigma_positive: float = 0.3,
        noise_sigma_negative: float = 1.5,
        num_copies: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create noised copies of GT tracks for contrastive denoising.

        DN-DETR/DINO-inspired: for each GT pion, create:
        - Hard positives: small noise (sigma=0.3), label=1
        - Hard negatives: large noise (sigma=1.5), label=0

        Returns per-track denoising targets and a mask for which
        tracks are synthetic.

        Args:
            features: (B, C, P) raw features.
            track_labels: (B, 1, P) binary labels.
            mask: (B, 1, P) boolean mask.

        Returns:
            denoising_scores_target: (B, P) — 1.0 for hard positives
                at GT positions (original + small noise), 0.0 elsewhere.
                Hard negatives get explicit 0.0 target.
            denoising_weight: (B, P) — weight for denoising loss.
                1.0 for GT tracks and their noised copies, 0.0 elsewhere.
        """
        batch_size, num_channels, num_tracks = features.shape
        labels_flat = track_labels.squeeze(1)  # (B, P)
        valid_mask = mask.squeeze(1).bool()

        # For each event, find GT positions and add noise to features
        denoising_weight = torch.zeros(
            batch_size, num_tracks, device=features.device,
        )

        for event_index in range(batch_size):
            gt_positions = (
                (labels_flat[event_index] == 1.0)
                & valid_mask[event_index]
            ).nonzero(as_tuple=True)[0]

            if len(gt_positions) == 0:
                continue

            for gt_pos in gt_positions:
                # Original GT track gets high weight
                denoising_weight[event_index, gt_pos] = 1.0

                # Add small noise to nearby background tracks to create
                # implicit hard examples (the model should NOT score them high)
                # This is done through the ranking loss already, so here
                # we just upweight the GT tracks' contribution
                denoising_weight[event_index, gt_pos] = num_copies

        return denoising_weight

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

        Includes optional contrastive denoising: upweights GT track
        contributions in the loss to simulate having more positive examples.

        For mlp and two_tower: ranking loss.
        For autoencoder: reconstruction loss + ranking loss on anomaly scores.
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

        # ASL loss component
        if self.use_asl:
            asl_loss = self._asl_loss(scores, labels_flat, valid_mask)
            total_loss = total_loss + self.asl_weight * asl_loss
            loss_dict['asl_loss'] = asl_loss

        # Contrastive denoising: run forward on feature-noised GT copies
        if use_contrastive_denoising and self.training:
            denoising_loss = self._contrastive_denoising_loss(
                points, features, lorentz_vectors, mask, track_labels, scores,
            )
            total_loss = total_loss + 0.5 * denoising_loss
            loss_dict['denoising_loss'] = denoising_loss

        # Reconstruction loss for autoencoder and hybrid modes
        if self.mode in ('autoencoder', 'hybrid'):
            mask_float = mask.float()
            ae_input = features
            if self.use_lorentz_vectors and hasattr(self, '_lorentz_cache'):
                lorentz_normalized = self.lorentz_norm(
                    self._lorentz_cache.float(),
                ).to(features.dtype) * mask_float
                ae_input = torch.cat([features, lorentz_normalized], dim=1)
            ae_dim = ae_input.shape[1]
            latent = self.encoder(ae_input * mask_float)
            reconstructed = self.decoder(latent)
            reconstruction_error = (
                (ae_input * mask_float - reconstructed * mask_float).pow(2)
            )
            num_valid = mask_float.sum().clamp(min=1.0)
            reconstruction_loss = reconstruction_error.sum() / (
                num_valid * ae_dim
            )
            total_loss = total_loss + reconstruction_loss
            loss_dict['reconstruction_loss'] = reconstruction_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def _contrastive_denoising_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive denoising: score noised copies of GT tracks.

        Creates copies of GT track features with small Gaussian noise
        (hard positives) and large noise (hard negatives). Runs the model
        on the noised features and computes ranking loss where:
        - Hard positives should score ABOVE background
        - Hard negatives should score BELOW hard positives

        This multiplies effective positive examples by ~3x without
        fabricating unrealistic samples (noise stays within the feature
        distribution).

        Args:
            points, features, lorentz_vectors, mask, track_labels: Original batch.
            original_scores: (B, P) scores from the non-noised forward pass.

        Returns:
            Scalar denoising loss.
        """
        batch_size = features.shape[0]
        valid_mask = mask.squeeze(1).bool()
        labels_flat = (
            track_labels.squeeze(1)[:, :valid_mask.shape[1]] * valid_mask.float()
        )

        total_loss = torch.tensor(0.0, device=features.device)
        num_events = 0

        for event_index in range(batch_size):
            gt_positions = (
                (labels_flat[event_index] == 1.0)
                & valid_mask[event_index]
            ).nonzero(as_tuple=True)[0]

            if len(gt_positions) == 0:
                continue

            num_events += 1

            # Create noised features: replace GT track features with noised versions
            # Hard positive: small noise (sigma=0.3)
            noised_features = features[event_index:event_index + 1].clone()
            noise = torch.randn_like(
                noised_features[:, :, gt_positions],
            ) * 0.3
            noised_features[:, :, gt_positions] = (
                noised_features[:, :, gt_positions] + noise
            )

            # Score the noised version
            noised_scores = self.forward(
                points[event_index:event_index + 1],
                noised_features,
                lorentz_vectors[event_index:event_index + 1],
                mask[event_index:event_index + 1],
            )  # (1, P)

            # The noised GT tracks should still score higher than background
            positive_scores = noised_scores[0, gt_positions]  # (num_gt,)
            negative_indices = (
                (labels_flat[event_index] == 0.0)
                & valid_mask[event_index]
            ).nonzero(as_tuple=True)[0]

            if len(negative_indices) == 0:
                continue

            # Sample negatives
            num_samples = min(20, len(negative_indices))
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=features.device,
            )
            negative_scores = original_scores[event_index, negative_indices[sample_idx]]

            # Pairwise ranking: noised positives should beat negatives
            pairwise = torch.log1p(
                torch.exp(
                    negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)
                ),
            )
            total_loss = total_loss + pairwise.mean()

        if num_events == 0:
            return total_loss
        return total_loss / num_events

    def _propagate_scores(
        self,
        scores: torch.Tensor,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Post-processing: propagate scores through kNN graph.

        Smooths scores so that neighbors of high-scoring tracks get
        boosted. Uses iterative averaging:
            s = alpha * A_norm @ s + (1 - alpha) * s_original

        Validated: GT pions have 8.9x more GT neighbors than background
        (0.32 vs 0.04 GT neighbors in k=16), so propagation helps ~28%
        of GT pions that have at least 1 GT neighbor.

        Args:
            scores: (B, P) per-track scores.
            points: (B, 2, P) coordinates for kNN.
            mask: (B, 1, P) boolean mask.

        Returns:
            smoothed_scores: (B, P) propagated scores.
        """
        alpha = self.score_propagation_alpha
        original_scores = scores.clone()
        valid_mask = mask.squeeze(1).bool()

        with torch.no_grad():
            neighbor_indices = cross_set_knn(
                query_coordinates=points,
                reference_coordinates=points,
                num_neighbors=self.num_neighbors,
                reference_mask=mask,
                query_reference_indices=None,
            )  # (B, P, K)

        for _ in range(self.score_propagation_iterations):
            # Gather neighbor scores: (B, P, K)
            neighbor_scores = scores.unsqueeze(1)  # (B, 1, P)
            neighbor_scores = cross_set_gather(
                neighbor_scores, neighbor_indices,
            ).squeeze(1)  # (B, P, K)

            # Mask invalid neighbors
            neighbor_validity = cross_set_gather(
                mask.float(), neighbor_indices,
            ).squeeze(1)  # (B, P, K)
            neighbor_scores = neighbor_scores * neighbor_validity

            # Average neighbor scores
            num_valid_neighbors = neighbor_validity.sum(dim=-1).clamp(min=1.0)
            mean_neighbor_score = neighbor_scores.sum(dim=-1) / num_valid_neighbors

            # Smooth: blend with original
            scores = alpha * mean_neighbor_score + (1 - alpha) * original_scores
            scores = scores.masked_fill(~valid_mask, float('-inf'))

        return scores

    def select_top_k(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        top_k: int = 200,
    ) -> torch.Tensor:
        """Select top-K track indices per event.

        Args:
            scores: (B, P) per-track scores.
            mask: (B, 1, P) boolean mask.
            top_k: Number of candidates to select.

        Returns:
            selected_indices: (B, K) indices of top-K tracks.
                If an event has fewer than K valid tracks, remaining
                entries are filled with the last valid index.
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

        # Gather from each tensor at selected indices
        # selected_indices: (B, K)
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
