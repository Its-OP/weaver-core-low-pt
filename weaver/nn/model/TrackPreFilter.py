"""Stage 1 pre-filter for two-stage track finding pipeline.

Scores each track and selects top-K candidates for downstream processing.
Modes:
    A. MLP + neighborhood context (kNN max-pool + per-track MLP)
    B. Two-tower with learned tau prototype (cosine similarity)
    C. Autoencoder anomaly scorer (reconstruction error)
    D. Hybrid: autoencoder features fed into MLP scorer

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
        use_lorentz_vectors: bool = False,
        num_message_rounds: int = 1,
        use_gap_attention: bool = False,
        denoising_sigma_start: float = 0.3,
        denoising_sigma_end: float = 0.3,
        ranking_temperature_start: float = 1.0,
        ranking_temperature_end: float = 1.0,
        drw_warmup_fraction: float = 1.0,
        drw_positive_weight: float = 1.0,
        aggregation_mode: str = 'max',
        focal_gamma: float = 0.0,
        contrastive_denoising_negative_sigma: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.num_neighbors = num_neighbors
        self.num_prototypes = num_prototypes
        self.ranking_num_samples = ranking_num_samples
        self.use_lorentz_vectors = use_lorentz_vectors
        self.num_message_rounds = num_message_rounds
        self.use_gap_attention = use_gap_attention
        # Dropout rate for the mlp-mode MLP hidden layers. 0.0 preserves the
        # pre-2026-04-07 behavior (zero dropout, the regime where the
        # dim256+cutoff run overfit at ~2pp val/train gap). Only the
        # mlp-mode branch currently plumbs this through — other modes
        # (two_tower/autoencoder/hybrid) retain their original architecture.
        self.dropout = dropout

        # Temperature scheduling (Kukleva et al., ICLR 2023):
        # σ(t) linearly interpolates between sigma_start and sigma_end.
        # T(t) linearly interpolates between temperature_start and temperature_end.
        # High T → smooth gradients from many pairs; low T → sharp focus on hard violations.
        self.denoising_sigma_start = denoising_sigma_start
        self.denoising_sigma_end = denoising_sigma_end
        self.ranking_temperature_start = ranking_temperature_start
        self.ranking_temperature_end = ranking_temperature_end
        self._temperature_progress: float = 0.0

        # Deferred Re-Weighting (Cao et al., NeurIPS 2019):
        # Train with uniform weights for drw_warmup_fraction of training,
        # then upweight positive-negative pairs by drw_positive_weight.
        self.drw_warmup_fraction = drw_warmup_fraction
        self.drw_positive_weight = drw_positive_weight
        self._drw_active: bool = False

        # PNA multi-aggregation (Corso et al., NeurIPS 2020):
        # 'max' = standard max-pool, 'pna' = cat([mean, max, min, std])
        self.aggregation_mode = aggregation_mode

        # Equalized focal weighting (Li et al., CVPR 2022):
        # 0.0 = disabled, >0 = smooth (1-p)^γ modulation on pairwise loss.
        # Unlike ASL (which zeroed gradients), focal never zeros any gradient.
        self.focal_gamma = focal_gamma

        # DINO-style contrastive denoising (Zhang et al., ICLR 2023):
        # 0.0 = disabled (positive copies only, current behavior).
        # >0 = also create negative copies with this sigma, must be rejected.
        self.contrastive_denoising_negative_sigma = contrastive_denoising_negative_sigma

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
            # Dropout insertion is CONDITIONAL on `dropout > 0`, not
            # unconditional. nn.Dropout has no parameters, but adding it
            # to a Sequential still occupies an index — which shifts every
            # subsequent Conv1d / BatchNorm1d state_dict key. Pre-2026-04-07
            # checkpoints were trained without dropout and have keys like
            # `track_mlp.3.weight` (second Conv1d at index 3) and
            # `scorer.3.weight` (final Conv1d at index 3). Preserving the
            # zero-dropout code path means those checkpoints still load
            # cleanly into a `dropout=0.0` model (the class default and
            # what `recall_at_k_sweep.load_prefilter_from_checkpoint` uses).
            # When `dropout > 0` the user is training from scratch, so
            # losing backward compat is acceptable.
            use_dropout = dropout > 0

            # --- Per-track MLP ---
            # Standard BN→ReLU→Dropout ordering for two Conv1d blocks.
            track_mlp_layers: list[nn.Module] = [
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
            if use_dropout:
                track_mlp_layers.append(nn.Dropout(p=dropout))
            track_mlp_layers += [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
            if use_dropout:
                track_mlp_layers.append(nn.Dropout(p=dropout))
            self.track_mlp = nn.Sequential(*track_mlp_layers)

            # --- Neighbor aggregation, one Sequential per message round ---
            if aggregation_mode == 'pna':
                # PNA: cat([current, mean, max, min, std]) = 5 * hidden_dim
                neighbor_input_dim = 5 * hidden_dim
            else:
                # Standard: cat([current, max_pooled]) = 2 * hidden_dim
                neighbor_input_dim = 2 * hidden_dim

            def _build_neighbor_mlp() -> nn.Sequential:
                layers: list[nn.Module] = [
                    nn.Conv1d(
                        neighbor_input_dim, hidden_dim, kernel_size=1, bias=False,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
                if use_dropout:
                    layers.append(nn.Dropout(p=dropout))
                return nn.Sequential(*layers)

            self.neighbor_mlps = nn.ModuleList([
                _build_neighbor_mlp() for _ in range(num_message_rounds)
            ])

            # --- Scoring head ---
            # Dropout goes after the middle ReLU only. The final
            # Conv1d(hidden_dim → 1) is the output layer and MUST NOT have
            # a Dropout before it in the zero-dropout path (that would
            # shift its key from scorer.3 to scorer.4 and break backward
            # compat). With `use_dropout=True` we accept the shift — the
            # new run is training from scratch anyway.
            scorer_layers: list[nn.Module] = [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ]
            if use_dropout:
                scorer_layers.append(nn.Dropout(p=dropout))
            scorer_layers.append(nn.Conv1d(hidden_dim, 1, kernel_size=1))
            self.scorer = nn.Sequential(*scorer_layers)

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
                neighbor_input_dim = 2 * hidden_dim
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
    ) -> torch.Tensor:
        """Compute per-track scores.

        Args:
            points: (B, 2, P) coordinates in (eta, phi).
            features: (B, input_dim, P) raw per-track features.
            lorentz_vectors: (B, 4, P) raw 4-vectors.
            mask: (B, 1, P) boolean mask.

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

        return scores

    def _pna_aggregate(
        self,
        neighbor_features: torch.Tensor,
        neighbor_validity: torch.Tensor,
    ) -> torch.Tensor:
        """PNA multi-aggregation over kNN neighbors.

        Computes 4 aggregation functions in parallel (Corso et al., NeurIPS 2020):
            mean_j(h_j), max_j(h_j), min_j(h_j), std_j(h_j)
        where j ranges over valid neighbors.

        Args:
            neighbor_features: (B, H, P, K) features gathered from neighbors.
            neighbor_validity: (B, 1, P, K) validity mask for neighbors.

        Returns:
            aggregated: (B, 4*H, P) concatenation of [mean, max, min, std].
        """
        num_valid = neighbor_validity.sum(dim=-1).clamp(min=1.0)

        masked_for_mean = neighbor_features * neighbor_validity
        mean_aggregated = masked_for_mean.sum(dim=-1) / num_valid

        masked_for_max = neighbor_features.masked_fill(
            neighbor_validity == 0, float('-inf'),
        )
        max_aggregated = masked_for_max.max(dim=-1)[0]
        max_aggregated = max_aggregated.masked_fill(
            max_aggregated == float('-inf'), 0.0,
        )

        masked_for_min = neighbor_features.masked_fill(
            neighbor_validity == 0, float('inf'),
        )
        min_aggregated = masked_for_min.min(dim=-1)[0]
        min_aggregated = min_aggregated.masked_fill(
            min_aggregated == float('inf'), 0.0,
        )

        # std = sqrt(E[x²] - E[x]² + ε)
        mean_of_squares = (masked_for_mean ** 2).sum(dim=-1) / num_valid
        std_aggregated = (
            (mean_of_squares - mean_aggregated ** 2).clamp(min=1e-6).sqrt()
        )

        return torch.cat(
            [mean_aggregated, max_aggregated, min_aggregated, std_aggregated],
            dim=1,
        )

    def _forward_mlp(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mode A: Per-track MLP with multi-round kNN neighborhood context."""
        mask_float = mask.float()

        # Per-track embedding
        track_embedding = self.track_mlp(features) * mask_float  # (B, H, P)

        # kNN indices in (eta, phi), computed once and reused across rounds
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
            neighbor_features = cross_set_gather(
                current, neighbor_indices,
            )
            neighbor_validity = cross_set_gather(
                mask.float(), neighbor_indices,
            )

            if self.aggregation_mode == 'pna':
                # PNA: cat([current, mean, max, min, std]) → (B, 5*H, P)
                pna_aggregated = self._pna_aggregate(
                    neighbor_features, neighbor_validity,
                )
                aggregated = torch.cat([current, pna_aggregated], dim=1)
            else:
                # Standard max-pool: cat([current, max_pooled]) → (B, 2*H, P)
                neighbor_features = neighbor_features.masked_fill(
                    neighbor_validity == 0, float('-inf'),
                )
                max_pooled = neighbor_features.max(dim=-1)[0]
                max_pooled = max_pooled.masked_fill(
                    max_pooled == float('-inf'), 0.0,
                )
                aggregated = torch.cat([current, max_pooled], dim=1)

            current = self.neighbor_mlps[round_index](aggregated) * mask_float

        # Score
        scores = self.scorer(current).squeeze(1)  # (B, P)
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
                # Gather neighbor features: (B, hidden_dim, P, K)
                neighbor_features = cross_set_gather(
                    current, neighbor_indices,
                )
                neighbor_validity = cross_set_gather(
                    mask.float(), neighbor_indices,
                )

                # Standard max-pool aggregation
                neighbor_features = neighbor_features.masked_fill(
                    neighbor_validity == 0, float('-inf'),
                )
                max_pooled = neighbor_features.max(dim=-1)[0]
                max_pooled = max_pooled.masked_fill(
                    max_pooled == float('-inf'), 0.0,
                )
                aggregated = torch.cat(
                    [current, max_pooled], dim=1,
                )

            current = self.neighbor_mlps[round_index](aggregated) * mask_float

        scores = self.scorer(current).squeeze(1)
        return scores

    # ---- Training schedule methods ----

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
        """Current noise sigma for contrastive denoising, interpolated by progress."""
        return (
            self.denoising_sigma_start
            + self._temperature_progress
            * (self.denoising_sigma_end - self.denoising_sigma_start)
        )

    @property
    def current_ranking_temperature(self) -> float:
        """Current temperature for ranking loss, interpolated by progress."""
        return (
            self.ranking_temperature_start
            + self._temperature_progress
            * (self.ranking_temperature_end - self.ranking_temperature_start)
        )

    def set_drw_active(self, active: bool) -> None:
        """Activate/deactivate Deferred Re-Weighting of positive samples.

        DRW (Cao et al. 2019): train with uniform weights first to learn
        representations, then upweight rare-class samples for ranking.

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
        """Temperature-scaled pairwise ranking loss optimized for recall@K.

        For each GT pion, sample S negatives and penalize negatives
        scoring above positives:
            L = T * softplus((s_neg - s_pos) / T)

        At T=1.0, this reduces to standard softplus(s_neg - s_pos).
        High T smooths gradients across more pairs; low T sharpens
        focus on hard violations near the decision boundary.

        When DRW is active, the loss is scaled by drw_positive_weight
        to upweight the rare positive class.

        # TODO: Vectorize this loop across the batch using padded positives
        # and a single torch.randint call. The per-event loop is kept for now
        # because the main cost savings come from the vectorized contrastive
        # denoising loss (Opt 2), and this loop operates on tiny tensors.
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
            # At T=1: standard softplus. At T>1: smoother. At T<1: sharper.
            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)

            # Equalized focal weighting (Li et al., CVPR 2022):
            # w = (1 - p)^γ where p = σ(s_pos - s_neg) = prob of correct ordering.
            # Easy pairs (large margin, p→1): downweighted. Hard pairs: full weight.
            # .detach() prevents focal weights from generating own gradients.
            # Unlike ASL's hard clip, this NEVER zeros any gradient.
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
        loss_dict['_scores'] = scores
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
        """DINO-style contrastive denoising loss.

        Creates two types of noised GT track copies (Zhang et al., ICLR 2023):
        1. Positive copies (small noise, scheduled σ): must score ABOVE background.
        2. Negative copies (large noise, σ_neg): must score BELOW positive copies.
           Teaches the decision boundary between real pions and "almost-pions."

        Vectorized: batched forward passes, per-event loop only for indexing.

        Args:
            points, features, lorentz_vectors, mask, track_labels: Original batch.
            original_scores: (B, P) scores from the non-noised forward pass.

        Returns:
            Scalar denoising loss (positive part + negative part if enabled).
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

        # --- Per-event loss (cheap — no forward passes) ---
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
            bg_scores = original_scores[event_index, negative_indices[sample_idx]]

            # Loss 1: positive copies should beat background
            positive_pairwise = temperature * functional.softplus(
                (bg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)) / temperature,
            )
            event_losses.append(positive_pairwise.mean())

            # Loss 2 (DINO): negative copies should score BELOW positive copies
            if use_negative_copies:
                neg_scores = negative_noised_scores[event_index, gt_positions]
                negative_pairwise = temperature * functional.softplus(
                    (neg_scores.unsqueeze(1) - pos_scores.unsqueeze(0))
                    / temperature,
                )
                event_losses.append(negative_pairwise.mean())

        if not event_losses:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        return torch.stack(event_losses).mean()

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
