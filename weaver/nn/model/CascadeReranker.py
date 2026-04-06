"""Stage 2 reranker: ParT-style pairwise-bias self-attention encoder.

Reuses ParticleTransformer components (PairEmbed, Block, Embed) from the
existing codebase. The key difference from standard ParT:
    - Per-track scoring head instead of CLS token classification
    - Stage 1 scores concatenated as an extra input feature
    - Trained with ranking loss (same as TrackPreFilter)

Architecture:
    1. Input embedding: cat(features, stage1_score) → MLP → (P, B, embed_dim)
    2. Pairwise features: lorentz_vectors → PairEmbed → (B*H, P, P) attention bias
       Pairwise physics features: ln kT, ln z, ln ΔR, ln m²
       These encode track-track relationships (e.g. ρ→ππ invariant mass).
    3. N transformer blocks with pairwise bias in attention:
       Attn(Q,K,V) = softmax(QK^T / √d_k + pairwise_bias) × V
    4. Per-track scoring MLP: encoded embedding → scalar score

Reference: Qu et al., ICML 2022 (arXiv:2202.03772) — Particle Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.ParticleTransformer import (
    Block,
    Embed,
    PairEmbed,
)


class CascadeReranker(nn.Module):
    """ParT-style pairwise-bias encoder for Stage 2 track reranking.

    Args:
        input_dim: Number of per-track features (default: 16).
        embed_dim: Transformer embedding dimension (default: 128).
        num_heads: Number of attention heads (default: 4).
        num_layers: Number of transformer blocks (default: 3).
        pair_input_dim: Number of pairwise Lorentz vector features (default: 4).
            4 = ln kT, ln z, ln ΔR, ln m².
        pair_embed_dims: MLP dims for pairwise feature embedding.
        ffn_ratio: Feed-forward expansion ratio in transformer blocks.
        dropout: Dropout rate in transformer blocks.
        ranking_num_samples: Negatives sampled per positive in ranking loss.
        ranking_temperature: Temperature for ranking loss.
    """

    def __init__(
        self,
        input_dim: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        pair_input_dim: int = 4,
        pair_extra_dim: int = 0,
        pair_embed_dims: list[int] | None = None,
        pair_embed_mode: str = 'concat',
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        ranking_num_samples: int = 50,
        ranking_temperature: float = 1.0,
        loss_mode: str = 'pairwise',
        rs_at_k_target: int = 200,
        rs_at_k_tau1: float = 1.0,
        rs_at_k_tau2: float = 1.0,
        use_contrastive_denoising: bool = False,
        denoising_sigma_start: float = 0.3,
        denoising_sigma_end: float = 0.05,
        denoising_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.ranking_num_samples = ranking_num_samples
        self.ranking_temperature = ranking_temperature
        self.pair_extra_dim = pair_extra_dim
        self.loss_mode = loss_mode
        self.rs_at_k_target = rs_at_k_target
        self.rs_at_k_tau1 = rs_at_k_tau1
        self.rs_at_k_tau2 = rs_at_k_tau2
        # For hybrid_lambda mode: pure pairwise until warmup_start, then ramp
        # LambdaRank to full weight by warmup_end.
        # Default: start at progress=0.4 (epoch 40/100), full by 0.7 (epoch 70/100).
        self.lambda_rank_warmup_start = 0.4
        self.lambda_rank_warmup_end = 0.7
        # Contrastive denoising auxiliary loss — ported from TrackPreFilter
        # (Zhang et al. ICLR 2023, DINO-style). GT track features get small
        # Gaussian noise (scheduled σ: start → end); a second forward pass
        # scores the noised batch; the noised GT scores must still beat
        # original-pass background scores. Helps regularize late-epoch
        # overfitting. σ is interpolated using _training_progress, the same
        # mechanism hybrid_lambda uses (set by train_cascade.py each epoch).
        self.use_contrastive_denoising = use_contrastive_denoising
        self.denoising_sigma_start = denoising_sigma_start
        self.denoising_sigma_end = denoising_sigma_end
        self.denoising_loss_weight = denoising_loss_weight
        self._training_progress: float = 0.0

        if pair_embed_dims is None:
            pair_embed_dims = [64, 64]

        # Input embedding: cat(features, stage1_score, z_pt) → embed_dim
        # +1 for stage1_score, +1 for energy sharing fraction z_pt
        self.embed = Embed(
            input_dim + 2,
            dims=[embed_dim],
            normalize_input=True,
        )

        # Pairwise feature embedding: lorentz_vectors (+ physics features) → attention bias
        # PairEmbed outputs (B*num_heads, P, P) used as attn_mask in Block.
        #
        # pair_input_dim: Lorentz-vector features (ln kT, ln z, ln ΔR, ln m²).
        # pair_extra_dim: Physics pairwise features passed via `uu` argument:
        #   ch 1: charge_product q_i × q_j — rho→π⁺π⁻ requires OS pair
        #   ch 2: |Δdz_sig| — shared tau vertex in longitudinal plane
        #   ch 3: rho_indicator exp(-(m_ij − 770)² / 2σ²) — ρ(770) mass resonance
        #   ch 4: rho_os_indicator (OS × rho_indicator) — conjunction
        #   ch 5: dxy_phi_corrected |Δdxy| / |2sin(Δφ/2)| — φ-corrected vertex compat
        self.pair_embed = PairEmbed(
            pairwise_lv_dim=pair_input_dim,
            pairwise_input_dim=pair_extra_dim,
            dims=pair_embed_dims + [num_heads],
            remove_self_pair=False,
            use_pre_activation_pair=True,
            mode=pair_embed_mode,
        )

        # Transformer blocks with pairwise attention bias
        block_config = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            attn_dropout=dropout,
            activation_dropout=dropout,
            activation='gelu',
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        self.blocks = nn.ModuleList([
            Block(**block_config) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)

        # Per-track scoring head: encoded embedding → scalar score
        self.scoring_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        stage1_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Score each track using pairwise-bias self-attention.

        Args:
            points: (B, 2, K1) coordinates in (η, φ).
            features: (B, input_dim, K1) per-track features.
            lorentz_vectors: (B, 4, K1) raw 4-vectors (px, py, pz, E).
            mask: (B, 1, K1) validity mask.
            stage1_scores: (B, K1) scores from Stage 1 pre-filter.

        Returns:
            scores: (B, K1) per-track scores. Padded tracks get -inf.
        """
        valid_mask = mask.squeeze(1).bool()  # (B, K1)
        padding_mask = ~valid_mask  # (B, K1) — True for padded positions
        mask_float = mask.float()

        # Concatenate stage1 scores as an extra feature channel.
        # Replace -inf (padded tracks from select_top_k) with 0 before
        # multiplication — (-inf * 0.0 = NaN) in float arithmetic.
        safe_stage1_scores = stage1_scores.masked_fill(
            ~valid_mask, 0.0,
        )
        stage1_channel = safe_stage1_scores.unsqueeze(1)  # (B, 1, K1)

        # Energy sharing fraction: z_i = pT_i / sum(pT) per event.
        # Tau daughters have characteristic z distributions from the
        # a1→rho+pi decay (correlated momentum sharing).
        px_track = lorentz_vectors[:, 0:1, :]  # (B, 1, K1)
        py_track = lorentz_vectors[:, 1:2, :]  # (B, 1, K1)
        pt_track = torch.sqrt(px_track ** 2 + py_track ** 2)  # (B, 1, K1)
        sum_pt = (pt_track * mask_float).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_pt = (pt_track / sum_pt) * mask_float  # (B, 1, K1)

        combined_features = torch.cat(
            [features, stage1_channel, z_pt], dim=1,
        ) * mask_float  # (B, input_dim+2, K1)

        # Input embedding: (B, C, K1) → (K1, B, embed_dim)
        track_embeddings = self.embed(combined_features)
        track_embeddings = track_embeddings.masked_fill(
            ~mask.bool().permute(2, 0, 1), 0,
        )

        # Pairwise attention bias from Lorentz vectors + physics features:
        # PairEmbed computes ln kT, ln z, ln ΔR, ln m² for all pairs
        # and projects to (B, num_heads, K1, K1) → reshape to (B*H, K1, K1)
        #
        # .detach() prevents NaN gradients from pairwise_lv_fts():
        # sqrt(ΔR²) has gradient 0.5/sqrt(0) = inf for self-pairs (ΔR=0).
        # Pairwise features are physics constants used as attention bias —
        # they don't need gradients w.r.t. input 4-vectors.
        # .float() ensures float32 precision for the ln/sqrt operations
        # even when AMP casts the rest to float16.
        lorentz_for_pairs = (lorentz_vectors * mask_float).detach().float()

        # Compute additional physics-motivated pairwise features
        extra_pairwise = self._compute_extra_pairwise_features(
            points, features, lorentz_for_pairs, mask_float,
        ) if self.pair_extra_dim > 0 else None
        # _compute_extra_pairwise_features always produces 6 channels.
        # pair_extra_dim must be 0 (disabled) or 6 (all physics features).
        if extra_pairwise is not None and extra_pairwise.shape[1] != self.pair_extra_dim:
            raise ValueError(
                f'pair_extra_dim={self.pair_extra_dim} but got '
                f'{extra_pairwise.shape[1]} pairwise channels. '
                f'Use pair_extra_dim=6 or 0.'
            )

        attention_bias = self.pair_embed(
            lorentz_for_pairs, uu=extra_pairwise,
        )  # (B, num_heads, K1, K1)
        num_heads = attention_bias.shape[1]
        attention_bias = attention_bias.view(
            -1, num_heads, attention_bias.shape[2], attention_bias.shape[3],
        ).reshape(-1, attention_bias.shape[2], attention_bias.shape[3])
        # (B*num_heads, K1, K1)

        # Transformer blocks with pairwise bias
        # Block expects: x=(K1, B, embed_dim), padding_mask=(B, K1),
        #                attn_mask=(B*H, K1, K1)
        encoded = track_embeddings
        for block in self.blocks:
            encoded = block(
                encoded,
                x_cls=None,
                padding_mask=padding_mask,
                attn_mask=attention_bias,
            )

        # Per-track scoring: (K1, B, embed_dim) → (B, K1)
        encoded = self.output_norm(encoded)  # (K1, B, embed_dim)
        encoded = encoded.permute(1, 0, 2)  # (B, K1, embed_dim)
        scores = self.scoring_head(encoded).squeeze(-1)  # (B, K1)

        # Mask padded tracks
        scores = scores.masked_fill(padding_mask, float('-inf'))

        return scores

    def _compute_extra_pairwise_features(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_for_pairs: torch.Tensor,
        mask_float: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics-motivated pairwise features for attention bias.

        These features exploit tau→3pi decay signatures that the generic
        Lorentz features (ln kT, ln z, ln ΔR, ln m²) don't capture:
        - Charge pairing (rho→π⁺π⁻ requires opposite sign)
        - Vertex compatibility (pions share the tau decay vertex)
        - Rho mass resonance (m_ij ~ 770 MeV for the rho daughter pair)

        Args:
            points: (B, 2, K1) — (eta, phi) coordinates, unstandardized.
            features: (B, input_dim, K1) — standardized per-track features.
            lorentz_for_pairs: (B, 4, K1) — detached float32 4-vectors.
            mask_float: (B, 1, K1) — float mask (1=valid, 0=padded).

        Returns:
            (B, pair_extra_dim, K1, K1) — pairwise feature tensor.
        """
        # Pair validity mask: zero out features involving padded tracks
        # pair_mask[b, 0, i, j] = 1 iff both tracks i and j are valid
        pair_mask = mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)

        # --- Channel 1: charge product q_i × q_j ---
        # Standardized charge has center=1.0, scale=0.5:
        #   raw +1 → standardized 0.0, raw -1 → standardized -1.0
        # Recover raw: charge_raw = standardized / 0.5 + 1.0
        charge_raw = (features[:, 5:6, :] / 0.5 + 1.0) * mask_float
        # q_i × q_j: (B, 1, K1, 1) × (B, 1, 1, K1) → (B, 1, K1, K1)
        charge_product = charge_raw.unsqueeze(-1) * charge_raw.unsqueeze(-2)

        # --- Channel 2: dz compatibility |dz_sig_i − dz_sig_j| ---
        # All 3 tau pions share the decay vertex → similar dz_significance.
        # Feature index 7 = track_log_dz_significance.
        dz_sig = features[:, 7:8, :] * mask_float
        dz_diff = (dz_sig.unsqueeze(-1) - dz_sig.unsqueeze(-2)).abs()

        # --- Channel 3: rho mass indicator exp(-(m_ij − 770)² / 2σ²) ---
        # The ρ(770) → π⁺π⁻ decay produces a mass peak at 770 MeV.
        # Gaussian with σ = 75 MeV highlights pairs near the resonance.
        # m_ij = sqrt((E_i + E_j)² − |p_i + p_j|²)
        lv = lorentz_for_pairs  # (B, 4, K1)
        px, py, pz, energy = lv[:, 0:1], lv[:, 1:2], lv[:, 2:3], lv[:, 3:4]
        sum_energy = energy.unsqueeze(-1) + energy.unsqueeze(-2)
        sum_px = px.unsqueeze(-1) + px.unsqueeze(-2)
        sum_py = py.unsqueeze(-1) + py.unsqueeze(-2)
        sum_pz = pz.unsqueeze(-1) + pz.unsqueeze(-2)
        m_squared = (
            sum_energy ** 2 - sum_px ** 2 - sum_py ** 2 - sum_pz ** 2
        ).clamp(min=1e-10)
        m_ij = m_squared.sqrt()  # (B, 1, K1, K1)
        rho_indicator = torch.exp(
            -0.5 * ((m_ij - 0.770) / 0.075) ** 2,
        )

        # --- Channel 4: rho OS indicator (OS × rho_indicator) ---
        # Conjunction of charge and mass: targets the specific ρ→π⁺π⁻ pair.
        # |d'| = 0.57, stronger than either component alone (0.33, 0.53).
        is_opposite_sign = (charge_product < 0).float()
        rho_os_indicator = is_opposite_sign * rho_indicator

        # --- Channel 5: phi-corrected dxy compatibility ---
        # Raw |dxy_i − dxy_j| has |d'| = 0.11 (poor) because dxy depends on
        # track φ relative to the flight direction. The correction divides by
        # |2 sin(Δφ/2)| which removes the φ-dependence for same-vertex tracks:
        #   d0_i − d0_j ≈ x_v(sinφ_j − sinφ_i) + y_v(cosφ_i − cosφ_j)
        # After correction: |d'| = 0.39.
        dxy_sig = features[:, 6:7, :] * mask_float
        dxy_diff = (dxy_sig.unsqueeze(-1) - dxy_sig.unsqueeze(-2)).abs()
        phi_raw = points[:, 1:2, :]  # unstandardized φ
        delta_phi = phi_raw.unsqueeze(-1) - phi_raw.unsqueeze(-2)
        delta_phi = (delta_phi + torch.pi) % (2 * torch.pi) - torch.pi
        sin_half_dphi = torch.abs(torch.sin(delta_phi / 2.0))
        # Clamp to avoid division by zero for nearly parallel tracks
        dxy_phi_corrected = dxy_diff / (2.0 * sin_half_dphi).clamp(min=0.05)

        # --- Channel 6: Minkowski dot product p_i · p_j ---
        # PELICAN (arXiv) shows pairwise Lorentz dot products alone match ParT.
        # p_i · p_j = E_i*E_j - px_i*px_j - py_i*py_j - pz_i*pz_j
        lorentz_dot = (
            energy.unsqueeze(-1) * energy.unsqueeze(-2)
            - px.unsqueeze(-1) * px.unsqueeze(-2)
            - py.unsqueeze(-1) * py.unsqueeze(-2)
            - pz.unsqueeze(-1) * pz.unsqueeze(-2)
        )  # (B, 1, K1, K1)

        # Stack and mask: (B, 6, K1, K1)
        extra_pairwise = torch.cat([
            charge_product,
            dz_diff,
            rho_indicator,
            rho_os_indicator,
            dxy_phi_corrected,
            lorentz_dot,
        ], dim=1) * pair_mask

        return extra_pairwise

    def set_training_progress(self, progress: float) -> None:
        """Set training progress for hybrid_lambda loss annealing and
        contrastive-denoising sigma schedule.

        Args:
            progress: float in [0, 1], where 0 = start, 1 = end of training.
        """
        self._training_progress = max(0.0, min(1.0, progress))

    @property
    def current_denoising_sigma(self) -> float:
        """Linearly interpolated sigma for contrastive denoising.

        At progress=0 returns ``denoising_sigma_start``; at progress=1 returns
        ``denoising_sigma_end``. Matches the TrackPreFilter scheduling pattern.
        """
        return (
            self.denoising_sigma_start
            + self._training_progress
            * (self.denoising_sigma_end - self.denoising_sigma_start)
        )

    def compute_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
        use_contrastive_denoising: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute ranking loss with togglable mode.

        Modes:
            'pairwise': L = T * softplus((s_neg - s_pos) / T)
                Standard pairwise ranking loss with random negative sampling.
            'lambda_rank': Same pairwise loss, but weighted by |DR@K|.
                Only pairs straddling the rank-K boundary get nonzero weight.
            'rs_at_k': RS@K = (1/|P|) sum_p sigma_t1(K - sum_n sigma_t2(s_n - s_p))
                Differentiable R@K surrogate (Patel et al., CVPR 2022).
            'hybrid_lambda': (1-alpha)*pairwise + alpha*lambda_rank
                Anneals alpha from 0 to 1 after warmup_fraction of training.

        If ``use_contrastive_denoising`` is True and the model is in training
        mode, an auxiliary denoising loss is added: noised GT track features
        must still score above the original-pass background scores. See
        ``_contrastive_denoising_loss`` for details.

        Returns:
            dict with 'total_loss', 'ranking_loss', '_scores', and (when
            denoising is active) 'denoising_loss'.
        """
        scores = self.forward(
            points, features, lorentz_vectors, mask, stage1_scores,
        )
        valid_mask = mask.squeeze(1).bool()
        labels = (
            track_labels.squeeze(1)[:, :scores.shape[1]] * valid_mask.float()
        )

        if self.loss_mode == 'rs_at_k':
            loss_dict = self._rs_at_k_loss(scores, labels, valid_mask)
        elif self.loss_mode == 'hybrid_lambda':
            loss_dict = self._hybrid_lambda_loss(scores, labels, valid_mask)
        else:
            loss_dict = self._pairwise_ranking_loss(scores, labels, valid_mask)

        # Contrastive denoising: training-only auxiliary regularizer.
        # Mirrors TrackPreFilter's pattern — adds a second forward pass on
        # feature-noised GT copies and requires them to still beat bg tracks.
        #
        # The three guards are:
        #   - self.use_contrastive_denoising: feature configured at init
        #   - use_contrastive_denoising kwarg: per-call override, lets the
        #     validate() loop disable denoising even though the model is in
        #     train() mode (necessary because train_cascade.py's validate()
        #     flips to train() mode for BatchNorm batch stats)
        #   - self.training: standard PyTorch training mode guard (safety net
        #     for pure-inference callers that forget the kwarg)
        if (
            self.use_contrastive_denoising
            and use_contrastive_denoising
            and self.training
        ):
            denoising_loss = self._contrastive_denoising_loss(
                points=points,
                features=features,
                lorentz_vectors=lorentz_vectors,
                mask=mask,
                track_labels=track_labels,
                stage1_scores=stage1_scores,
                original_scores=scores,
            )
            loss_dict['denoising_loss'] = denoising_loss
            loss_dict['total_loss'] = (
                loss_dict['total_loss']
                + self.denoising_loss_weight * denoising_loss
            )

        loss_dict['_scores'] = scores
        return loss_dict

    def _contrastive_denoising_loss(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        lorentz_vectors: torch.Tensor,
        mask: torch.Tensor,
        track_labels: torch.Tensor,
        stage1_scores: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> torch.Tensor:
        """DINO-style contrastive denoising loss on GT track features.

        Adds Gaussian noise with σ = ``current_denoising_sigma`` to the
        *features* of GT tracks only (background features unchanged), then
        runs a second forward pass through the full transformer. The noised
        GT scores must still beat the (original-pass) background scores:

            L_den = T · softplus( (s_bg_original − s_gt_noised) / T )

        averaged per event over sampled background pairs. This is the same
        loss structure as the main pairwise ranking loss, but applied to
        perturbed positives — teaching the model to be invariant to small
        feature jitter near the GT manifold.

        Only ``features`` are noised — not ``lorentz_vectors`` (which drive
        the physics pairwise attention bias) or ``stage1_scores`` (which
        come from the frozen pre-filter on the original features).

        Args:
            points: (B, 2, K1) coordinates.
            features: (B, input_dim, K1) original per-track features.
            lorentz_vectors: (B, 4, K1) raw 4-vectors (unchanged by denoising).
            mask: (B, 1, K1) validity mask.
            track_labels: (B, 1, K1) binary GT labels.
            stage1_scores: (B, K1) scores from the frozen Stage 1.
            original_scores: (B, K1) scores from the original (non-noised)
                forward pass. Used for the background comparison terms.

        Returns:
            Scalar denoising loss (0.0 if no event in the batch has GT tracks).
        """
        valid_mask = mask.squeeze(1).bool()  # (B, K1)
        labels_flat = (
            track_labels.squeeze(1)[:, :valid_mask.shape[1]]
            * valid_mask.float()
        )  # (B, K1)
        gt_mask = (labels_flat == 1.0) & valid_mask  # (B, K1)

        if not gt_mask.any():
            return torch.tensor(
                0.0, device=features.device, dtype=features.dtype,
            )

        # Build noised features: noise applied only at GT positions.
        # gt_mask broadcasts over the feature channel dimension.
        gt_mask_expanded = gt_mask.unsqueeze(1)  # (B, 1, K1)
        positive_noise = (
            torch.randn_like(features) * self.current_denoising_sigma
        )
        positive_noised_features = torch.where(
            gt_mask_expanded, features + positive_noise, features,
        )

        # Second forward pass with noised features. lorentz_vectors and
        # stage1_scores are NOT noised — they remain the physics ground truth
        # so the attention bias and the frozen Stage 1 output stay consistent.
        positive_noised_scores = self.forward(
            points, positive_noised_features, lorentz_vectors, mask, stage1_scores,
        )  # (B, K1)

        batch_size = features.shape[0]
        temperature = self.ranking_temperature
        event_losses: list[torch.Tensor] = []

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

            num_samples = min(
                self.ranking_num_samples, len(negative_indices),
            )
            sample_idx = torch.randint(
                0, len(negative_indices), (num_samples,),
                device=features.device,
            )
            # Background scores come from the ORIGINAL forward pass
            # (not the noised one) — matches TrackPreFilter's pattern and
            # lets the main ranking loss + denoising loss share the same
            # background statistics.
            bg_scores = original_scores[
                event_index, negative_indices[sample_idx],
            ]

            # Same pairwise softplus as _pairwise_ranking_loss, but with
            # noised positives vs original-pass backgrounds.
            scaled_margin = (
                bg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
            ) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)
            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            return torch.tensor(
                0.0, device=features.device, dtype=features.dtype,
            )
        return torch.stack(event_losses).mean()

    def _pairwise_ranking_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pairwise or LambdaRank loss depending on self.loss_mode.

        LambdaRank weights each (pos, neg) pair by whether swapping them
        would change R@K. Only pairs where one track is in the current
        top-K and the other isn't receive nonzero gradient.
        """
        batch_size = scores.shape[0]
        temperature = self.ranking_temperature
        k_boundary = self.rs_at_k_target
        event_losses = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = labels[event_index]
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

            # L = T * softplus((s_neg - s_pos) / T)
            scaled_margin = (negative_scores - positive_scores) / temperature
            pairwise_loss = temperature * functional.softplus(scaled_margin)

            if self.loss_mode == 'lambda_rank':
                with torch.no_grad():
                    masked_scores = event_scores.clone()
                    masked_scores[~event_valid] = float('-inf')
                    ranks = torch.argsort(
                        torch.argsort(masked_scores, descending=True),
                    )
                    pos_ranks = ranks[positive_indices]
                    neg_ranks = ranks[sampled_negatives]
                    pos_in_topk = (pos_ranks < k_boundary).float().unsqueeze(1)
                    neg_in_topk = (neg_ranks < k_boundary).float().unsqueeze(0)
                    n_gt = max(1, len(positive_indices))
                    lambda_weights = (
                        (pos_in_topk * (1.0 - neg_in_topk))
                        + ((1.0 - pos_in_topk) * neg_in_topk)
                    ) / n_gt

                pairwise_loss = pairwise_loss * lambda_weights

            event_losses.append(pairwise_loss.mean())

        if not event_losses:
            ranking_loss = torch.tensor(
                0.0, device=scores.device, dtype=scores.dtype,
            )
        else:
            ranking_loss = torch.stack(event_losses).mean()

        return {
            'total_loss': ranking_loss,
            'ranking_loss': ranking_loss,
        }

    def _rs_at_k_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Differentiable R@K surrogate loss (Patel et al., CVPR 2022).

        RS@K = (1/|P|) * sum_p sigma_t1(K - sum_n sigma_t2(s_n - s_p))

        The loss is 1 - RS@K (minimize to maximize recall).
        """
        batch_size = scores.shape[0]
        k_target = self.rs_at_k_target
        tau1 = self.rs_at_k_tau1
        tau2 = self.rs_at_k_tau2
        event_losses = []

        for event_index in range(batch_size):
            event_scores = scores[event_index]
            event_labels = labels[event_index]
            event_valid = valid_mask[event_index]

            positive_indices = (
                (event_labels == 1.0) & event_valid
            ).nonzero(as_tuple=True)[0]
            negative_indices = (
                (event_labels == 0.0) & event_valid
            ).nonzero(as_tuple=True)[0]

            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue

            pos_scores = event_scores[positive_indices]
            neg_scores = event_scores[negative_indices]

            # score_diffs[p, n] = s_neg[n] - s_pos[p]
            score_diffs = neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
            soft_rank_contributions = torch.sigmoid(score_diffs / tau2)
            soft_ranks = soft_rank_contributions.sum(dim=1)
            in_top_k = torch.sigmoid((k_target - soft_ranks) / tau1)
            rs_at_k = in_top_k.mean()
            event_losses.append(1.0 - rs_at_k)

        if not event_losses:
            rs_loss = torch.tensor(
                0.0, device=scores.device, dtype=scores.dtype,
            )
        else:
            rs_loss = torch.stack(event_losses).mean()

        return {
            'total_loss': rs_loss,
            'ranking_loss': rs_loss,
            'rs_at_k_loss': rs_loss,
        }

    def _hybrid_lambda_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Hybrid loss: pairwise early, LambdaRank late.

        L = (1 - alpha) * L_pairwise + alpha * L_lambda_rank

        Alpha ramps from 0 to 1 between warmup_start and warmup_end.
        Default: epoch 40 → epoch 70 (at 100 total epochs).
        """
        start = self.lambda_rank_warmup_start
        end = self.lambda_rank_warmup_end
        progress = self._training_progress

        if progress <= start:
            alpha = 0.0
        elif progress >= end:
            alpha = 3.0
        else:
            alpha = 3.0 * (progress - start) / (end - start)

        original_mode = self.loss_mode

        self.loss_mode = 'pairwise'
        pairwise_dict = self._pairwise_ranking_loss(scores, labels, valid_mask)

        self.loss_mode = 'lambda_rank'
        lambda_dict = self._pairwise_ranking_loss(scores, labels, valid_mask)

        self.loss_mode = original_mode

        pairwise_loss = pairwise_dict['ranking_loss']
        lambda_loss = lambda_dict['ranking_loss']
        # Pairwise reduces from 1.0 to 0.5; lambda scales from 0 to 3.
        # At alpha=0: 1.0*pairwise + 0*lambda (pure pairwise)
        # At alpha=3: 0.5*pairwise + 3*lambda (lambda dominates, pairwise anchors)
        pairwise_weight = 1.0 - alpha / 6.0  # 1.0 → 0.5 as alpha goes 0 → 3
        combined = pairwise_weight * pairwise_loss + alpha * lambda_loss

        return {
            'total_loss': combined,
            'ranking_loss': combined,
            'pairwise_loss': pairwise_loss,
            'lambda_rank_loss': lambda_loss,
            'lambda_alpha': torch.tensor(alpha, device=scores.device),
        }
