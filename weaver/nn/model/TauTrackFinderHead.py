"""DETR-style encoder-decoder head for tau-origin track finding.

Architecture follows DETR (Carion et al., ECCV 2020) with a dual
cross-attention extension in the decoder:

    1. Compact Token Encoder: self-attention on 128 compact tokens from the
       backbone, capturing global event context.
    2. Query Decoder: queries cross-attend to BOTH encoded compact tokens
       (global event context) AND enriched per-track features (fine-grained
       track identity) via DualCrossAttentionDecoderLayer.
    3. Mask Head: dot-product scoring between decoded queries and per-track
       enriched features, with a learned temperature parameter.
    4. Confidence Head: per-query binary prediction (exists / empty).
    5. Auxiliary losses at every decoder layer for deep supervision.

Queries are initialized via Farthest Point Sampling (FPS) from enriched
track features rather than learned embeddings, providing spatially diverse
starting points in (eta, phi) space.

The backbone is external -- this module only receives backbone outputs
(enriched features and compact tokens) and produces mask logits and
confidence logits. Loss computation is handled by TauTrackFinder.

References:
    Carion, N. et al. "End-to-End Object Detection with Transformers."
    ECCV 2020. https://arxiv.org/abs/2005.12872

    Cheng, B. et al. "Masked-attention Mask Transformer for Universal
    Image Segmentation." CVPR 2022. (Mask2Former -- dual cross-attention)
"""
import torch
import torch.nn as nn

from weaver.nn.model.HierarchicalGraphBackbone import farthest_point_sampling


class DualCrossAttentionDecoderLayer(nn.Module):
    """Decoder layer with dual cross-attention to compact tokens and tracks.

    Each layer has 4 sublayers (all post-norm, following original DETR):
        1. Self-attention among queries (coordination to avoid duplicates)
        2. Cross-attention: queries -> encoded compact tokens (global context)
        3. Cross-attention: queries -> enriched per-track features (track identity)
        4. Feed-forward network (two linear layers with GELU)

    Post-norm (norm_first=False) applies LayerNorm AFTER each residual addition,
    bounding representation norms across layers. Pre-norm was found to cause
    query norms to explode from ~1 to ~598 across 6 layers, making
    cross-attention contributions negligible (<0.2% of residual norm).

    Args:
        decoder_dim: Dimension of query, key, and value vectors.
        num_heads: Number of attention heads in each MultiheadAttention.
        dim_feedforward: Hidden dimension of the feed-forward network.
        dropout: Dropout rate applied in attention and FFN.
    """

    def __init__(
        self,
        decoder_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        # Sublayer 1: Self-attention among queries for coordination
        # Prevents multiple queries from converging to the same track.
        self.self_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Sublayer 2: Cross-attention to encoded compact tokens
        # Queries attend to 128 compact spatial tokens for global event context.
        self.compact_cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Sublayer 3: Cross-attention to enriched per-track features
        # Queries attend to ~1130 enriched track features for fine-grained
        # track identity information. Uses track_padding_mask to ignore
        # padded track positions.
        self.track_cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Sublayer 4: Feed-forward network
        # FFN(x) = Linear_2(GELU(Linear_1(x)))
        # Expands to dim_feedforward then projects back to decoder_dim.
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, decoder_dim),
        )

        # Post-norm: LayerNorm applied AFTER residual addition at each sublayer
        # output = LayerNorm(sublayer(x) + x)
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        self.norm3 = nn.LayerNorm(decoder_dim)
        self.norm4 = nn.LayerNorm(decoder_dim)

        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        compact_memory: torch.Tensor,
        track_memory: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        track_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all 4 sublayers.

        Args:
            queries: (B, num_queries, decoder_dim) query embeddings.
            compact_memory: (B, 128, decoder_dim) encoded compact tokens.
            track_memory: (B, P, decoder_dim) enriched per-track features
                projected to decoder space.
            self_attention_mask: (total_queries, total_queries) or None.
                Boolean attention mask for self-attention. True = ignore.
                Used for denoising training to prevent information leaking
                between learnable and denoising query groups.
            track_padding_mask: (B, P) or None. Boolean key padding mask
                for track cross-attention. True = padded (ignore).

        Returns:
            queries: (B, num_queries, decoder_dim) updated query embeddings.
        """
        # Sublayer 1: Self-attention among queries
        # output = LayerNorm(dropout(SelfAttn(Q, Q, Q)) + Q)
        self_attention_output, _ = self.self_attention(
            query=queries,
            key=queries,
            value=queries,
            attn_mask=self_attention_mask,
        )
        queries = self.norm1(queries + self.dropout1(self_attention_output))

        # Sublayer 2: Cross-attention to compact tokens (global context)
        # output = LayerNorm(dropout(CrossAttn(Q, K_compact, V_compact)) + Q)
        compact_cross_output, _ = self.compact_cross_attention(
            query=queries,
            key=compact_memory,
            value=compact_memory,
        )
        queries = self.norm2(queries + self.dropout2(compact_cross_output))

        # Sublayer 3: Cross-attention to enriched tracks (fine-grained identity)
        # output = LayerNorm(dropout(CrossAttn(Q, K_track, V_track)) + Q)
        # track_padding_mask: (B, P) where True = padded positions to ignore
        track_cross_output, _ = self.track_cross_attention(
            query=queries,
            key=track_memory,
            value=track_memory,
            key_padding_mask=track_padding_mask,
        )
        queries = self.norm3(queries + self.dropout3(track_cross_output))

        # Sublayer 4: Feed-forward network
        # output = LayerNorm(dropout(FFN(Q)) + Q)
        ffn_output = self.ffn(queries)
        queries = self.norm4(queries + self.dropout4(ffn_output))

        return queries


class TauTrackFinderHead(nn.Module):
    """DETR-style encoder-decoder head with dual cross-attention for track finding.

    Takes enriched per-track features and compact spatial tokens from
    the backbone, processes them through a transformer encoder-decoder,
    and produces mask logits (per-query soft assignment over all tracks)
    and confidence logits (per-query exists/empty prediction).

    Key differences from standard DETR decoder:
        - Dual cross-attention: decoder attends to both compact tokens AND
          enriched per-track features (inspired by Mask2Former).
        - FPS query initialization: queries are seeded from spatially diverse
          track features via Farthest Point Sampling in (eta, phi), rather
          than learned embeddings. Provides input-dependent initialization.
        - Auxiliary losses: mask and confidence predictions at every decoder
          layer for deep supervision (following Mask2Former / DETR variants).
        - Support for denoising training: additional noised queries can be
          concatenated and processed alongside learnable queries.

    Args:
        backbone_dim: Channel dimension of backbone outputs (default: 256).
        decoder_dim: Internal dimension for encoder/decoder (default: 256).
        mask_dim: Dimension for dot-product mask scoring (default: 128).
        num_queries: Number of query embeddings (default: 30).
            Increased from 15 for ranking-based inference.
        num_heads: Number of attention heads in encoder/decoder (default: 8).
        num_encoder_layers: Number of self-attention layers for compact
            tokens (default: 6). Captures global event context.
        num_decoder_layers: Number of dual cross-attention decoder layers
            (default: 4). Reduced from 6 -- auxiliary losses at every layer
            compensate for fewer layers.
        dropout: Dropout rate in attention and feedforward layers (default: 0.1).
    """

    def __init__(
        self,
        backbone_dim: int = 256,
        decoder_dim: int = 256,
        mask_dim: int = 128,
        num_queries: int = 30,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.decoder_dim = decoder_dim
        self.mask_dim = mask_dim
        self.num_queries = num_queries

        # ---- Memory projection (compact tokens -> encoder input) ----
        # Projects backbone compact tokens from backbone_dim to decoder_dim.
        # LayerNorm stabilizes the scale for attention logits.
        self.memory_projection = nn.Linear(backbone_dim, decoder_dim)
        self.memory_norm = nn.LayerNorm(decoder_dim)

        # ---- Compact Token Encoder ----
        # Self-attention on 128 compact tokens to capture global event context
        # before queries interact with them.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,  # Post-norm (original DETR, Carion et al., ECCV 2020):
            # Pre-norm caused encoded compact tokens to have cross-event cosine
            # similarity ~0.97, meaning only ~3% of the representation is
            # event-specific. Post-norm bounds token norms after each residual
            # addition, preserving input-dependent variation across layers.
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ---- Track projection (enriched features -> decoder cross-attention) ----
        # Projects enriched per-track features to decoder_dim for the track
        # cross-attention sublayer in DualCrossAttentionDecoderLayer.
        self.track_projection = nn.Linear(backbone_dim, decoder_dim)
        self.track_norm = nn.LayerNorm(decoder_dim)

        # ---- Query projection (FPS-selected track features -> query space) ----
        # Projects enriched features at FPS-selected seed indices into the
        # decoder query space. Provides input-dependent query initialization.
        self.query_projection = nn.Linear(backbone_dim, decoder_dim)

        # ---- Dual Cross-Attention Decoder Layers ----
        # Each layer: self-attention -> compact cross-attention ->
        # track cross-attention -> FFN. All post-norm.
        self.decoder_layers = nn.ModuleList([
            DualCrossAttentionDecoderLayer(
                decoder_dim=decoder_dim,
                num_heads=num_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # ---- Track Key MLP ----
        # Projects enriched per-track features (B, backbone_dim, P) to mask
        # scoring key space (B, mask_dim, P) for dot-product scoring with
        # decoded queries.
        # Uses Conv1d (operates on channel dimension across tracks).
        # track_key = BN(GELU(Conv1d(backbone_dim -> backbone_dim))) -> Conv1d(backbone_dim -> mask_dim)
        self.track_key_mlp = nn.Sequential(
            nn.Conv1d(backbone_dim, backbone_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(backbone_dim),
            nn.GELU(),
            nn.Conv1d(backbone_dim, mask_dim, kernel_size=1),
        )

        # ---- Query Scoring MLP ----
        # Projects decoded queries to mask scoring query space for dot-product
        # scoring against track keys.
        # query_score = LayerNorm(GELU(Linear(decoder_dim -> decoder_dim))) -> Linear(decoder_dim -> mask_dim)
        self.query_scoring_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, mask_dim),
        )

        # ---- Learned Temperature ----
        # Controls sharpness of softmax distribution over ~1130 tracks.
        # score(query, key) = (query . key) / tau
        # Learnable tau lets the model tune the trade-off between:
        #   - Too flat (tau large): can't commit to a specific track
        #   - Too sharp (tau small): gradients vanish for non-selected tracks
        # Clamped to min=0.01 to prevent division by near-zero.
        self.temperature = nn.Parameter(torch.ones(1))

        # ---- Confidence Head ----
        # Per-query binary prediction: does this query point to a real tau
        # track (exists) or is it an unused slot (empty / no-object)?
        #
        # Input: [decoded_query, pointed_context] of dimension 2 * decoder_dim.
        #   - decoded_query (decoder_dim): the query's learned representation
        #     from self-attention + cross-attention in the decoder.
        #   - pointed_context (decoder_dim): soft attention readout of track
        #     features weighted by mask probabilities. This tells the confidence
        #     head WHAT the query is pointing at (tau-like vs background features).
        #     Computed with detached mask_logits so confidence loss does not
        #     interfere with the mask scoring path.
        #
        # confidence = Linear(128 -> 1, bias)(GELU(Linear(2*decoder_dim -> 128)))
        #
        # Bias initialization: with 30 queries and typically ~3 active (~10%),
        # init to logit(0.2) = ln(0.2 / 0.8) ~ -1.4 so the network starts
        # predicting low confidence for most queries.
        self.confidence_head = nn.Sequential(
            nn.Linear(2 * decoder_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # sigma(-1.4) ~ 0.198 ~ prior fraction of active queries
        nn.init.constant_(self.confidence_head[-1].bias, -1.4)

    def forward(
        self,
        enriched_features: torch.Tensor,
        compact_tokens: torch.Tensor,
        mask: torch.Tensor,
        points: torch.Tensor,
        num_denoising_queries: int = 0,
        denoising_query_features: torch.Tensor | None = None,
        denoising_attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass: encode compact tokens, decode queries, score tracks.

        Produces mask logits and confidence logits at every decoder layer
        for auxiliary loss supervision (deep supervision).

        Args:
            enriched_features: (B, backbone_dim, P) per-track enriched features
                from backbone.enrich(). Used for mask scoring and track
                cross-attention.
            compact_tokens: (B, backbone_dim, 128) compact spatial tokens from
                backbone.compact(). Used as encoder input / decoder memory.
            mask: (B, 1, P) boolean mask, True for valid tracks.
            points: (B, 2, P) coordinates in (eta, phi). Used for FPS query
                initialization to select spatially diverse seed tracks.
            num_denoising_queries: Number of denoising queries to append
                (training only). Default: 0 (no denoising).
            denoising_query_features: (B, num_denoising, backbone_dim) or None.
                Pre-constructed denoising query inputs. Must be provided when
                num_denoising_queries > 0.
            denoising_attention_mask: (total_queries, total_queries) or None.
                Boolean self-attention mask preventing information leaking
                between learnable and denoising query groups. True = ignore.
                Must be provided when num_denoising_queries > 0.

        Returns:
            Tuple of:
                all_layer_mask_logits: list of (B, total_queries, P) mask logits
                    at each decoder layer. Padded positions are -inf.
                    Length = num_decoder_layers.
                all_layer_confidence_logits: list of (B, total_queries)
                    confidence logits at each decoder layer (pre-sigmoid).
                    Length = num_decoder_layers.
        """
        batch_size = enriched_features.shape[0]

        # ------------------------------------------------------------------
        # Step 1: Project enriched features to mask scoring key space
        # ------------------------------------------------------------------
        # track_key_mlp: Conv1d on (B, backbone_dim, P) -> (B, mask_dim, P)
        # Transpose to (B, P, mask_dim) for dot-product with query projections.
        track_keys = self.track_key_mlp(enriched_features)  # (B, mask_dim, P)
        track_keys = track_keys.transpose(1, 2)  # (B, P, mask_dim)

        # ------------------------------------------------------------------
        # Step 2: Project enriched features to decoder cross-attention space
        # ------------------------------------------------------------------
        # (B, backbone_dim, P) -> transpose -> (B, P, backbone_dim) -> project
        # -> (B, P, decoder_dim)
        track_memory = self.track_norm(
            self.track_projection(enriched_features.transpose(1, 2))
        )  # (B, P, decoder_dim)

        # ------------------------------------------------------------------
        # Step 3: Project and encode compact tokens
        # ------------------------------------------------------------------
        # (B, backbone_dim, 128) -> transpose -> (B, 128, backbone_dim) -> project
        memory = self.memory_norm(
            self.memory_projection(compact_tokens.transpose(1, 2))
        )  # (B, 128, decoder_dim)

        # Self-attention encoder on compact tokens for global event context
        encoded_memory = self.transformer_encoder(memory)  # (B, 128, decoder_dim)

        # ------------------------------------------------------------------
        # Step 4: Initialize queries via FPS from enriched features
        # ------------------------------------------------------------------
        # Farthest Point Sampling in (eta, phi) space selects num_queries
        # spatially diverse seed track indices. The enriched features at
        # these indices are projected into the decoder query space.
        seed_indices = farthest_point_sampling(
            points, mask, self.num_queries,
        )  # (B, num_queries)

        # Gather enriched features at seed indices
        # seed_indices: (B, num_queries) -> (B, num_queries, 1) -> expand
        # enriched_features: (B, backbone_dim, P) -> transpose -> (B, P, backbone_dim)
        enriched_transposed = enriched_features.transpose(1, 2)  # (B, P, backbone_dim)
        seed_indices_expanded = seed_indices.unsqueeze(-1).expand(
            -1, -1, self.backbone_dim,
        )  # (B, num_queries, backbone_dim)
        seed_features = enriched_transposed.gather(
            1, seed_indices_expanded,
        )  # (B, num_queries, backbone_dim)

        # Project seed features into decoder query space
        learnable_queries = self.query_projection(
            seed_features,
        )  # (B, num_queries, decoder_dim)

        # ------------------------------------------------------------------
        # Step 5: Concatenate denoising queries (training only)
        # ------------------------------------------------------------------
        if num_denoising_queries > 0 and denoising_query_features is not None:
            # Project denoising query features into decoder space
            denoising_queries = self.query_projection(
                denoising_query_features,
            )  # (B, num_denoising, decoder_dim)

            # Concatenate: [learnable_queries, denoising_queries]
            all_queries = torch.cat(
                [learnable_queries, denoising_queries], dim=1,
            )  # (B, num_queries + num_denoising, decoder_dim)
        else:
            all_queries = learnable_queries  # (B, num_queries, decoder_dim)

        # ------------------------------------------------------------------
        # Step 6: Build track padding mask for cross-attention
        # ------------------------------------------------------------------
        # MultiheadAttention key_padding_mask: (B, S) where True = ignore
        # mask: (B, 1, P) with True = valid -> invert for padding mask
        track_padding_mask = ~mask.squeeze(1).bool()  # (B, P): True = padded

        # ------------------------------------------------------------------
        # Step 7: Build padding mask for pointer logits
        # ------------------------------------------------------------------
        # (B, 1, P) where True = padded, used to mask pointer logits with -inf
        pointer_padding_mask = ~mask.bool()  # (B, 1, P): True = padded

        # Clamp temperature once (shared across all layers)
        # tau clamped to min=0.01 to prevent division by near-zero
        clamped_temperature = self.temperature.clamp(min=0.01)

        # ------------------------------------------------------------------
        # Step 8: Run through decoder layers with auxiliary outputs
        # ------------------------------------------------------------------
        all_layer_mask_logits = []
        all_layer_confidence_logits = []

        queries = all_queries
        for decoder_layer in self.decoder_layers:
            # Forward through DualCrossAttentionDecoderLayer
            queries = decoder_layer(
                queries=queries,
                compact_memory=encoded_memory,
                track_memory=track_memory,
                self_attention_mask=denoising_attention_mask,
                track_padding_mask=track_padding_mask,
            )  # (B, total_queries, decoder_dim)

            # Compute mask logits via dot-product scoring
            # query_scores: (B, total_queries, mask_dim)
            query_scores = self.query_scoring_mlp(queries)

            # mask_logits = (query_scores @ track_keys^T) / tau
            # score(query, key) = (query . key) / tau
            # (B, total_queries, mask_dim) @ (B, mask_dim, P) -> (B, total_queries, P)
            mask_logits = torch.bmm(
                query_scores, track_keys.transpose(1, 2),
            ) / clamped_temperature  # (B, total_queries, P)

            # Mask out padded track positions with -inf
            # softmax(-inf) -> 0 probability for padded positions
            mask_logits = mask_logits.masked_fill(
                pointer_padding_mask, float('-inf'),
            )

            # Compute mask-informed confidence logits
            # Step 1: soft attention readout of track features using mask probs.
            # pointed_context = softmax(mask_logits) @ track_memory
            # This encodes the enriched features of the track(s) each query
            # is pointing at, giving the confidence head direct information
            # about the quality and identity of the selected track.
            #
            # detach mask_logits so confidence loss does NOT backpropagate
            # through the mask scoring path (query_scoring_mlp, track_key_mlp,
            # temperature). The confidence loss still flows through queries
            # and track_memory via the pointed_context computation.
            mask_probabilities = torch.softmax(
                mask_logits.detach(), dim=-1,
            )  # (B, total_queries, P): padded positions get 0 from softmax(-inf)
            pointed_context = torch.bmm(
                mask_probabilities, track_memory,
            )  # (B, total_queries, decoder_dim)

            # Step 2: concatenate decoded query + pointed context
            # confidence_input = [query_representation, what_it_points_at]
            confidence_input = torch.cat(
                [queries, pointed_context], dim=-1,
            )  # (B, total_queries, 2 * decoder_dim)

            # Step 3: confidence_head: Linear(2*decoder_dim -> 128) -> GELU -> Linear(128 -> 1)
            confidence_logits = self.confidence_head(
                confidence_input,
            ).squeeze(-1)  # (B, total_queries)

            all_layer_mask_logits.append(mask_logits)
            all_layer_confidence_logits.append(confidence_logits)

        return all_layer_mask_logits, all_layer_confidence_logits
