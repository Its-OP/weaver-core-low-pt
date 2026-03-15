"""DETR-style decoder head for tau-origin track finding.

Minimal architecture: queries cross-attend to enriched per-track features,
then score tracks via dot-product mask logits. No compact token encoder,
no denoising, no auxiliary losses — just the core decoder.

Architecture:
    1. Query Initialization: FPS in (eta, phi) → project to decoder space
    2. Decoder: N layers of [self-attention → track cross-attention → FFN]
    3. Mask Head: dot-product scoring between decoded queries and track keys,
       with a learned temperature parameter
    4. Confidence Head: per-query binary exists/empty prediction

The backbone is external — this module only receives enriched per-track
features and produces mask logits + confidence logits. Loss computation
is handled by TauTrackFinder.

References:
    Carion, N. et al. "End-to-End Object Detection with Transformers."
    ECCV 2020. https://arxiv.org/abs/2005.12872
"""
import torch
import torch.nn as nn

from weaver.nn.model.HierarchicalGraphBackbone import farthest_point_sampling


class DecoderLayer(nn.Module):
    """Standard transformer decoder layer with 3 sublayers (post-norm).

    Sublayers:
        1. Self-attention among queries (prevents duplicate predictions)
        2. Cross-attention: queries → enriched per-track features
        3. Feed-forward network

    All sublayers use post-norm: output = LayerNorm(sublayer(x) + x)

    Args:
        decoder_dim: Dimension of query, key, and value vectors.
        num_heads: Number of attention heads.
        dim_feedforward: Hidden dimension of the FFN.
        dropout: Dropout rate in attention and FFN.
    """

    def __init__(
        self,
        decoder_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        # Sublayer 1: Self-attention among queries
        self.self_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Sublayer 2: Cross-attention to enriched per-track features
        # Queries attend to ~1130 track features for fine-grained identity.
        # key_padding_mask ignores padded track positions.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Sublayer 3: Feed-forward network
        # FFN(x) = Linear_2(GELU(Linear_1(x)))
        self.feed_forward_network = nn.Sequential(
            nn.Linear(decoder_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, decoder_dim),
        )

        # Post-norm: LayerNorm AFTER residual addition
        self.norm_self_attention = nn.LayerNorm(decoder_dim)
        self.norm_cross_attention = nn.LayerNorm(decoder_dim)
        self.norm_feed_forward = nn.LayerNorm(decoder_dim)

        self.dropout_self_attention = nn.Dropout(dropout)
        self.dropout_cross_attention = nn.Dropout(dropout)
        self.dropout_feed_forward = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        track_memory: torch.Tensor,
        track_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through 3 sublayers.

        Args:
            queries: (B, num_queries, decoder_dim) query embeddings.
            track_memory: (B, P, decoder_dim) projected track features.
            track_padding_mask: (B, P) boolean, True = padded (ignore).

        Returns:
            queries: (B, num_queries, decoder_dim) updated queries.
        """
        # Sublayer 1: Self-attention
        # output = LayerNorm(dropout(SelfAttn(Q, Q, Q)) + Q)
        self_attention_output, _ = self.self_attention(
            query=queries, key=queries, value=queries,
        )
        queries = self.norm_self_attention(
            queries + self.dropout_self_attention(self_attention_output),
        )

        # Sublayer 2: Cross-attention to track features
        # output = LayerNorm(dropout(CrossAttn(Q, K_track, V_track)) + Q)
        cross_attention_output, _ = self.cross_attention(
            query=queries,
            key=track_memory,
            value=track_memory,
            key_padding_mask=track_padding_mask,
        )
        queries = self.norm_cross_attention(
            queries + self.dropout_cross_attention(cross_attention_output),
        )

        # Sublayer 3: FFN
        # output = LayerNorm(dropout(FFN(Q)) + Q)
        feed_forward_output = self.feed_forward_network(queries)
        queries = self.norm_feed_forward(
            queries + self.dropout_feed_forward(feed_forward_output),
        )

        return queries


class TauTrackFinderHead(nn.Module):
    """DETR decoder head for track finding.

    Takes enriched per-track features from a frozen backbone, runs them
    through a transformer decoder with cross-attention, and produces
    mask logits and confidence logits for the last decoder layer only.

    Args:
        backbone_dim: Channel dimension of backbone outputs (default: 256).
        decoder_dim: Internal decoder dimension (default: 256).
        mask_dim: Dimension for dot-product mask scoring (default: 128).
        num_queries: Number of FPS-initialized queries (default: 30).
        num_heads: Attention heads per layer (default: 8).
        num_decoder_layers: Number of decoder layers (default: 4).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        backbone_dim: int = 256,
        decoder_dim: int = 256,
        mask_dim: int = 128,
        num_queries: int = 30,
        num_heads: int = 8,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.decoder_dim = decoder_dim
        self.mask_dim = mask_dim
        self.num_queries = num_queries

        # ---- Track projection (enriched features → decoder space) ----
        self.track_projection = nn.Linear(backbone_dim, decoder_dim)
        self.track_norm = nn.LayerNorm(decoder_dim)

        # ---- Query projection (FPS seed features → decoder space) ----
        self.query_projection = nn.Linear(backbone_dim, decoder_dim)

        # ---- Decoder layers ----
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                decoder_dim=decoder_dim,
                num_heads=num_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])

        # ---- Track Key MLP ----
        # Projects enriched features to mask scoring key space
        # track_key = BN(GELU(Conv1d(backbone_dim → backbone_dim))) → Conv1d(→ mask_dim)
        self.track_key_mlp = nn.Sequential(
            nn.Conv1d(backbone_dim, backbone_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(backbone_dim),
            nn.GELU(),
            nn.Conv1d(backbone_dim, mask_dim, kernel_size=1),
        )

        # ---- Query Scoring MLP ----
        # Projects decoded queries to mask scoring space
        # query_score = LayerNorm(GELU(Linear(decoder_dim))) → Linear(→ mask_dim)
        self.query_scoring_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, mask_dim),
        )

        # ---- Learned Temperature ----
        # score(query, key) = (query · key) / τ
        # τ clamped to [0.1, 2.0] to prevent too-sharp or too-flat distributions.
        self.temperature = nn.Parameter(torch.ones(1))

        # ---- Confidence Head ----
        # Per-query binary exists/empty prediction from decoded query only.
        # confidence = Linear(128 → 1)(GELU(Linear(decoder_dim → 128)))
        # Bias init: σ(-1.4) ≈ 0.198, matching prior ~10% active query rate.
        self.confidence_head = nn.Sequential(
            nn.Linear(decoder_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.confidence_head[-1].bias, -1.4)

    def forward(
        self,
        enriched_features: torch.Tensor,
        mask: torch.Tensor,
        points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: initialize queries, decode, score tracks.

        Args:
            enriched_features: (B, backbone_dim, P) from frozen backbone.
            mask: (B, 1, P) boolean mask, True for valid tracks.
            points: (B, 2, P) coordinates in (eta, phi) for FPS.

        Returns:
            mask_logits: (B, num_queries, P) — padded positions are -inf.
            confidence_logits: (B, num_queries) — pre-sigmoid.
        """
        # ---- Step 1: Track key projection for mask scoring ----
        # (B, backbone_dim, P) → Conv1d → (B, mask_dim, P) → transpose → (B, P, mask_dim)
        track_keys = self.track_key_mlp(enriched_features).transpose(1, 2)

        # ---- Step 2: Track memory projection for cross-attention ----
        # (B, backbone_dim, P) → transpose → (B, P, backbone_dim) → Linear → (B, P, decoder_dim)
        track_memory = self.track_norm(
            self.track_projection(enriched_features.transpose(1, 2)),
        )

        # ---- Step 3: FPS query initialization ----
        # Select num_queries spatially diverse seed tracks in (eta, phi)
        seed_indices = farthest_point_sampling(points, mask, self.num_queries)

        # Gather enriched features at seed indices → project to decoder space
        enriched_transposed = enriched_features.transpose(1, 2)  # (B, P, backbone_dim)
        seed_indices_expanded = seed_indices.unsqueeze(-1).expand(
            -1, -1, self.backbone_dim,
        )
        seed_features = enriched_transposed.gather(1, seed_indices_expanded)
        queries = self.query_projection(seed_features)  # (B, num_queries, decoder_dim)

        # ---- Step 4: Padding masks ----
        track_padding_mask = ~mask.squeeze(1).bool()  # (B, P): True = padded
        pointer_padding_mask = ~mask.bool()  # (B, 1, P): True = padded

        # ---- Step 5: Decoder layers ----
        for decoder_layer in self.decoder_layers:
            queries = decoder_layer(
                queries=queries,
                track_memory=track_memory,
                track_padding_mask=track_padding_mask,
            )

        # ---- Step 6: Mask logits via dot-product scoring ----
        # mask_logits = (query_scores @ track_keys^T) / τ
        clamped_temperature = self.temperature.clamp(min=0.1, max=2.0)
        query_scores = self.query_scoring_mlp(queries)
        mask_logits = torch.bmm(
            query_scores, track_keys.transpose(1, 2),
        ) / clamped_temperature

        # Padded positions → -inf so softmax gives 0 probability
        mask_logits = mask_logits.masked_fill(pointer_padding_mask, float('-inf'))

        # ---- Step 7: Confidence logits ----
        confidence_logits = self.confidence_head(queries).squeeze(-1)

        return mask_logits, confidence_logits
