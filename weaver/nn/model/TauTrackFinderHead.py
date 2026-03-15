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
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from weaver.nn.model.HierarchicalGraphBackbone import farthest_point_sampling


class QKNormMultiheadAttention(nn.Module):
    """Multi-head attention with L2-normalized queries and keys.

    Standard attention computes:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_head)) @ V

    With QK-Norm, Q and K are L2-normalized per head AFTER projection,
    then scaled by a learnable per-head temperature (initialized to sqrt(d_head)):
        Q_norm = Q / ||Q||_2
        K_norm = K / ||K||_2
        Attention(Q, K, V) = softmax(Q_norm @ K_norm^T * temperature) @ V

    This bounds attention logits to [-1, 1] * temperature, preventing
    softmax collapse to one-hot when Q/K norms grow during training.

    References:
        - Used in Gemini 2.0, DeepSeek-V3, LLaMA 3.1
        - arXiv:2511.21377 "Controlling Changes to Attention Logits"

    Args:
        embed_dim: Total dimension of the model (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        batch_first: If True, input/output tensors are (batch, sequence, feature).
            Always True in this implementation (our convention).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Separate Q, K, V projections (not fused) for clarity.
        # Each projects from embed_dim to embed_dim.
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Learnable per-head temperature, initialized to sqrt(d_head).
        # After L2 normalization, Q_norm @ K_norm^T lies in [-1, 1].
        # Multiplying by temperature = sqrt(d_head) recovers the standard
        # attention logit scale at initialization:
        #   logits = (Q_norm @ K_norm^T) * temperature
        # Shape: (num_heads,), broadcast as (1, num_heads, 1, 1) during forward.
        self.head_temperature = nn.Parameter(
            torch.full((num_heads,), math.sqrt(self.head_dim))
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass with QK-normalized multi-head attention.

        Args:
            query: (B, S_q, embed_dim) query tensor.
            key: (B, S_kv, embed_dim) key tensor.
            value: (B, S_kv, embed_dim) value tensor.
            key_padding_mask: (B, S_kv) boolean mask where True = padded (ignore).
                Expanded to (B, 1, 1, S_kv) for broadcasting with attention logits.
            attn_mask: (S_q, S_kv) or (B*num_heads, S_q, S_kv) boolean mask
                where True = ignore. Converted to additive float mask (-inf)
                for F.scaled_dot_product_attention.

        Returns:
            Tuple of (attention_output, None):
                attention_output: (B, S_q, embed_dim) result tensor.
                None: placeholder for attention weights (not computed).
        """
        batch_size, query_sequence_length, _ = query.shape
        _, key_sequence_length, _ = key.shape

        # Step 1: Project Q, K, V through separate linear layers
        projected_query = self.query_projection(query)    # (B, S_q, embed_dim)
        projected_key = self.key_projection(key)          # (B, S_kv, embed_dim)
        projected_value = self.value_projection(value)    # (B, S_kv, embed_dim)

        # Step 2: Reshape to multi-head format (B, num_heads, S, head_dim)
        projected_query = projected_query.view(
            batch_size, query_sequence_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)  # (B, num_heads, S_q, head_dim)

        projected_key = projected_key.view(
            batch_size, key_sequence_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)  # (B, num_heads, S_kv, head_dim)

        projected_value = projected_value.view(
            batch_size, key_sequence_length, self.num_heads, self.head_dim,
        ).transpose(1, 2)  # (B, num_heads, S_kv, head_dim)

        # Step 3: L2-normalize Q and K along the head dimension (last dim)
        # Q_norm = Q / ||Q||_2, K_norm = K / ||K||_2
        # This bounds the dot product Q_norm @ K_norm^T to [-1, 1] per element
        normalized_query = functional.normalize(
            projected_query, dim=-1,
        )  # (B, num_heads, S_q, head_dim)
        normalized_key = functional.normalize(
            projected_key, dim=-1,
        )  # (B, num_heads, S_kv, head_dim)

        # Step 4: Scale Q by learnable per-head temperature
        # logits = (Q_norm * temperature) @ K_norm^T
        #        = (Q_norm @ K_norm^T) * temperature
        # Broadcast temperature (num_heads,) -> (1, num_heads, 1, 1)
        temperature_broadcast = self.head_temperature.view(
            1, self.num_heads, 1, 1,
        )
        scaled_query = normalized_query * temperature_broadcast

        # Step 5: Build combined attention mask for F.scaled_dot_product_attention
        # SDPA expects additive float mask where -inf = ignore, 0.0 = attend
        combined_attention_mask = None

        if attn_mask is not None:
            # attn_mask: boolean (True=ignore) -> additive float (-inf=ignore)
            if attn_mask.dim() == 2:
                # (S_q, S_kv) -> (1, 1, S_q, S_kv) for broadcasting
                combined_attention_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (B*num_heads, S_q, S_kv) -> (B, num_heads, S_q, S_kv)
                combined_attention_mask = attn_mask.view(
                    batch_size, self.num_heads,
                    query_sequence_length, key_sequence_length,
                )
            # Convert boolean mask to additive float mask
            combined_attention_mask = torch.zeros_like(
                combined_attention_mask, dtype=scaled_query.dtype,
            ).masked_fill(combined_attention_mask, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: (B, S_kv) boolean, True = padded (ignore)
            # Expand to (B, 1, 1, S_kv) for broadcasting across heads and queries
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Convert to additive float mask
            padding_mask = torch.zeros_like(
                padding_mask, dtype=scaled_query.dtype,
            ).masked_fill(padding_mask, float('-inf'))

            if combined_attention_mask is not None:
                # Combine both masks: either one saying -inf means ignore
                combined_attention_mask = combined_attention_mask + padding_mask
            else:
                combined_attention_mask = padding_mask

        # Step 6: Compute attention via F.scaled_dot_product_attention
        # We set scale=1.0 because we already applied our own temperature scaling.
        # SDPA computes: softmax(scaled_query @ K^T * scale + attn_mask) @ V
        # With scale=1.0: softmax(scaled_query @ K_norm^T + mask) @ V
        attention_output = functional.scaled_dot_product_attention(
            query=scaled_query,
            key=normalized_key,
            value=projected_value,
            attn_mask=combined_attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,  # We handle scaling ourselves via head_temperature
        )  # (B, num_heads, S_q, head_dim)

        # Step 7: Reshape back to (B, S_q, embed_dim) and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_sequence_length, self.embed_dim,
        )  # (B, S_q, embed_dim)
        attention_output = self.output_projection(attention_output)

        # Return (output, None) to match nn.MultiheadAttention interface.
        # We don't compute attention weights (not needed for training).
        return attention_output, None


class DropPath(nn.Module):
    """Stochastic depth: randomly skip a layer's contribution during training.

    Implements the residual-scaling variant of stochastic depth from
    Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.

    During training, the layer's residual delta (output - input) is either
    passed through with probability (1 - drop_probability) and scaled by
    1 / (1 - drop_probability) to preserve expected values, or zeroed out
    with probability drop_probability (effectively skipping the layer).

    The scaling ensures that:
        E[DropPath(delta)] = delta
    so the expected output is the same at train and eval time.

    During evaluation, the delta is always passed through unchanged
    (no dropout, no scaling).

    Args:
        drop_probability: Probability of dropping this layer's contribution.
            Must be in [0, 1). Default: 0.0 (no dropout).
    """

    def __init__(self, drop_probability: float = 0.0):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, layer_delta: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth to the layer's residual delta.

        Args:
            layer_delta: (B, num_queries, decoder_dim) difference between
                decoder layer output and its input (the layer's contribution).

        Returns:
            Scaled or zeroed delta with same shape as input.
        """
        if not self.training or self.drop_probability == 0.0:
            return layer_delta

        keep_probability = 1.0 - self.drop_probability
        # Binary mask: shape (batch_size, 1, 1) broadcasts across
        # sequence length and feature dimensions.
        # Each sample in the batch independently decides to keep or drop.
        random_mask = torch.rand(
            layer_delta.shape[0], 1, 1,
            device=layer_delta.device,
            dtype=layer_delta.dtype,
        )
        # Bernoulli mask: 1 if rand < keep_prob, else 0
        random_mask = (random_mask < keep_probability).float()

        # Scale surviving outputs by 1/keep_probability so that:
        #   E[output] = keep_prob * (delta / keep_prob) + drop_prob * 0 = delta
        return layer_delta * random_mask / keep_probability

    def extra_repr(self) -> str:
        return f'drop_probability={self.drop_probability:.3f}'


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
        layer_scale_init: Initial value for LayerScale gamma parameters.
            Set to 0 or negative to disable LayerScale (default: 1e-4).
    """

    def __init__(
        self,
        decoder_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()

        # Sublayer 1: Self-attention among queries for coordination.
        # Prevents multiple queries from converging to the same track.
        # Uses QK-Norm to bound attention logits and prevent softmax collapse.
        self.self_attention = QKNormMultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Sublayer 2: Cross-attention to encoded compact tokens.
        # Queries attend to 128 compact spatial tokens for global event context.
        # Uses QK-Norm to bound attention logits and prevent softmax collapse.
        self.compact_cross_attention = QKNormMultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Sublayer 3: Cross-attention to enriched per-track features.
        # Queries attend to ~1130 enriched track features for fine-grained
        # track identity information. Uses track_padding_mask to ignore
        # padded track positions.
        # Uses QK-Norm to bound attention logits and prevent softmax collapse.
        self.track_cross_attention = QKNormMultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=num_heads,
            dropout=dropout,
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

        # LayerScale (CaiT, Touvron et al., ICCV 2021):
        # Learnable per-channel scaling gamma on each sublayer output,
        # initialized to a small value (default 1e-4).
        # output = LayerNorm(x + dropout(gamma * sublayer(x)))
        # Prevents early layers from dominating the residual stream in
        # deep networks by starting with near-identity residual connections.
        if layer_scale_init > 0:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init * torch.ones(decoder_dim),
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init * torch.ones(decoder_dim),
            )
            self.layer_scale_3 = nn.Parameter(
                layer_scale_init * torch.ones(decoder_dim),
            )
            self.layer_scale_4 = nn.Parameter(
                layer_scale_init * torch.ones(decoder_dim),
            )
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None
            self.layer_scale_3 = None
            self.layer_scale_4 = None

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
        # With LayerScale: output = LayerNorm(x + dropout(gamma_1 * SelfAttn(Q, Q, Q)))
        # Without LayerScale: output = LayerNorm(x + dropout(SelfAttn(Q, Q, Q)))
        self_attention_output, _ = self.self_attention(
            query=queries,
            key=queries,
            value=queries,
            attn_mask=self_attention_mask,
        )
        if self.layer_scale_1 is not None:
            self_attention_output = self.layer_scale_1 * self_attention_output
        queries = self.norm1(queries + self.dropout1(self_attention_output))

        # Sublayer 2: Cross-attention to compact tokens (global context)
        # With LayerScale: output = LayerNorm(x + dropout(gamma_2 * CrossAttn(Q, K, V)))
        # Without LayerScale: output = LayerNorm(x + dropout(CrossAttn(Q, K, V)))
        compact_cross_output, _ = self.compact_cross_attention(
            query=queries,
            key=compact_memory,
            value=compact_memory,
        )
        if self.layer_scale_2 is not None:
            compact_cross_output = self.layer_scale_2 * compact_cross_output
        queries = self.norm2(queries + self.dropout2(compact_cross_output))

        # Sublayer 3: Cross-attention to enriched tracks (fine-grained identity)
        # With LayerScale: output = LayerNorm(x + dropout(gamma_3 * CrossAttn(Q, K, V)))
        # Without LayerScale: output = LayerNorm(x + dropout(CrossAttn(Q, K, V)))
        # track_padding_mask: (B, P) where True = padded positions to ignore
        track_cross_output, _ = self.track_cross_attention(
            query=queries,
            key=track_memory,
            value=track_memory,
            key_padding_mask=track_padding_mask,
        )
        if self.layer_scale_3 is not None:
            track_cross_output = self.layer_scale_3 * track_cross_output
        queries = self.norm3(queries + self.dropout3(track_cross_output))

        # Sublayer 4: Feed-forward network
        # With LayerScale: output = LayerNorm(x + dropout(gamma_4 * FFN(Q)))
        # Without LayerScale: output = LayerNorm(x + dropout(FFN(Q)))
        ffn_output = self.ffn(queries)
        if self.layer_scale_4 is not None:
            ffn_output = self.layer_scale_4 * ffn_output
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
        drop_path_rate: Maximum stochastic depth drop probability for the
            last decoder layer (default: 0.0, disabled). Drop probabilities
            increase linearly from 0 for the first decoder layer to
            drop_path_rate for the last. Regularizes deep decoders by
            randomly skipping layer contributions during training.
            Reference: Huang et al., "Deep Networks with Stochastic Depth",
            ECCV 2016.
        layer_scale_init: Initial value for LayerScale gamma parameters in
            each decoder layer (default: 1e-4). Set to 0 or negative to
            disable LayerScale.
            Reference: Touvron et al., "Going deeper with Image Transformers",
            ICCV 2021 (CaiT).
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
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-4,
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
        # track cross-attention -> FFN. All post-norm with optional LayerScale.
        self.decoder_layers = nn.ModuleList([
            DualCrossAttentionDecoderLayer(
                decoder_dim=decoder_dim,
                num_heads=num_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=dropout,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(num_decoder_layers)
        ])

        # ---- Stochastic Depth (DropPath) per decoder layer ----
        # Drop probability increases linearly from 0 (first layer) to
        # drop_path_rate (last layer):
        #   p_i = drop_path_rate * i / (num_decoder_layers - 1)
        # for i = 0, 1, ..., num_decoder_layers - 1.
        # When num_decoder_layers == 1, the single layer gets rate 0.
        # Applied to the layer's residual delta: queries_out - queries_in.
        # Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
        per_layer_drop_rates = [
            drop_path_rate * layer_index / max(num_decoder_layers - 1, 1)
            for layer_index in range(num_decoder_layers)
        ]
        self.drop_paths = nn.ModuleList([
            DropPath(drop_probability=rate)
            for rate in per_layer_drop_rates
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
        # Clamped to [0.1, 2.0]: min=0.1 prevents too-sharp distributions,
        # max=2.0 prevents too-flat distributions that can't select tracks.
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
        # tau clamped to [0.1, 2.0]:
        #   min=0.1 prevents too-sharp distributions where logits explode
        #     and cross-entropy gradients spike when the matching changes
        #   max=2.0 prevents too-flat distributions that can't commit
        #     to selecting specific tracks over ~1130 candidates
        clamped_temperature = self.temperature.clamp(min=0.1, max=2.0)

        # ------------------------------------------------------------------
        # Step 8: Run through decoder layers with auxiliary outputs
        # ------------------------------------------------------------------
        all_layer_mask_logits = []
        all_layer_confidence_logits = []

        queries = all_queries
        for layer_index, decoder_layer in enumerate(self.decoder_layers):
            # Save queries before this layer for stochastic depth residual
            previous_queries = queries

            # Forward through DualCrossAttentionDecoderLayer
            queries = decoder_layer(
                queries=queries,
                compact_memory=encoded_memory,
                track_memory=track_memory,
                self_attention_mask=denoising_attention_mask,
                track_padding_mask=track_padding_mask,
            )  # (B, total_queries, decoder_dim)

            # Stochastic depth: apply DropPath to the layer's residual delta.
            # layer_delta = queries_after - queries_before captures the layer's
            # contribution. DropPath randomly zeros this delta during training
            # (with per-sample Bernoulli mask), effectively skipping the layer.
            # The surviving deltas are scaled by 1/(1-p) to preserve expected
            # values. During eval, all deltas pass through unchanged.
            layer_delta = queries - previous_queries
            queries = previous_queries + self.drop_paths[layer_index](layer_delta)

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
