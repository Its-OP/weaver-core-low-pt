"""DETR-style encoder-decoder head for tau-origin track finding.

Architecture follows the original DETR (Carion et al., ECCV 2020):
    1. Compact Token Encoder: self-attention on 128 compact tokens from the
       backbone, capturing global event context.
    2. Query Decoder: 15 learned queries cross-attend to encoded compact
       tokens, with self-attention among queries for coordination.
    3. Pointer Head: dot-product scoring between decoded queries and per-track
       enriched features, with a learned temperature parameter.
    4. Confidence Head: per-query binary prediction (exists / empty).

The backbone is external — this module only receives backbone outputs
(enriched features and compact tokens) and produces pointer logits and
confidence logits. Loss computation is handled by TauTrackFinder.

Reference:
    Carion, N. et al. "End-to-End Object Detection with Transformers."
    ECCV 2020. https://arxiv.org/abs/2005.12872
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional


class TauTrackFinderHead(nn.Module):
    """Encoder-decoder head for tau-origin pion track finding.

    Takes enriched per-track features and compact spatial tokens from
    the backbone, processes them through a transformer encoder-decoder,
    and produces pointer logits (which track each query selects) and
    confidence logits (whether each query represents a real track).

    Args:
        backbone_dim: Channel dimension of backbone outputs (default: 256).
        decoder_dim: Internal dimension for encoder/decoder (default: 256).
        pointer_dim: Dimension for dot-product pointer scoring (default: 128).
        num_queries: Number of learned query embeddings (default: 15).
            Should be >= max GT tracks for over-prediction.
        num_heads: Number of attention heads in encoder/decoder (default: 8).
        num_encoder_layers: Number of self-attention layers for compact
            tokens (default: 6). Captures global event context.
        num_decoder_layers: Number of decoder layers with self-attention
            among queries + cross-attention to encoded tokens (default: 6).
        dropout: Dropout rate in attention and feedforward layers (default: 0.1).
    """

    def __init__(
        self,
        backbone_dim: int = 256,
        decoder_dim: int = 256,
        pointer_dim: int = 128,
        num_queries: int = 15,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.decoder_dim = decoder_dim
        self.pointer_dim = pointer_dim
        self.num_queries = num_queries

        # ---- Memory projection (compact tokens → encoder input) ----
        # Projects backbone compact tokens from backbone_dim to decoder_dim.
        # LayerNorm stabilizes the scale for attention logits (same pattern
        # as MaskedTrackDecoder.backbone_projection + memory_norm).
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
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # ---- Learned Query Embeddings ----
        # Each query is a distinct learned vector producing unique attention
        # patterns over compact tokens via W_Q projection.
        # Initialization std = decoder_dim^{-0.5} (same as MaskedTrackDecoder).
        self.query_embeddings = nn.Embedding(num_queries, decoder_dim)
        nn.init.normal_(
            self.query_embeddings.weight, std=decoder_dim ** -0.5,
        )

        # ---- Query Decoder ----
        # Each layer: self-attention among queries (coordination to avoid
        # duplicates) → cross-attention to encoded compact tokens (event
        # reasoning) → FFN. All within the same layer.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,  # Post-norm (original DETR, Carion et al., ECCV 2020):
            # Applies LayerNorm AFTER each residual addition, bounding query norms.
            # Pre-norm (norm_first=True) caused query norms to explode from ~1 to
            # ~598 across 6 layers, making cross-attention contributions (<0.2% of
            # residual norm) negligible — decoded queries became input-independent.
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # ---- Track Key MLP ----
        # Projects enriched per-track features (B, backbone_dim, P) to pointer
        # key space (B, pointer_dim, P) for dot-product scoring with queries.
        # Uses Conv1d (operates on channel dimension across tracks).
        self.track_key_mlp = nn.Sequential(
            nn.Conv1d(backbone_dim, backbone_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(backbone_dim),
            nn.GELU(),
            nn.Conv1d(backbone_dim, pointer_dim, kernel_size=1),
        )

        # ---- Query Scoring MLP ----
        # Projects decoded queries to pointer query space for dot-product
        # scoring against track keys.
        self.query_scoring_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, pointer_dim),
        )

        # ---- Learned Temperature ----
        # Controls sharpness of softmax distribution over ~1130 tracks.
        # score(q, k) = (q · k) / τ
        # Learnable τ lets the model tune the trade-off between:
        #   - Too flat (τ large): can't commit to a specific track
        #   - Too sharp (τ small): gradients vanish for non-selected tracks
        # Clamped to min=0.01 to prevent division by near-zero.
        self.temperature = nn.Parameter(torch.ones(1))

        # ---- Pointer Context Projection ----
        # After computing pointer_logits, each query's soft attention distribution
        # over tracks is used to read out a "pointed context" vector:
        #   pointed_context_q = Σ_p softmax(pointer_logits)_{q,p} · enriched_features_p
        # This projects from backbone feature space to decoder space so it can be
        # concatenated with decoded queries for the confidence head.
        self.pointer_context_projection = nn.Sequential(
            nn.Linear(backbone_dim, decoder_dim),
            nn.GELU(),
        )

        # ---- Confidence Head ----
        # Per-query binary prediction: does this query point to a real tau
        # track (exists) or is it an unused slot (empty/∅)?
        # At inference: only trust queries with confident "exists" prediction.
        #
        # Input is concatenation of decoded query (decoder_dim) and pointed
        # context (decoder_dim), giving the head direct information about WHAT
        # each query is pointing to — not just the decoded query representation.
        #
        # Bias initialization: with 15 queries and typically ~3 active (20%),
        # init to logit(0.2) ≈ −1.4 so the network starts with a reasonable
        # prior instead of predicting 50/50.
        self.confidence_head = nn.Sequential(
            nn.Linear(2 * decoder_dim, pointer_dim),
            nn.GELU(),
            nn.Linear(pointer_dim, 1),
        )
        # Initialize bias of final linear layer to −1.4
        # σ(−1.4) ≈ 0.198 ≈ 3/15 (expected fraction of active queries)
        nn.init.constant_(self.confidence_head[-1].bias, -1.4)

    def forward(
        self,
        enriched_features: torch.Tensor,
        compact_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode compact tokens, decode queries, score tracks.

        Args:
            enriched_features: (B, backbone_dim, P) per-track enriched features
                from backbone.enrich(). Used for pointer scoring.
            compact_tokens: (B, backbone_dim, 128) compact spatial tokens from
                backbone.compact(). Used as encoder input / decoder memory.
            mask: (B, 1, P) boolean mask, True for valid tracks.

        Returns:
            Tuple of:
                pointer_logits: (B, num_queries, P) scores for each query
                    over all tracks. Padded positions are -inf.
                confidence_logits: (B, num_queries) per-query exists/empty
                    logits (pre-sigmoid).
        """
        batch_size = enriched_features.shape[0]
        device = enriched_features.device

        # Step 1: Project enriched features to pointer key space
        # track_key_mlp: Conv1d on (B, backbone_dim, P) → (B, pointer_dim, P)
        track_keys = self.track_key_mlp(enriched_features)  # (B, pointer_dim, P)
        track_keys = track_keys.transpose(1, 2)  # (B, P, pointer_dim)

        # Step 2: Project and encode compact tokens
        # (B, backbone_dim, 128) → transpose → (B, 128, backbone_dim) → project
        memory = self.memory_norm(
            self.memory_projection(compact_tokens.transpose(1, 2))
        )  # (B, 128, decoder_dim)

        # Self-attention encoder on compact tokens
        encoded_memory = self.transformer_encoder(memory)  # (B, 128, decoder_dim)

        # Step 3: Build query embeddings and decode
        query_indices = torch.arange(self.num_queries, device=device)
        queries = self.query_embeddings(query_indices)  # (num_queries, decoder_dim)
        queries = queries.unsqueeze(0).expand(
            batch_size, -1, -1,
        )  # (B, num_queries, decoder_dim)

        # Decoder: self-attention among queries + cross-attention to encoded memory
        decoded_queries = self.transformer_decoder(
            tgt=queries, memory=encoded_memory,
        )  # (B, num_queries, decoder_dim)

        # Step 4: Pointer scoring via dot-product with learned temperature
        # query_projections: (B, num_queries, pointer_dim)
        query_projections = self.query_scoring_mlp(decoded_queries)

        # score(q, k) = (q · k) / τ
        # τ clamped to min=0.01 to prevent division by near-zero
        clamped_temperature = self.temperature.clamp(min=0.01)
        pointer_logits = torch.bmm(
            query_projections, track_keys.transpose(1, 2),
        ) / clamped_temperature  # (B, num_queries, P)

        # Mask out padded track positions with -inf (softmax → 0 probability)
        padding_mask = ~mask.bool()  # (B, 1, P): True where padded
        pointer_logits = pointer_logits.masked_fill(
            padding_mask, float('-inf'),
        )

        # Step 5: Compute pointed context — soft-attention readout of enriched
        # track features weighted by each query's pointer distribution.
        # pointer_probabilities_{q,p} = softmax(pointer_logits_{q,p}) over tracks
        # pointed_features_q = Σ_p pointer_probabilities_{q,p} · enriched_features_p
        pointer_probabilities = functional.softmax(
            pointer_logits, dim=2,
        )  # (B, num_queries, P)
        # enriched_features: (B, backbone_dim, P) → transpose → (B, P, backbone_dim)
        pointed_features = torch.bmm(
            pointer_probabilities, enriched_features.transpose(1, 2),
        )  # (B, num_queries, backbone_dim)
        pointed_context = self.pointer_context_projection(
            pointed_features,
        )  # (B, num_queries, decoder_dim)

        # Step 6: Confidence prediction with pointer context
        # Concatenate decoded query representation with pointed context so the
        # confidence head sees both the query's internal state AND what track
        # features it is pointing to. This breaks the constant-logit equilibrium
        # where all queries output the same confidence regardless of input.
        confidence_input = torch.cat(
            [decoded_queries, pointed_context], dim=-1,
        )  # (B, num_queries, 2 * decoder_dim)
        confidence_logits = self.confidence_head(
            confidence_input,
        ).squeeze(-1)  # (B, num_queries)

        return pointer_logits, confidence_logits
