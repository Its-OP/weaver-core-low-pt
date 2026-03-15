"""Object condensation head for per-track tau-pion identification.

Lightweight MLP head on frozen per-track enriched features. No transformer
decoder, no Hungarian matching — just per-track predictions.

Each track predicts:
    1. beta ∈ (0, 1) — confidence that this track is a tau-origin pion.
       High beta → likely tau pion. Used for ranking at inference.
    2. Clustering coordinates ∈ R^D — learned embedding space where same-object
       tracks cluster together and different-object tracks repel.

At inference: rank all tracks by beta score, take top-K as tau pion candidates.
This is a recommender/ranking approach (recall@K evaluation).

Reference:
    Kieseler, J. "Object condensation: one-stage grid-free multi-particle
    reconstruction in physics detector event reconstruction."
    Eur. Phys. J. C 80, 886 (2020). https://arxiv.org/abs/2002.03605
"""

import torch
import torch.nn as nn


class ObjectCondensationHead(nn.Module):
    """Per-track MLP head producing beta scores and clustering coordinates.

    Architecture per track (applied via Conv1d, kernel_size=1):
        beta_head:       256 → 128 → 1 → sigmoid  ∈ (0, 1)
        clustering_head: 256 → 128 → D  (D = clustering_dim)

    Args:
        input_dim: Channel dimension of enriched features (default: 256).
        hidden_dim: Hidden layer width (default: 128).
        clustering_dim: Dimensionality of the learned clustering space
            (default: 8). Higher values give more capacity for separating
            objects but increase the loss computation.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        clustering_dim: int = 8,
    ):
        super().__init__()
        self.clustering_dim = clustering_dim

        # ---- Beta head ----
        # Predicts P(tau pion | enriched features) per track.
        # Conv1d(kernel=1) is equivalent to per-track Linear layer,
        # applied efficiently across the track dimension.
        self.beta_head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

        # ---- Clustering head ----
        # Projects each track to a learned embedding space where same-tau
        # tracks are pulled together and different-tau tracks are pushed apart.
        self.clustering_head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, clustering_dim, kernel_size=1),
        )

    def forward(
        self,
        enriched_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict beta scores and clustering coordinates per track.

        Args:
            enriched_features: (B, input_dim, P) per-track enriched features
                from frozen backbone.

        Returns:
            Tuple of:
                beta: (B, P) confidence scores ∈ (0, 1) per track.
                beta_logits: (B, P) pre-sigmoid logits for focal BCE loss.
                clustering_coordinates: (B, clustering_dim, P) learned
                    embedding coordinates per track.
        """
        # Beta logits: (B, input_dim, P) → (B, 1, P) → (B, P)
        beta_logits = self.beta_head(enriched_features).squeeze(1)

        # Beta: sigmoid(logits) → (B, P) ∈ (0, 1)
        beta = torch.sigmoid(beta_logits)

        # Clustering: (B, input_dim, P) → (B, clustering_dim, P)
        clustering_coordinates = self.clustering_head(enriched_features)

        return beta, beta_logits, clustering_coordinates
