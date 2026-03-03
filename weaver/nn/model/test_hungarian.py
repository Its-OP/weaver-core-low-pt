"""Unit tests for the Hungarian matching algorithm.

Tests adapted from Google Research's Scenic library:
    https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/tests/test_matchers.py
Original code licensed under Apache 2.0 by The Scenic Authors.

Uses scipy.optimize.linear_sum_assignment as the ground-truth oracle
to verify that our PyTorch implementation finds optimal assignments.
"""
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from weaver.nn.model.hungarian_matcher import hungarian_matcher
from weaver.nn.model.hungarian_matcher import hungarian_matcher_tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scipy_assignment_cost(cost_matrix_np: np.ndarray) -> float:
    """Compute optimal assignment cost using scipy (ground truth)."""
    row_indices, col_indices = linear_sum_assignment(cost_matrix_np)
    return cost_matrix_np[row_indices, col_indices].sum()


def our_assignment_cost(
    cost_matrix: torch.Tensor,
) -> list[float]:
    """Compute assignment costs using our matcher for a batch."""
    # hungarian_matcher returns (B, 2, K) where K = min(N, M)
    # indices[b, 0, :] = row indices, indices[b, 1, :] = col indices
    indices = hungarian_matcher(cost_matrix)
    batch_size = cost_matrix.shape[0]
    costs = []
    for b in range(batch_size):
        row = indices[b, 0]
        col = indices[b, 1]
        costs.append(cost_matrix[b, row, col].sum().item())
    return costs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHungarianMatcher:
    """Test suite for the PyTorch Hungarian matching implementation."""

    def test_manual_cost_matrix(self):
        """Test with known optimal assignments (from Scenic test suite).

        Adapted from test_manual_cost_matrix in Scenic's test_matchers.py.
        """
        cost_matrix = torch.tensor([
            # Expect (0, 0) and (1, 1) matched
            [[-100.0, 100.0],
             [100.0, -100.0],
             [100.0, 100.0]],
            # Expect (0, 0) and (2, 1) matched
            [[-100.0, 100.0],
             [100.0, 100.0],
             [100.0, -100.0]],
        ])

        indices = hungarian_matcher(cost_matrix)

        # Check shape: (B=2, 2, K=2) since min(3, 2) = 2
        assert indices.shape == (2, 2, 2)

        # Check costs match scipy
        for b in range(2):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost_matrix[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_matrix[b].numpy())
            assert abs(our_cost - scipy_cost) < 1e-4, (
                f'Batch {b}: our_cost={our_cost}, scipy_cost={scipy_cost}'
            )

    def test_identity_assignment(self):
        """When diagonal is cheapest, matcher should find identity permutation."""
        num_items = 10
        # Diagonal = 0, off-diagonal = 100
        cost_matrix = torch.full((1, num_items, num_items), 100.0)
        cost_matrix[0].fill_diagonal_(0.0)

        indices = hungarian_matcher(cost_matrix)

        row = indices[0, 0]
        col = indices[0, 1]
        # Optimal assignment should have cost 0 (diagonal)
        total_cost = cost_matrix[0, row, col].sum().item()
        assert total_cost == 0.0

    def test_matches_scipy_square(self):
        """Random square cost matrices must yield same optimal cost as scipy.

        Adapted from test_cost_matches_scipy in Scenic's test_matchers.py.
        """
        rng = np.random.RandomState(42)
        batch_size = 4
        num_items = 50

        cost_np = rng.randn(batch_size, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        our_costs = our_assignment_cost(cost)

        for b in range(batch_size):
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert abs(our_costs[b] - scipy_cost) < 1e-3, (
                f'Batch {b}: our_cost={our_costs[b]:.6f}, '
                f'scipy_cost={scipy_cost:.6f}'
            )

    def test_matches_scipy_rectangular_more_rows(self):
        """Rectangular cost matrix with N > M (more predictions than targets).

        Adapted from test_cost_matches_scipy_rect_n_bigger_m in Scenic.
        """
        rng = np.random.RandomState(123)
        batch_size = 4
        num_rows = 30
        num_cols = 15

        cost_np = rng.randn(batch_size, num_rows, num_cols).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = hungarian_matcher(cost)
        # K = min(30, 15) = 15
        assert indices.shape == (batch_size, 2, num_cols)

        for b in range(batch_size):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert abs(our_cost - scipy_cost) < 1e-3, (
                f'Batch {b}: our_cost={our_cost:.6f}, '
                f'scipy_cost={scipy_cost:.6f}'
            )

    def test_matches_scipy_rectangular_more_cols(self):
        """Rectangular cost matrix with N < M (fewer predictions than targets).

        Adapted from test_cost_matches_scipy_rect_n_smaller_m in Scenic.
        """
        rng = np.random.RandomState(456)
        batch_size = 4
        num_rows = 15
        num_cols = 30

        cost_np = rng.randn(batch_size, num_rows, num_cols).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = hungarian_matcher(cost)
        # K = min(15, 30) = 15
        assert indices.shape == (batch_size, 2, num_rows)

        for b in range(batch_size):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert abs(our_cost - scipy_cost) < 1e-3, (
                f'Batch {b}: our_cost={our_cost:.6f}, '
                f'scipy_cost={scipy_cost:.6f}'
            )

    def test_permuted_identity(self):
        """A randomly permuted identity matrix should be perfectly solved."""
        rng = np.random.RandomState(789)
        batch_size = 3
        num_items = 20

        cost = torch.full((batch_size, num_items, num_items), 10.0)
        for b in range(batch_size):
            perm = rng.permutation(num_items)
            for i, j in enumerate(perm):
                cost[b, i, j] = 0.0

        our_costs = our_assignment_cost(cost)
        for b in range(batch_size):
            assert our_costs[b] == 0.0, (
                f'Batch {b}: expected cost 0.0, got {our_costs[b]}'
            )

    def test_unique_assignments(self):
        """Each row and column index must appear at most once."""
        rng = np.random.RandomState(101)
        cost_np = rng.randn(2, 20, 20).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = hungarian_matcher(cost)

        for b in range(2):
            row = indices[b, 0].numpy()
            col = indices[b, 1].numpy()
            assert len(set(row)) == len(row), f'Duplicate row indices: {row}'
            assert len(set(col)) == len(col), f'Duplicate col indices: {col}'

    def test_batch_consistency(self):
        """Processing as a batch should give same result as processing one-by-one."""
        rng = np.random.RandomState(202)
        batch_size = 4
        num_items = 15

        cost_np = rng.randn(batch_size, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        # Batch
        batch_indices = hungarian_matcher(cost)
        # One-by-one
        for b in range(batch_size):
            single_indices = hungarian_matcher(cost[b:b+1])
            batch_cost = cost[b, batch_indices[b, 0], batch_indices[b, 1]].sum()
            single_cost = cost[b, single_indices[0, 0], single_indices[0, 1]].sum()
            assert abs(batch_cost.item() - single_cost.item()) < 1e-4

    def test_larger_matrix(self):
        """Test with a larger matrix closer to our actual use case (~280 tracks)."""
        rng = np.random.RandomState(303)
        num_items = 100  # Not 280 to keep test fast on CPU

        cost_np = rng.randn(2, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        our_costs = our_assignment_cost(cost)

        for b in range(2):
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert abs(our_costs[b] - scipy_cost) < 1e-2, (
                f'Batch {b}: our_cost={our_costs[b]:.6f}, '
                f'scipy_cost={scipy_cost:.6f}'
            )

    def test_output_shape_square(self):
        """Output shape for square matrices."""
        cost = torch.randn(3, 10, 10)
        indices = hungarian_matcher(cost)
        assert indices.shape == (3, 2, 10)

    def test_output_shape_rectangular(self):
        """Output shape for rectangular matrices."""
        cost = torch.randn(2, 20, 8)
        indices = hungarian_matcher(cost)
        # K = min(20, 8) = 8
        assert indices.shape == (2, 2, 8)


class TestHungarianMatcherTensor:
    """Tests for the pure-PyTorch tensor implementation.

    Uses smaller matrices to keep tests fast (the tensor implementation
    is slower than scipy for large matrices due to Python-level iteration).
    """

    def test_manual_cost_matrix(self):
        """Known optimal assignments (from Scenic test suite)."""
        cost_matrix = torch.tensor([
            [[-100.0, 100.0],
             [100.0, -100.0],
             [100.0, 100.0]],
            [[-100.0, 100.0],
             [100.0, 100.0],
             [100.0, -100.0]],
        ])
        indices = hungarian_matcher_tensor(cost_matrix)
        assert indices.shape == (2, 2, 2)

        for b in range(2):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost_matrix[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_matrix[b].numpy())
            assert abs(our_cost - scipy_cost) < 1e-4

    def test_matches_scipy_square(self):
        """Random square matrices must yield same cost as scipy."""
        rng = np.random.RandomState(42)
        cost_np = rng.randn(4, 20, 20).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = hungarian_matcher_tensor(cost)
        for b in range(4):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert abs(our_cost - scipy_cost) < 1e-3

    def test_matches_scipy_rectangular(self):
        """Rectangular (N > M) matrix."""
        rng = np.random.RandomState(123)
        cost_np = rng.randn(2, 15, 8).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = hungarian_matcher_tensor(cost)
        assert indices.shape == (2, 2, 8)

        for b in range(2):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert abs(our_cost - scipy_cost) < 1e-3

    def test_agrees_with_scipy_matcher(self):
        """Both implementations should find equal-cost assignments."""
        rng = np.random.RandomState(999)
        cost_np = rng.randn(3, 15, 15).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        scipy_indices = hungarian_matcher(cost)
        tensor_indices = hungarian_matcher_tensor(cost)

        for b in range(3):
            scipy_cost = cost[b, scipy_indices[b, 0], scipy_indices[b, 1]].sum()
            tensor_cost = cost[b, tensor_indices[b, 0], tensor_indices[b, 1]].sum()
            assert abs(scipy_cost.item() - tensor_cost.item()) < 1e-3
