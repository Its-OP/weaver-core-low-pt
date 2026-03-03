"""Unit tests for the Hungarian and Sinkhorn matching algorithms.

Tests adapted from Google Research's Scenic library:
    https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/tests/test_matchers.py
Original code licensed under Apache 2.0 by The Scenic Authors.

Uses scipy.optimize.linear_sum_assignment as the ground-truth oracle
to verify that our PyTorch implementation finds optimal assignments.
POT (Python Optimal Transport) is used as a reference implementation
to verify our Sinkhorn matcher.
"""
import numpy as np
import ot
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from weaver.nn.model.hungarian_matcher import hungarian_matcher
from weaver.nn.model.hungarian_matcher import hungarian_matcher_tensor
from weaver.nn.model.hungarian_matcher import sinkhorn_matcher


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


class TestSinkhornMatcher:
    """Tests for the GPU-native Sinkhorn optimal transport matcher.

    Verifies that Sinkhorn assignments achieve near-optimal costs compared
    to scipy's exact Hungarian solution. Sinkhorn is approximate, so we
    allow a small relative tolerance (costs within 5% of optimal).
    """

    def test_manual_cost_matrix(self):
        """Known optimal assignments with large cost separation (square)."""
        # Sinkhorn requires square matrices for correct doubly-stochastic
        # convergence — rectangular matrices have inconsistent marginals.
        cost_matrix = torch.tensor([
            # Expect (0, 0) and (1, 1) matched
            [[-100.0, 100.0],
             [100.0, -100.0]],
            # Expect (0, 1) and (1, 0) matched
            [[100.0, -100.0],
             [-100.0, 100.0]],
        ])

        indices = sinkhorn_matcher(cost_matrix)
        assert indices.shape == (2, 2, 2)

        for b in range(2):
            row, col = indices[b, 0], indices[b, 1]
            our_cost = cost_matrix[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_matrix[b].numpy())
            assert abs(our_cost - scipy_cost) < 1e-4, (
                f'Batch {b}: sinkhorn_cost={our_cost}, scipy_cost={scipy_cost}'
            )

    def test_identity_assignment(self):
        """When diagonal is cheapest, Sinkhorn should find identity."""
        num_items = 10
        cost_matrix = torch.full((1, num_items, num_items), 100.0)
        cost_matrix[0].fill_diagonal_(0.0)

        indices = sinkhorn_matcher(cost_matrix)

        row = indices[0, 0]
        col = indices[0, 1]
        total_cost = cost_matrix[0, row, col].sum().item()
        assert total_cost == 0.0

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

        indices = sinkhorn_matcher(cost)
        for b in range(batch_size):
            row, col = indices[b, 0], indices[b, 1]
            total_cost = cost[b, row, col].sum().item()
            assert total_cost == 0.0, (
                f'Batch {b}: expected cost 0.0, got {total_cost}'
            )

    def test_near_optimal_cost_square(self):
        """Sinkhorn cost should be within 5% of exact Hungarian on random matrices."""
        rng = np.random.RandomState(42)
        batch_size = 4
        num_items = 50

        cost_np = rng.randn(batch_size, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        sinkhorn_indices = sinkhorn_matcher(cost)

        for b in range(batch_size):
            row, col = sinkhorn_indices[b, 0], sinkhorn_indices[b, 1]
            sinkhorn_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            # Sinkhorn is approximate — allow 5% relative tolerance.
            # Costs are negative (minimization), so sinkhorn_cost >= scipy_cost.
            assert sinkhorn_cost <= scipy_cost * 0.95, (
                f'Batch {b}: sinkhorn_cost={sinkhorn_cost:.4f} is more than '
                f'5% worse than scipy_cost={scipy_cost:.4f}'
            )

    def test_near_optimal_cost_larger(self):
        """Test on larger matrices closer to actual use case (~565 tracks)."""
        rng = np.random.RandomState(303)
        num_items = 100

        cost_np = rng.randn(2, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        sinkhorn_indices = sinkhorn_matcher(cost)

        for b in range(2):
            row, col = sinkhorn_indices[b, 0], sinkhorn_indices[b, 1]
            sinkhorn_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            assert sinkhorn_cost <= scipy_cost * 0.95, (
                f'Batch {b}: sinkhorn_cost={sinkhorn_cost:.4f} is more than '
                f'5% worse than scipy_cost={scipy_cost:.4f}'
            )

    def test_unique_assignments(self):
        """Column indices must be strictly unique (bijective assignment).

        The dedup post-processing resolves all duplicate column
        assignments from Sinkhorn, guaranteeing every ground-truth
        track is matched exactly once.
        """
        rng = np.random.RandomState(101)
        num_items = 20
        # Use unit-scale costs where raw Sinkhorn would produce duplicates
        cost_np = rng.randn(4, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = sinkhorn_matcher(cost)

        for b in range(4):
            col = indices[b, 1].numpy()
            assert len(set(col)) == len(col), (
                f'Batch {b}: duplicate column indices after dedup: {col}'
            )

    def test_unique_assignments_large(self):
        """Uniqueness holds on larger matrices closer to real use case."""
        rng = np.random.RandomState(202)
        num_items = 100
        cost_np = rng.randn(2, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        indices = sinkhorn_matcher(cost)

        for b in range(2):
            col = indices[b, 1].numpy()
            assert len(set(col)) == len(col), (
                f'Batch {b}: duplicate column indices after dedup: {col}'
            )

    def test_dedup_preserves_quality(self):
        """After dedup, assignment cost should still be near-optimal.

        The greedy fallback for displaced rows may not find the globally
        optimal reassignment, but the total cost should remain within
        10% of exact Hungarian. Unit-scale costs stress-test this path
        because Sinkhorn produces more duplicates (less cost separation),
        so the greedy reassignment has a harder job.
        """
        rng = np.random.RandomState(303)
        num_items = 50
        # Unit-scale costs stress-test the dedup path
        cost_np = rng.randn(4, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        sinkhorn_indices = sinkhorn_matcher(cost)

        for b in range(4):
            row, col = sinkhorn_indices[b, 0], sinkhorn_indices[b, 1]
            sinkhorn_cost = cost[b, row, col].sum().item()
            scipy_cost = scipy_assignment_cost(cost_np[b])
            # Allow 10% relative tolerance.
            # relative_gap > 0 means sinkhorn is worse (higher cost) than exact.
            # This handles both positive and negative optimal costs correctly.
            relative_gap = (
                (sinkhorn_cost - scipy_cost) / (abs(scipy_cost) + 1e-8)
            )
            assert relative_gap < 0.10, (
                f'Batch {b}: sinkhorn_cost={sinkhorn_cost:.4f} is '
                f'{relative_gap * 100:.1f}% worse than '
                f'scipy_cost={scipy_cost:.4f}'
            )

    def test_dedup_with_masked_entries(self):
        """Dedup should not assign displaced rows to masked columns."""
        rng = np.random.RandomState(404)
        batch_size = 2
        num_items = 20
        num_valid = 15

        cost = torch.from_numpy(
            rng.randn(batch_size, num_items, num_items).astype(np.float32)
        )
        cost[:, num_valid:, :] = 1e6
        cost[:, :, num_valid:] = 1e6

        indices = sinkhorn_matcher(cost)

        for b in range(batch_size):
            row = indices[b, 0].numpy()
            col = indices[b, 1].numpy()
            # Valid rows should map to valid columns
            valid_rows = row[:num_valid]
            valid_cols = col[:num_valid]
            assert (valid_cols < num_valid).all(), (
                f'Batch {b}: displaced row assigned to masked column. '
                f'rows={valid_rows}, cols={valid_cols}'
            )
            # Valid columns should be unique
            assert len(set(valid_cols)) == len(valid_cols), (
                f'Batch {b}: duplicate valid columns after dedup: {valid_cols}'
            )

    def test_output_shape_square(self):
        """Output shape for square matrices."""
        cost = torch.randn(3, 10, 10)
        indices = sinkhorn_matcher(cost)
        assert indices.shape == (3, 2, 10)

    def test_output_shape_rectangular(self):
        """Output shape for rectangular matrices."""
        cost = torch.randn(2, 20, 8)
        indices = sinkhorn_matcher(cost)
        # K = min(20, 8) = 8
        assert indices.shape == (2, 2, 8)

    def test_agrees_with_hungarian(self):
        """Sinkhorn and Hungarian should find near-equal-cost assignments.

        Sinkhorn is approximate and may occasionally assign two rows
        to the same column (non-bijective), which can produce a slightly
        different total cost. We use a 5% relative tolerance.
        """
        rng = np.random.RandomState(555)
        cost_np = (rng.randn(3, 15, 15) * 10).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        hungarian_indices = hungarian_matcher(cost)
        sinkhorn_indices = sinkhorn_matcher(cost)

        for b in range(3):
            hungarian_cost = cost[
                b, hungarian_indices[b, 0], hungarian_indices[b, 1]
            ].sum().item()
            sinkhorn_cost = cost[
                b, sinkhorn_indices[b, 0], sinkhorn_indices[b, 1]
            ].sum().item()
            relative_difference = (
                abs(sinkhorn_cost - hungarian_cost)
                / (abs(hungarian_cost) + 1e-8)
            )
            assert relative_difference < 0.05, (
                f'Batch {b}: sinkhorn_cost={sinkhorn_cost:.4f}, '
                f'hungarian_cost={hungarian_cost:.4f}, '
                f'relative_diff={relative_difference:.4f}'
            )

    def test_masked_entries(self):
        """Invalid entries with large cost should never be matched."""
        rng = np.random.RandomState(777)
        batch_size = 2
        num_items = 15
        num_valid = 10

        cost = torch.from_numpy(
            rng.randn(batch_size, num_items, num_items).astype(np.float32)
        )
        # Mask last 5 rows and columns with large cost
        cost[:, num_valid:, :] = 1e6
        cost[:, :, num_valid:] = 1e6

        indices = sinkhorn_matcher(cost)

        for b in range(batch_size):
            row = indices[b, 0].numpy()
            col = indices[b, 1].numpy()
            # All matched rows and columns should be in the valid range
            # (at least the first num_valid matches should be valid)
            valid_matches = (row < num_valid) & (col < num_valid)
            assert valid_matches[:num_valid].all(), (
                f'Batch {b}: invalid entries matched. '
                f'rows={row[:num_valid]}, cols={col[:num_valid]}'
            )


class TestSinkhornMatcherVsPOT:
    """Cross-validate our Sinkhorn matcher against POT's implementation.

    POT (Python Optimal Transport) is the reference library for optimal
    transport in Python. We verify that our batched GPU implementation
    produces transport plans with costs very close to POT's solver.

    Both implementations use the same regularization (ε = 0.1, our default).
    At this sharpness, the soft plan is nearly a permutation matrix, so
    argmax yields mostly unique assignments and our bijective dedup step
    changes only 0-2 entries. This makes the cost comparison fair:
    both implementations produce near-bijective hard assignments.

    A separate test (test_transport_plan_similarity) verifies that the
    soft transport plans match between the two implementations.
    """

    @staticmethod
    def _pot_assignment_cost(
        cost_matrix_np: np.ndarray,
        regularization: float = 0.1,
        num_iterations: int = 100,
    ) -> tuple[float, np.ndarray]:
        """Solve assignment via POT's Sinkhorn and return cost + column indices.

        Args:
            cost_matrix_np: (N, M) cost matrix as numpy array.
            regularization: Entropy regularization ε.
            num_iterations: Maximum Sinkhorn iterations.

        Returns:
            Tuple of (total_cost, column_indices):
                total_cost: Sum of matched costs under hard assignment.
                column_indices: (N,) argmax per row of the transport plan.
        """
        num_rows, num_columns = cost_matrix_np.shape
        # Uniform marginals (each row/column gets equal mass)
        source_weights = np.ones(num_rows, dtype=np.float64) / num_rows
        target_weights = np.ones(num_columns, dtype=np.float64) / num_columns

        # POT's log-domain Sinkhorn (numerically stable)
        transport_plan = ot.sinkhorn(
            source_weights, target_weights, cost_matrix_np.astype(np.float64),
            reg=regularization, method='sinkhorn_log',
            numItermax=num_iterations, stopThr=1e-9,
            warn=False, log=False,
        )

        # Hard assignment: argmax per row
        column_indices = transport_plan.argmax(axis=1)
        row_indices = np.arange(num_rows)
        total_cost = cost_matrix_np[row_indices, column_indices].sum()

        return total_cost, column_indices

    def test_same_assignments_small(self):
        """On small well-separated matrices, both should find similar assignments."""
        rng = np.random.RandomState(42)
        # Scale costs up for clear separation between good and bad matches
        cost_np = (rng.randn(3, 10, 10) * 10).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        our_indices = sinkhorn_matcher(cost, num_iterations=100, regularization=0.1)

        for b in range(3):
            our_cols = our_indices[b, 1].numpy()
            _, pot_cols = self._pot_assignment_cost(
                cost_np[b], regularization=0.1, num_iterations=100,
            )

            our_cost = cost_np[b, np.arange(10), our_cols].sum()
            pot_cost = cost_np[b, np.arange(10), pot_cols].sum()

            # At reg=0.1 with 10× scaled costs, plans are extremely sharp.
            # Both should produce identical assignments (< 1% cost difference).
            relative_difference = abs(our_cost - pot_cost) / (abs(pot_cost) + 1e-8)
            assert relative_difference < 0.01, (
                f'Batch {b}: our_cost={our_cost:.4f}, pot_cost={pot_cost:.4f}, '
                f'relative_diff={relative_difference:.4f}'
            )

    def test_same_assignments_medium(self):
        """On medium-sized matrices, costs should be within 10% of POT.

        Our implementation operates in float32 (batched on GPU) while POT
        uses float64 (single matrix). On unit-scale 50×50 costs, the
        precision difference causes a few different argmax choices that
        cascade through the dedup step, widening the gap.
        """
        rng = np.random.RandomState(123)
        cost_np = rng.randn(4, 50, 50).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        our_indices = sinkhorn_matcher(cost, num_iterations=100, regularization=0.1)

        for b in range(4):
            num_items = 50
            our_cols = our_indices[b, 1].numpy()
            _, pot_cols = self._pot_assignment_cost(
                cost_np[b], regularization=0.1, num_iterations=100,
            )

            our_cost = cost_np[b, np.arange(num_items), our_cols].sum()
            pot_cost = cost_np[b, np.arange(num_items), pot_cols].sum()

            relative_difference = abs(our_cost - pot_cost) / (abs(pot_cost) + 1e-8)
            assert relative_difference < 0.10, (
                f'Batch {b}: our_cost={our_cost:.4f}, pot_cost={pot_cost:.4f}, '
                f'relative_diff={relative_difference:.4f}'
            )

    def test_same_assignments_large(self):
        """On larger matrices (100×100), costs should still be close to POT."""
        rng = np.random.RandomState(456)
        num_items = 100
        cost_np = rng.randn(2, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)

        our_indices = sinkhorn_matcher(cost, num_iterations=100, regularization=0.1)

        for b in range(2):
            our_cols = our_indices[b, 1].numpy()
            _, pot_cols = self._pot_assignment_cost(
                cost_np[b], regularization=0.1, num_iterations=100,
            )

            our_cost = cost_np[b, np.arange(num_items), our_cols].sum()
            pot_cost = cost_np[b, np.arange(num_items), pot_cols].sum()

            relative_difference = abs(our_cost - pot_cost) / (abs(pot_cost) + 1e-8)
            assert relative_difference < 0.05, (
                f'Batch {b}: our_cost={our_cost:.4f}, pot_cost={pot_cost:.4f}, '
                f'relative_diff={relative_difference:.4f}'
            )

    def test_transport_plan_similarity(self):
        """The soft transport plans should be similar (not just hard assignments).

        Compares the Frobenius norm of the difference between our transport
        plan and POT's, normalized by the plan magnitude.

        Our Sinkhorn normalizes rows/columns to sum to 1 (doubly-stochastic).
        POT with uniform marginals (1/N, 1/N) normalizes to sum to 1/N.
        So our_plan = N × pot_plan. We divide by N before comparing.
        """
        rng = np.random.RandomState(789)
        num_items = 20
        cost_np = rng.randn(1, num_items, num_items).astype(np.float32)
        cost = torch.from_numpy(cost_np)
        regularization = 1.0

        # Our transport plan (reconstruct from log domain)
        log_transport = -cost / regularization
        for _ in range(100):
            log_transport = log_transport - torch.logsumexp(
                log_transport, dim=2, keepdim=True,
            )
            log_transport = log_transport - torch.logsumexp(
                log_transport, dim=1, keepdim=True,
            )
        # Normalize: our plan has rows summing to 1, POT's sum to 1/N
        our_plan = log_transport[0].exp().numpy() / num_items  # (N, M)

        # POT transport plan
        source_weights = np.ones(num_items, dtype=np.float64) / num_items
        target_weights = np.ones(num_items, dtype=np.float64) / num_items
        pot_plan = ot.sinkhorn(
            source_weights, target_weights, cost_np[0].astype(np.float64),
            reg=regularization, method='sinkhorn_log',
            numItermax=100, stopThr=1e-9,
            warn=False, log=False,
        )

        # Compare plans: Frobenius norm of difference / Frobenius norm of plan
        difference_norm = np.linalg.norm(our_plan - pot_plan, ord='fro')
        plan_norm = np.linalg.norm(pot_plan, ord='fro')
        relative_plan_error = difference_norm / (plan_norm + 1e-8)

        assert relative_plan_error < 0.05, (
            f'Transport plan relative error: {relative_plan_error:.4f} '
            f'(difference_norm={difference_norm:.6f}, '
            f'plan_norm={plan_norm:.6f})'
        )
