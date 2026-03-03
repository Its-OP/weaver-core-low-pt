"""Batched Hungarian matching (optimal bipartite assignment) for PyTorch.

Provides `hungarian_matcher(cost_matrix)` which solves batched linear
assignment problems. Cost matrix is computed on GPU; the assignment
solver uses scipy on CPU (fast C implementation, ~1ms per 280×280 matrix).
Only the cost matrix and index tensors cross the device boundary.

Also includes a pure-PyTorch implementation (`hungarian_matcher_tensor`)
ported from Google Research's Scenic library, which avoids CPU transfers
entirely. It is correct (verified against scipy in tests) but currently
slower for large matrices due to Python-level iteration overhead. It will
become the preferred backend when PyTorch adds native `while_loop` or
when a CUDA Hungarian kernel becomes available.

Scenic source (Apache 2.0, by Jiquan Ngiam <jngiam@google.com>):
    https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/hungarian_cover.py

Reference:
    Kuhn, H.W. "The Hungarian Method for the Assignment Problem."
    Naval Research Logistics Quarterly, 2(1-2):83-97, 1955.
    https://www.cse.ust.hk/~golin/COMP572/Notes/Matching.pdf
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _prepare(weights: torch.Tensor) -> torch.Tensor:
    """Subtract row and column minima to create initial zeros.

    Neither operation changes the optimal assignment but provides
    a better starting point for the greedy matching.
    Corresponds to preprocessing + step 1 of the Hungarian algorithm.

    Args:
        weights: (B, N, M) cost matrix.

    Returns:
        Prepared weights of the same shape with many near-zero entries.
    """
    assert weights.ndim == 3
    # Subtract row minima: min over columns (dim=2)
    weights = weights - weights.min(dim=2, keepdim=True).values
    # Subtract column minima: min over rows (dim=1)
    weights = weights - weights.min(dim=1, keepdim=True).values
    return weights


def _greedy_assignment(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Greedily assign rows to columns based on an adjacency matrix.

    Iterates over rows using a sequential scan. Each row is assigned to
    its highest-index available column (not yet taken by a previous row).

    Args:
        adjacency_matrix: (B, N, M) boolean tensor indicating valid pairings.

    Returns:
        assignment: (B, N, M) boolean tensor with at most one True per
            row and per column.
    """
    batch_size, num_rows, num_columns = adjacency_matrix.shape

    # Transpose to (N, B, M) for sequential row-wise scan
    adjacency_transposed = adjacency_matrix.permute(1, 0, 2)  # (N, B, M)

    column_assigned = torch.zeros(
        batch_size, num_columns, dtype=torch.bool,
        device=adjacency_matrix.device,
    )
    assignment_rows = []

    for row_idx in range(num_rows):
        row_adjacency = adjacency_transposed[row_idx]  # (B, M)

        # Viable candidates: adjacent AND not yet assigned
        candidates = row_adjacency & ~column_assigned  # (B, M)

        # Deterministically assign to the highest-index candidate
        # (matching Scenic's argmax behavior)
        max_candidate_index = candidates.long().argmax(dim=1)  # (B,)
        candidate_indicator = torch.nn.functional.one_hot(
            max_candidate_index, num_columns,
        ).bool()  # (B, M)
        # Only assign if there was at least one candidate
        candidate_indicator = candidate_indicator & candidates

        # Update column tracking
        column_assigned = column_assigned | candidate_indicator

        assignment_rows.append(candidate_indicator)

    # Stack rows: (N, B, M) → permute → (B, N, M)
    assignment = torch.stack(assignment_rows, dim=0).permute(1, 0, 2)
    return assignment


def _find_augmenting_path(
    assignment: torch.Tensor,
    adjacency_matrix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Find augmenting paths from unassigned rows through the bipartite graph.

    Alternates between:
        - Unassigned edges: unassigned row → column (via adjacency, not assignment)
        - Assigned edges: column → row (via existing assignment)

    The search propagates labels (1-indexed source row IDs) through the graph.
    When an unassigned column is reached, an augmenting path exists.

    Args:
        assignment: (B, N, M) boolean assignment matrix.
        adjacency_matrix: (B, N, M) boolean adjacency matrix.

    Returns:
        State dict with tensors for backtracking the augmenting path:
            - 'columns': (B, 1, M) int — source row ID reaching each column
            - 'columns_from_row': (B, M) int — row immediately before each column
            - 'rows': (B, N, 1) int — source row ID reaching each row
            - 'rows_from_column': (B, N) int — column immediately before each row
            - 'new_columns': (B, M) bool — unassigned columns reachable via a path
    """
    batch_size, num_rows, num_columns = assignment.shape
    device = assignment.device

    unassigned_rows = ~assignment.any(dim=2, keepdim=True)  # (B, N, 1)
    unassigned_columns = ~assignment.any(dim=1, keepdim=True)  # (B, 1, M)

    # Edges available for path extension
    unassigned_pairings = (
        adjacency_matrix & ~assignment
    ).int()  # (B, N, M)
    existing_pairings = assignment.int()  # (B, N, M)

    # Initialize: unassigned rows get unique 1-indexed IDs
    row_indices = torch.arange(
        1, num_rows + 1, dtype=torch.int32, device=device,
    )  # (N,)
    initial_rows = row_indices.reshape(1, -1, 1).expand(
        batch_size, -1, 1,
    )  # (B, N, 1)
    initial_rows = initial_rows * unassigned_rows.int()  # Zero out assigned

    state = {
        'columns': torch.zeros(
            batch_size, 1, num_columns, dtype=torch.int32, device=device,
        ),
        'columns_from_row': torch.zeros(
            batch_size, num_columns, dtype=torch.int32, device=device,
        ),
        'rows': initial_rows.clone(),
        'rows_from_column': torch.zeros(
            batch_size, num_rows, dtype=torch.int32, device=device,
        ),
    }

    current_rows = initial_rows.clone()  # (B, N, 1)

    while current_rows.sum() > 0:
        # Forward step: rows → columns via unassigned pairings
        # potential_columns[b, i, j] = current_rows[b, i] * unassigned_pairings[b, i, j]
        potential_columns = current_rows * unassigned_pairings  # (B, N, M)
        current_columns = potential_columns.max(dim=1, keepdim=True).values  # (B, 1, M)
        current_columns_from_row = 1 + potential_columns.argmax(dim=1)  # (B, M)

        # Only keep newly discovered columns
        already_reached = state['columns'] > 0
        current_columns = current_columns.where(
            ~already_reached, torch.zeros_like(current_columns),
        )
        current_columns_from_row = (
            current_columns_from_row * (current_columns > 0).int().squeeze(1)
        )

        # Backward step: columns → rows via existing pairings
        potential_rows = current_columns * existing_pairings  # (B, N, M)
        current_rows = potential_rows.max(dim=2, keepdim=True).values  # (B, N, 1)
        current_rows_from_column = 1 + potential_rows.argmax(dim=2)  # (B, N)

        # Only keep newly discovered rows
        already_reached_rows = state['rows'] > 0
        current_rows = current_rows.where(
            ~already_reached_rows, torch.zeros_like(current_rows),
        )
        current_rows_from_column = (
            current_rows_from_column * (current_rows > 0).int().squeeze(2)
        )

        # Update state
        state['columns'] = torch.maximum(state['columns'], current_columns)
        state['columns_from_row'] = torch.maximum(
            state['columns_from_row'], current_columns_from_row,
        )
        state['rows'] = torch.maximum(state['rows'], current_rows)
        state['rows_from_column'] = torch.maximum(
            state['rows_from_column'], current_rows_from_column,
        )

    # Columns reachable via augmenting path AND unassigned
    new_columns = (state['columns'] > 0) & unassigned_columns  # (B, 1, M)
    state['new_columns'] = new_columns.squeeze(1)  # (B, M)
    return state


def _improve_assignment(
    assignment: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Improve assignment by flipping edges along the augmenting path.

    Backtrack from a newly reachable unassigned column through the state
    dict, alternating column→row and row→column steps. Flip all
    assignment edges on this path (add unassigned, remove assigned).

    Args:
        assignment: (B, N, M) boolean assignment matrix.
        state: State dict from _find_augmenting_path.

    Returns:
        Updated assignment with one more matched pair.
    """
    batch_size, num_rows, num_columns = assignment.shape
    device = assignment.device

    # Start from an unassigned column reachable via augmenting path
    current_column_index = state['new_columns'].long().argmax(dim=1)  # (B,)
    active = state['new_columns'].gather(
        1, current_column_index.unsqueeze(1),
    ).squeeze(1)  # (B,) bool

    flip_matrix = torch.zeros(
        batch_size, num_rows, num_columns, dtype=torch.int32, device=device,
    )
    batch_range = torch.arange(batch_size, device=device)

    while active.any():
        # Column → row: find the row that reached this column
        current_row_index = (
            state['columns_from_row'].gather(
                1, current_column_index.unsqueeze(1),
            ).squeeze(1) - 1
        )  # (B,) 0-indexed
        current_row_index = current_row_index.clamp(min=0)

        # Flip this (row, column) edge
        flip_matrix[batch_range, current_row_index, current_column_index] += (
            active.int()
        )

        # Row → column: find the column this row was reached from
        next_column_index = (
            state['rows_from_column'].gather(
                1, current_row_index.unsqueeze(1),
            ).squeeze(1) - 1
        )  # (B,) 0-indexed

        # Deactivate paths that have reached an unassigned row (index < 0)
        active = active & (next_column_index >= 0)
        next_column_index = next_column_index.clamp(min=0)

        # Flip this (row, next_column) edge too
        flip_matrix[batch_range, current_row_index, next_column_index] += (
            active.int()
        )

        current_column_index = next_column_index

    # XOR: unassigned edges become assigned, assigned edges become unassigned
    assignment = assignment ^ (flip_matrix > 0)
    return assignment


def _maximum_bipartite_matching(
    adjacency_matrix: torch.Tensor,
    assignment: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Find maximum bipartite matching via augmenting paths.

    Args:
        adjacency_matrix: (B, N, M) boolean adjacency matrix.
        assignment: Optional initial assignment to warm-start from.

    Returns:
        Tuple of (state, assignment) where state is the final augmenting
        path search result and assignment is the maximum matching.
    """
    if assignment is None:
        assignment = _greedy_assignment(adjacency_matrix)

    state = _find_augmenting_path(assignment, adjacency_matrix)

    while state['new_columns'].any():
        assignment = _improve_assignment(assignment, state)
        state = _find_augmenting_path(assignment, adjacency_matrix)

    return state, assignment


def _compute_cover(
    state: dict[str, torch.Tensor],
    assignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute minimum vertex cover via König's theorem.

    The cover consists of:
        - Rows that are assigned AND NOT reachable from unassigned rows
        - Columns that are assigned AND reachable from unassigned rows

    Reference:
        https://en.wikipedia.org/wiki/König's_theorem_(graph_theory)#Proof

    Args:
        state: State dict from _find_augmenting_path.
        assignment: (B, N, M) boolean assignment matrix.

    Returns:
        Tuple of (rows_cover, columns_cover):
            rows_cover: (B, N, 1) boolean
            columns_cover: (B, 1, M) boolean
    """
    assigned_rows = assignment.any(dim=2, keepdim=True)  # (B, N, 1)
    assigned_columns = assignment.any(dim=1, keepdim=True)  # (B, 1, M)

    rows_cover = assigned_rows & (state['rows'] <= 0)  # (B, N, 1)
    columns_cover = assigned_columns & (state['columns'] > 0)  # (B, 1, M)

    return rows_cover, columns_cover


def _update_weights_using_cover(
    rows_cover: torch.Tensor,
    columns_cover: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Update weights for the next Hungarian iteration.

    Find the minimum uncovered weight. Subtract it from all uncovered
    elements. Add it to all doubly-covered elements. This creates new
    zeros while preserving existing ones under the cover.

    Args:
        rows_cover: (B, N, 1) boolean.
        columns_cover: (B, 1, M) boolean.
        weights: (B, N, M) float.

    Returns:
        Updated weights of the same shape.
    """
    max_value = weights.max()
    covered = rows_cover | columns_cover  # (B, N, M)
    double_covered = rows_cover & columns_cover  # (B, N, M)

    # Find minimum among uncovered elements
    uncovered_weights = weights.where(~covered, max_value)
    min_weight = uncovered_weights.flatten(1).min(dim=1).values  # (B,)
    min_weight = min_weight.reshape(-1, 1, 1)  # (B, 1, 1)

    # Add to doubly covered, subtract from uncovered
    addition = torch.where(double_covered, min_weight, torch.zeros_like(weights))
    subtraction = torch.where(~covered, min_weight, torch.zeros_like(weights))

    return weights + addition - subtraction


@torch.no_grad()
def hungarian_matcher(
    cost_matrix: torch.Tensor,
) -> torch.Tensor:
    """Solve batched linear assignment problems via scipy (CPU).

    Computes optimal 1-to-1 assignment minimizing total cost.
    Supports rectangular matrices. The cost matrix is transferred
    to CPU for scipy's C-optimized solver (~1ms per 280×280 matrix),
    then indices are returned as tensors on the original device.

    This is the same approach used by Facebook's DETR:
        https://github.com/facebookresearch/detr/blob/main/models/matcher.py

    Args:
        cost_matrix: (B, N, M) cost tensor on any device. Lower = better.

    Returns:
        indices: (B, 2, K) long tensor on the same device as cost_matrix,
            where K = min(N, M).
            indices[:, 0, :] = matched row indices
            indices[:, 1, :] = matched column indices
    """
    batch_size, num_rows, num_columns = cost_matrix.shape
    device = cost_matrix.device
    num_matched = min(num_rows, num_columns)

    cost_numpy = cost_matrix.detach().cpu().numpy()

    row_indices_np = np.empty((batch_size, num_matched), dtype=np.int64)
    col_indices_np = np.empty((batch_size, num_matched), dtype=np.int64)
    for batch_idx in range(batch_size):
        row_ind, col_ind = linear_sum_assignment(cost_numpy[batch_idx])
        row_indices_np[batch_idx] = row_ind[:num_matched]
        col_indices_np[batch_idx] = col_ind[:num_matched]

    row_indices = torch.from_numpy(row_indices_np).to(device)  # (B, K)
    col_indices = torch.from_numpy(col_indices_np).to(device)  # (B, K)

    return torch.stack([row_indices, col_indices], dim=1)  # (B, 2, K)


@torch.no_grad()
def hungarian_matcher_tensor(
    cost_matrix: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Solve batched linear assignment via pure PyTorch tensor ops.

    Fully on-device (no CPU transfers). Correct but currently slower
    than scipy for large matrices due to Python-level iteration overhead
    in the augmenting path search. Kept for future use when PyTorch/CUDA
    native while_loop support becomes available.

    Ported from Google Research's Scenic library (Apache 2.0):
        https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/hungarian_cover.py

    Args:
        cost_matrix: (B, N, M) cost tensor. Lower cost = better match.
        epsilon: Tolerance for zero-detection in the prepared cost matrix.

    Returns:
        indices: (B, 2, K) where K = min(N, M).
            indices[:, 0, :] = matched row indices
            indices[:, 1, :] = matched column indices
    """
    batch_size, num_rows, num_columns = cost_matrix.shape

    # Handle rectangular matrices by transposing if N > M
    should_transpose = num_rows > num_columns
    if should_transpose:
        cost_matrix = cost_matrix.transpose(1, 2)
        num_rows, num_columns = num_columns, num_rows

    # Pad to square if N < M (padding rows with high cost)
    if num_rows != num_columns:
        pad_rows = 1  # Scenic uses 1 padding row (sufficient for correctness)
        pad_values = (
            cost_matrix.flatten(1).max(dim=1).values.reshape(-1, 1, 1) * 1.1
        )
        pad_block = pad_values.expand(batch_size, pad_rows, num_columns)
        cost_matrix = torch.cat([cost_matrix, pad_block], dim=1)
        num_rows += pad_rows
    else:
        pad_rows = 0

    # Step 1: Prepare — subtract row and column minima
    weights = _prepare(cost_matrix)

    # Step 2: Find maximum matching on zero-adjacency graph
    adjacency_matrix = weights.abs() < epsilon
    state, assignment = _maximum_bipartite_matching(adjacency_matrix)
    rows_cover, columns_cover = _compute_cover(state, assignment)

    # Step 3-4: Iterate until cover size = N (optimal)
    cover_size = (
        rows_cover.sum(dtype=torch.int32)
        + columns_cover.sum(dtype=torch.int32)
    )
    target_cover = batch_size * num_rows

    while cover_size < target_cover:
        weights = _update_weights_using_cover(rows_cover, columns_cover, weights)
        adjacency_matrix = weights.abs() < epsilon
        state, assignment = _maximum_bipartite_matching(
            adjacency_matrix, assignment,
        )
        rows_cover, columns_cover = _compute_cover(state, assignment)
        cover_size = (
            rows_cover.sum(dtype=torch.int32)
            + columns_cover.sum(dtype=torch.int32)
        )

    # Extract indices from assignment matrix
    row_indices = torch.arange(num_rows, device=cost_matrix.device)
    row_indices = row_indices.unsqueeze(0).expand(batch_size, -1)  # (B, N)
    column_indices = assignment.long().argmax(dim=2)  # (B, N)

    # Remove padding rows
    row_indices = row_indices[:, :num_rows - pad_rows]
    column_indices = column_indices[:, :num_rows - pad_rows]

    # Undo transpose
    if should_transpose:
        indices = torch.stack([column_indices, row_indices], dim=1)
    else:
        indices = torch.stack([row_indices, column_indices], dim=1)

    return indices
