from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N = 1008
M = 504
M0 = 168
ROW_WEIGHT = 6
COLUMN_WEIGHT = 3
MATRIX_SEED = 20261124


@dataclass(frozen=True)
class LDPCStructure:
    H: np.ndarray
    rref: np.ndarray
    pivot_columns: np.ndarray
    free_columns: np.ndarray
    rank: int
    k: int
    matrix_hash: str


def matrix_sha256(H: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(H, dtype=np.uint8).tobytes(order="C")).hexdigest()


def construct_matrix(seed: int = MATRIX_SEED) -> np.ndarray:
    """Construct the exact matrix described in Appendix A of the manuscript."""
    rng = np.random.Generator(np.random.PCG64(seed))
    B0 = np.zeros((M0, N), dtype=np.uint8)
    for row in range(M0):
        B0[row, 6 * row : 6 * row + 6] = 1
    p1 = rng.permutation(N)
    p2 = rng.permutation(N)
    H = np.vstack([B0, B0[:, p1], B0[:, p2]])
    return H.astype(np.uint8, copy=False)


def gf2_rref(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return RREF and pivot columns over GF(2)."""
    a = np.asarray(A, dtype=np.uint8).copy() % 2
    rows, cols = a.shape
    pivots: list[int] = []
    pivot_row = 0
    for col in range(cols):
        candidates = np.flatnonzero(a[pivot_row:, col])
        if candidates.size == 0:
            continue
        row = pivot_row + int(candidates[0])
        if row != pivot_row:
            a[[pivot_row, row]] = a[[row, pivot_row]]
        other_rows = np.flatnonzero(a[:, col])
        other_rows = other_rows[other_rows != pivot_row]
        if other_rows.size:
            a[other_rows] ^= a[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break
    return a, np.asarray(pivots, dtype=np.int64)


def build_structure(seed: int = MATRIX_SEED) -> LDPCStructure:
    H = construct_matrix(seed)
    rref, pivots = gf2_rref(H)
    rank = int(pivots.size)
    pivot_set = set(int(x) for x in pivots)
    free = np.asarray([j for j in range(H.shape[1]) if j not in pivot_set], dtype=np.int64)
    k = H.shape[1] - rank
    if k != free.size:
        raise AssertionError("Free-column count does not equal null-space dimension")
    return LDPCStructure(
        H=H,
        rref=rref,
        pivot_columns=pivots,
        free_columns=free,
        rank=rank,
        k=k,
        matrix_hash=matrix_sha256(H),
    )


def syndrome(H: np.ndarray, word: np.ndarray) -> np.ndarray:
    word = np.asarray(word, dtype=np.uint8).reshape(-1)
    return ((H @ word) % 2).astype(np.uint8)


def encode(message: np.ndarray, structure: LDPCStructure) -> np.ndarray:
    """Encode k free-coordinate message bits into a length-n codeword.

    The message occupies the non-pivot coordinates in increasing original-column
    order. RREF equations then determine the pivot coordinates.
    """
    m = np.asarray(message, dtype=np.uint8).reshape(-1)
    if m.size != structure.k or not np.all((m == 0) | (m == 1)):
        raise ValueError(f"Expected {structure.k} binary message bits")
    c = np.zeros(structure.H.shape[1], dtype=np.uint8)
    c[structure.free_columns] = m
    for row, pivot in enumerate(structure.pivot_columns):
        # RREF equation: x_pivot + sum(R[row, free] x_free) = 0.
        c[pivot] = np.dot(structure.rref[row, structure.free_columns], m) % 2
    return c.astype(np.uint8)


def decode_free_coordinates(codeword: np.ndarray, structure: LDPCStructure) -> np.ndarray:
    codeword = np.asarray(codeword, dtype=np.uint8).reshape(-1)
    return codeword[structure.free_columns].copy()


def save_structure(structure: LDPCStructure, root: Path) -> dict[str, object]:
    import hashlib
    import json

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "ldpc_H_seed20261124.npy", structure.H)
    metadata = {
        "n": N,
        "m": M,
        "rank_gf2": structure.rank,
        "k": structure.k,
        "rate": structure.k / N,
        "matrix_seed": MATRIX_SEED,
        "matrix_sha256": structure.matrix_hash,
        "matrix_shape": list(structure.H.shape),
        "ones": int(structure.H.sum()),
        "density": float(structure.H.mean()),
        "pivot_columns_file": "ldpc_pivot_columns.npy",
        "free_columns_file": "ldpc_free_columns.npy",
    }
    np.save(root / "ldpc_pivot_columns.npy", structure.pivot_columns)
    np.save(root / "ldpc_free_columns.npy", structure.free_columns)
    (root / "ldpc_structure.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


@dataclass(frozen=True)
class BPDecodeResult:
    estimate: np.ndarray
    message: np.ndarray
    converged: bool
    iterations: int
    unsatisfied_checks: int


def _adjacency(H: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    check_neighbors = [np.flatnonzero(H[a]).astype(np.int64) for a in range(H.shape[0])]
    variable_neighbors = [np.flatnonzero(H[:, j]).astype(np.int64) for j in range(H.shape[1])]
    return check_neighbors, variable_neighbors


_ADJACENCY_CACHE: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
_EDGE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _edge_indices(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return check-order variable indices and variable-order edge indices."""
    edge_variables = np.nonzero(H)[1].astype(np.int64)
    variable_edges = [[] for _ in range(H.shape[1])]
    for edge, variable in enumerate(edge_variables):
        variable_edges[int(variable)].append(edge)
    variable_edge_indices = np.asarray(variable_edges, dtype=np.int64)
    return edge_variables, variable_edge_indices


def belief_propagation_decode(
    received: np.ndarray,
    structure: LDPCStructure,
    p: float,
    max_iterations: int = 50,
    llr_clip: float = 30.0,
) -> BPDecodeResult:
    """Decode a hard BSC observation using flooding log-domain sum-product BP."""
    r = np.asarray(received, dtype=np.uint8).reshape(-1)
    if r.size != structure.H.shape[1] or not np.all((r == 0) | (r == 1)):
        raise ValueError(f"Expected {structure.H.shape[1]} binary received bits")
    if not (0 <= p < 0.5):
        raise ValueError("BSC crossover p must satisfy 0 <= p < 0.5")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    if p == 0:
        unsatisfied = int(np.count_nonzero(syndrome(structure.H, r)))
        return BPDecodeResult(
            estimate=r.copy(),
            message=decode_free_coordinates(r, structure),
            converged=(unsatisfied == 0),
            iterations=0,
            unsatisfied_checks=unsatisfied,
        )

    cache_key = id(structure.H)
    if cache_key not in _EDGE_CACHE:
        _EDGE_CACHE[cache_key] = _edge_indices(structure.H)
    edge_variables, variable_edge_indices = _EDGE_CACHE[cache_key]
    channel_weight = min(float(np.log((1.0 - p) / p)), llr_clip)
    llr = np.where(r == 0, channel_weight, -channel_weight).astype(float)

    m_checks = structure.H.shape[0]
    n_bits = structure.H.shape[1]
    check_degree = int(structure.H.sum(axis=1)[0])
    q = llr[edge_variables].copy()

    estimate = r.copy()
    for iteration in range(1, max_iterations + 1):
        # Check-to-variable updates in check-edge order.
        q_by_check = q.reshape(m_checks, check_degree)
        tanh_values = np.tanh(q_by_check / 2.0)
        product_excluding = np.ones_like(tanh_values)
        for local_index in range(check_degree):
            product_excluding[:, local_index] = np.prod(
                np.delete(tanh_values, local_index, axis=1), axis=1
            )
        product_excluding = np.clip(product_excluding, -1.0 + 1e-12, 1.0 - 1e-12)
        u = 2.0 * np.arctanh(product_excluding).reshape(-1)

        # Variable-to-check updates and posterior hard decision.
        posterior = llr + u[variable_edge_indices].sum(axis=1)
        q = posterior[edge_variables] - u
        estimate = (posterior < 0).astype(np.uint8)
        check_parities = np.bitwise_xor.reduce(
            estimate[edge_variables].reshape(m_checks, check_degree), axis=1
        )
        unsatisfied = int(np.count_nonzero(check_parities))
        if unsatisfied == 0:
            return BPDecodeResult(
                estimate=estimate,
                message=decode_free_coordinates(estimate, structure),
                converged=True,
                iterations=iteration,
                unsatisfied_checks=0,
            )

    unsatisfied = int(np.count_nonzero(syndrome(structure.H, estimate)))
    return BPDecodeResult(
        estimate=estimate,
        message=decode_free_coordinates(estimate, structure),
        converged=False,
        iterations=max_iterations,
        unsatisfied_checks=unsatisfied,
    )
