from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The parity-last convention used in the manuscript.
P = np.array(
    [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.uint8,
)
G = np.concatenate([np.eye(4, dtype=np.uint8), P], axis=1)
H = np.concatenate([P.T, np.eye(3, dtype=np.uint8)], axis=1)


@dataclass(frozen=True)
class DecodeResult:
    received: np.ndarray
    syndrome: np.ndarray
    corrected: np.ndarray
    message: np.ndarray
    correction_position: int | None


def _bits(values: np.ndarray | list[int], length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    if arr.size != length or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"Expected {length} binary values, got {arr.tolist()}")
    return arr


def encode(message: np.ndarray | list[int]) -> np.ndarray:
    """Encode a 4-bit row message as [m1,m2,m3,m4,p1,p2,p3]."""
    m = _bits(message, 4)
    return ((m @ G) % 2).astype(np.uint8)


def syndrome(word: np.ndarray | list[int]) -> np.ndarray:
    """Compute the 3-bit syndrome s = word H^T over GF(2)."""
    r = _bits(word, 7)
    return ((r @ H.T) % 2).astype(np.uint8)


def decode(word: np.ndarray | list[int]) -> DecodeResult:
    """Decode by zero-syndrome retention or matching a nonzero H column.

    A nonzero syndrome identifies a single-bit error for this Hamming code.
    For multiple errors, the procedure may miscorrect to another valid codeword;
    this behavior is intentionally retained and can be inspected by the caller.
    """
    r = _bits(word, 7)
    s = syndrome(r)
    corrected = r.copy()
    correction_position: int | None = None

    if np.any(s):
        matches = np.flatnonzero(np.all(H.T == s, axis=1))
        if matches.size != 1:
            raise RuntimeError(f"Syndrome {s.tolist()} does not match exactly one H column")
        correction_position = int(matches[0]) + 1  # Report one-based position.
        corrected[matches[0]] ^= 1

    return DecodeResult(
        received=r,
        syndrome=s,
        corrected=corrected,
        message=corrected[:4].copy(),
        correction_position=correction_position,
    )


def flip_positions(word: np.ndarray | list[int], positions_one_based: list[int]) -> np.ndarray:
    """Return a copy with the requested one-based bit positions flipped."""
    out = _bits(word, 7).copy()
    for pos in positions_one_based:
        if not 1 <= pos <= 7:
            raise ValueError(f"Position must be in 1..7, got {pos}")
        out[pos - 1] ^= 1
    return out
