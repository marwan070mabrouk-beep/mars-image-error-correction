from __future__ import annotations

import hashlib

import numpy as np

from ldpc import build_structure, decode_free_coordinates, encode, save_structure, syndrome

ROOT = __import__("pathlib").Path(__file__).resolve().parent
EXPECTED_HASH_FROM_MANUSCRIPT = "7f99c987293300245ee3d28c5a81c389968b3d68b3bd104a516ce2f0df4c398e"


def main() -> None:
    structure = build_structure()
    assert structure.H.shape == (504, 1008)
    assert int(structure.H.sum()) == 3024
    assert structure.rank == 502
    assert structure.k == 506
    assert structure.free_columns.size == 506
    assert np.all(structure.H.sum(axis=1) == 6)
    assert np.all(structure.H.sum(axis=0) == 3)

    print("H shape, edge count, row degree, column degree: PASS")
    print(f"GF(2) rank: {structure.rank} (expected 502)")
    print(f"k: {structure.k} (expected 506)")
    print(f"rate: {structure.k / 1008:.9f}")
    print(f"matrix SHA-256: {structure.matrix_hash}")
    if structure.matrix_hash != EXPECTED_HASH_FROM_MANUSCRIPT:
        raise AssertionError("Constructed matrix hash differs from the manuscript record")
    print("Matrix hash matches the manuscript: PASS")

    rng = np.random.Generator(np.random.PCG64(12345))
    for index in range(20):
        message = rng.integers(0, 2, size=structure.k, dtype=np.uint8)
        codeword = encode(message, structure)
        assert np.all(syndrome(structure.H, codeword) == 0)
        recovered = decode_free_coordinates(codeword, structure)
        assert np.array_equal(recovered, message)
    print("20 noiseless encode/parity/message-recovery trials: PASS")

    metadata = save_structure(structure, ROOT / "metadata")
    print("Saved structure metadata:")
    print(metadata)


if __name__ == "__main__":
    main()
