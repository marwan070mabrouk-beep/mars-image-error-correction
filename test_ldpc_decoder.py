from __future__ import annotations

import numpy as np

from ldpc import belief_propagation_decode, build_structure, encode, syndrome


def main() -> None:
    structure = build_structure()
    rng = np.random.Generator(np.random.PCG64(424242))

    # No-noise validation: exact message recovery for several random blocks.
    for _ in range(5):
        message = rng.integers(0, 2, size=structure.k, dtype=np.uint8)
        codeword = encode(message, structure)
        result = belief_propagation_decode(codeword, structure, p=0.0, max_iterations=50)
        assert result.converged
        assert np.array_equal(result.message, message)
        assert np.array_equal(result.estimate, codeword)
        assert result.unsatisfied_checks == 0
    print("LDPC no-noise BP recovery (5 blocks): PASS")

    # Small pilot at p=0.01. This is an implementation check, not a paper result.
    trials = 5
    p = 0.01
    total_channel_flips = 0
    total_residual_bits = 0
    nonconverged = 0
    for trial in range(trials):
        message = rng.integers(0, 2, size=structure.k, dtype=np.uint8)
        codeword = encode(message, structure)
        errors = (rng.random(codeword.size) < p).astype(np.uint8)
        received = codeword ^ errors
        result = belief_propagation_decode(received, structure, p=p, max_iterations=50)
        total_channel_flips += int(errors.sum())
        total_residual_bits += int(np.count_nonzero(result.message != message))
        nonconverged += int(not result.converged)
        assert np.count_nonzero(syndrome(structure.H, result.estimate)) == result.unsatisfied_checks
    print(f"LDPC pilot: trials={trials}, p={p}")
    print(f"  channel flips: {total_channel_flips}")
    print(f"  residual message-bit errors: {total_residual_bits}")
    print(f"  nonconverged blocks: {nonconverged}")
    print("LDPC pilot structural checks: PASS")


if __name__ == "__main__":
    main()
