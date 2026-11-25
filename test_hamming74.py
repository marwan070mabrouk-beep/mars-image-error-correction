from __future__ import annotations

from itertools import product

import numpy as np

from hamming74 import G, H, decode, encode, flip_positions, syndrome


def bits(text: str) -> np.ndarray:
    return np.array([int(ch) for ch in text], dtype=np.uint8)


def main() -> None:
    assert np.array_equal(G @ H.T % 2, np.zeros((4, 3), dtype=np.uint8))

    message_count = 0
    single_error_count = 0
    for message_tuple in product([0, 1], repeat=4):
        message = np.array(message_tuple, dtype=np.uint8)
        codeword = encode(message)
        assert np.array_equal(codeword[:4], message)
        assert np.array_equal(syndrome(codeword), np.zeros(3, dtype=np.uint8))

        no_error = decode(codeword)
        assert np.array_equal(no_error.message, message)
        assert np.array_equal(no_error.corrected, codeword)
        assert no_error.correction_position is None
        message_count += 1

        for position in range(1, 8):
            received = flip_positions(codeword, [position])
            result = decode(received)
            assert np.array_equal(result.message, message), (message, position, result)
            assert np.array_equal(result.corrected, codeword), (message, position, result)
            assert result.correction_position == position
            single_error_count += 1

    # The manuscript's worked example: 1011 -> 1011010.
    example_message = bits("1011")
    example_codeword = encode(example_message)
    assert np.array_equal(example_codeword, bits("1011010"))
    assert np.array_equal(syndrome(bits("1001010")), bits("011"))
    example_result = decode(bits("1001010"))
    assert np.array_equal(example_result.message, example_message)
    assert example_result.correction_position == 3

    # Two errors at positions 3 and 6 cause the documented miscorrection.
    double_received = flip_positions(example_codeword, [3, 6])
    assert np.array_equal(double_received, bits("1001000"))
    double_result = decode(double_received)
    assert np.array_equal(double_result.syndrome, bits("001"))
    assert double_result.correction_position == 7
    assert np.array_equal(double_result.corrected, bits("1001001"))
    assert np.array_equal(double_result.message, bits("1001"))
    assert not np.array_equal(double_result.message, example_message)

    print("Hamming matrix check: PASS")
    print(f"All {message_count} messages: PASS")
    print(f"All {single_error_count} single-bit cases: PASS")
    print("Worked single-error example: PASS")
    print("Documented double-error miscorrection: PASS")
    print("G=")
    print(G)
    print("H=")
    print(H)


if __name__ == "__main__":
    main()
