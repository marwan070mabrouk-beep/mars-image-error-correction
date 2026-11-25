from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from hamming74 import decode as hamming_decode
from hamming74 import encode as hamming_encode
from ldpc import belief_propagation_decode, build_structure, decode_free_coordinates, encode as ldpc_encode

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "data" / "processed" / "PIA24428_gray_256x256.png"
RESULTS_DIR = ROOT / "results"


def load_bits() -> tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(IMAGE_PATH).convert("L"), dtype=np.uint8)
    bits = np.unpackbits(image.reshape(-1), bitorder="big").astype(np.uint8)
    return image, bits


def bits_to_image(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").reshape(shape)


def bsc(bits: np.ndarray, p: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    errors = (rng.random(bits.size) < p).astype(np.uint8)
    return bits ^ errors, errors


def encode_hamming_stream(bits: np.ndarray) -> tuple[np.ndarray, int, int]:
    k = 4
    padding = (-bits.size) % k
    payload = np.pad(bits, (0, padding))
    messages = payload.reshape(-1, k)
    codewords = np.stack([hamming_encode(m) for m in messages], axis=0)
    return codewords.reshape(-1), padding, messages.shape[0]


def decode_hamming_stream(received: np.ndarray, original_length: int) -> tuple[np.ndarray, int, int]:
    words = received.reshape(-1, 7)
    decoded = []
    corrections = 0
    for word in words:
        result = hamming_decode(word)
        decoded.append(result.message)
        corrections += int(result.correction_position is not None)
    payload = np.concatenate(decoded)[:original_length]
    return payload.astype(np.uint8), corrections, words.shape[0]


def encode_ldpc_stream(bits: np.ndarray, structure) -> tuple[np.ndarray, int, int]:
    k = structure.k
    padding = (-bits.size) % k
    payload = np.pad(bits, (0, padding))
    messages = payload.reshape(-1, k)
    codewords = np.stack([ldpc_encode(m, structure) for m in messages], axis=0)
    return codewords.reshape(-1), padding, messages.shape[0]


def decode_ldpc_stream(received: np.ndarray, original_length: int, structure, p: float, max_iterations: int = 50):
    words = received.reshape(-1, 1008)
    decoded = []
    nonconverged = 0
    total_iterations = 0
    unsatisfied_checks = 0
    for word in words:
        result = belief_propagation_decode(word, structure, p=p, max_iterations=max_iterations)
        decoded.append(result.message)
        nonconverged += int(not result.converged)
        total_iterations += result.iterations
        unsatisfied_checks += result.unsatisfied_checks
    payload = np.concatenate(decoded)[:original_length]
    return payload.astype(np.uint8), nonconverged, total_iterations, unsatisfied_checks, words.shape[0]


def block_errors(original: np.ndarray, recovered: np.ndarray, k: int) -> int:
    errors = 0
    for start in range(0, original.size, k):
        end = min(start + k, original.size)
        if np.any(original[start:end] != recovered[start:end]):
            errors += 1
    return errors


def image_metrics(reference: np.ndarray, recovered_bits: np.ndarray) -> dict[str, float | bool]:
    reconstructed = bits_to_image(recovered_bits, reference.shape)
    diff = reference.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255.0**2) / mse))
    ssim = float(structural_similarity(reference, reconstructed, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "exact_image": bool(np.array_equal(reference, reconstructed)),
    }


def run_trial(method: str, reference: np.ndarray, source_bits: np.ndarray, p: float, seed: int, structure=None) -> dict[str, object]:
    rng = np.random.Generator(np.random.PCG64(seed))
    start = time.perf_counter()
    if method == "uncoded":
        transmitted = source_bits.copy()
        padding = 0
        native_k = 1
        blocks = source_bits.size
        received, errors = bsc(transmitted, p, rng)
        recovered = received.copy()
        decoder_status = "none"
        nonconverged = 0
        iterations = 0
        unsatisfied_checks = 0
        corrections = 0
    elif method == "hamming":
        transmitted, padding, blocks = encode_hamming_stream(source_bits)
        native_k = 4
        received, errors = bsc(transmitted, p, rng)
        recovered, corrections, _ = decode_hamming_stream(received, source_bits.size)
        decoder_status = "syndrome"
        nonconverged = 0
        iterations = 0
        unsatisfied_checks = 0
    elif method == "ldpc":
        if structure is None:
            structure = build_structure()
        transmitted, padding, blocks = encode_ldpc_stream(source_bits, structure)
        native_k = structure.k
        received, errors = bsc(transmitted, p, rng)
        recovered, nonconverged, iterations, unsatisfied_checks, _ = decode_ldpc_stream(received, source_bits.size, structure, p)
        decoder_status = "sum-product"
        corrections = 0
    else:
        raise ValueError(method)
    decoder_time = time.perf_counter() - start

    metrics = image_metrics(reference, recovered)
    decoded_bit_errors = int(np.count_nonzero(source_bits != recovered))
    row = {
        "method": method,
        "p": p,
        "seed": seed,
        "source_bits": int(source_bits.size),
        "transmitted_bits": int(transmitted.size),
        "padding_bits": int(padding),
        "native_k": int(native_k),
        "blocks": int(blocks),
        "channel_flips": int(errors.sum()),
        "raw_ber": float(errors.mean()),
        "decoded_bit_errors": decoded_bit_errors,
        "decoded_ber": float(decoded_bit_errors / source_bits.size),
        "wrong_blocks": int(block_errors(source_bits, recovered, native_k)),
        "native_bler": float(block_errors(source_bits, recovered, native_k) / blocks),
        "image_failure": int(decoded_bit_errors > 0),
        "fer": float(decoded_bit_errors > 0),
        "mse": metrics["mse"],
        "psnr_db": metrics["psnr_db"],
        "ssim": metrics["ssim"],
        "exact_image": metrics["exact_image"],
        "decoder_status": decoder_status,
        "hamming_corrections": corrections,
        "ldpc_nonconverged_blocks": nonconverged,
        "ldpc_total_iterations": iterations,
        "ldpc_unsatisfied_checks": unsatisfied_checks,
        "decoder_time_seconds": decoder_time,
    }
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reference, source_bits = load_bits()
    structure = build_structure()
    methods = ["uncoded", "hamming", "ldpc"]
    rows = []
    for method_index, method in enumerate(methods):
        rows.append(run_trial(method, reference, source_bits, p=0.0, seed=1000 + method_index, structure=structure))
    output = RESULTS_DIR / "smoke_p0.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, allow_nan=True))
    print(f"Saved {output}")
    if not all(row["exact_image"] for row in rows):
        raise AssertionError("p=0 exact-reconstruction validation failed")
    print("End-to-end p=0 validation: PASS")


if __name__ == "__main__":
    main()
