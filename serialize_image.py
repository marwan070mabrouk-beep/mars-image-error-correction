from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "data" / "processed" / "PIA24428_gray_256x256.png"
BITSTREAM = ROOT / "data" / "processed" / "PIA24428_bits_msb_first.npy"
ROUNDTRIP = ROOT / "data" / "processed" / "PIA24428_roundtrip.png"
METADATA = ROOT / "metadata" / "bitstream_serialization.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_to_bits(image: np.ndarray) -> np.ndarray:
    """Serialize uint8 grayscale pixels row-major, eight bits MSB-first."""
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError(f"Expected 2D uint8 image, got dtype={image.dtype}, ndim={image.ndim}")
    return np.unpackbits(image.reshape(-1), bitorder="big").astype(np.uint8)


def bits_to_image(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reconstruct a uint8 grayscale image from row-major MSB-first bits."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    expected = shape[0] * shape[1] * 8
    if bits.size != expected:
        raise ValueError(f"Expected {expected} bits, got {bits.size}")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Bitstream contains values other than 0 and 1")
    return np.packbits(bits, bitorder="big").reshape(shape)


def main() -> None:
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    BITSTREAM.parent.mkdir(parents=True, exist_ok=True)

    original = np.array(Image.open(IMAGE).convert("L"), dtype=np.uint8)
    bits = image_to_bits(original)
    reconstructed = bits_to_image(bits, original.shape)
    Image.fromarray(reconstructed, mode="L").save(ROUNDTRIP, format="PNG", optimize=False)
    np.save(BITSTREAM, bits)

    if not np.array_equal(original, reconstructed):
        raise AssertionError("Image-to-bits-to-image round trip failed")

    metadata = {
        "source_image": IMAGE.name,
        "source_image_sha256": sha256(IMAGE),
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "scan_order": "row-major, top-to-bottom and left-to-right",
        "bit_order": "MSB-first within each 8-bit pixel",
        "bits_per_pixel": 8,
        "bit_length": int(bits.size),
        "expected_bit_length": int(original.shape[0] * original.shape[1] * 8),
        "first_pixel_value": int(original[0, 0]),
        "first_pixel_bits": bits[:8].tolist(),
        "last_pixel_value": int(original[-1, -1]),
        "last_pixel_bits": bits[-8:].tolist(),
        "bitstream_file": BITSTREAM.name,
        "roundtrip_image": ROUNDTRIP.name,
        "roundtrip_exact": bool(np.array_equal(original, reconstructed)),
        "roundtrip_max_absolute_pixel_error": int(np.max(np.abs(original.astype(int) - reconstructed.astype(int)))),
        "roundtrip_image_sha256": sha256(ROUNDTRIP),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
