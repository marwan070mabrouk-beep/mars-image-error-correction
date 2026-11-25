from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "source" / "PIA24428_original.png"
OUTPUT = ROOT / "data" / "processed" / "PIA24428_gray_256x256.png"
METADATA = ROOT / "metadata" / "image_preprocessing.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METADATA.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(SOURCE) as im:
        source_format = im.format
        source_mode = im.mode
        source_size = list(im.size)

        # Largest centered square crop: 1536 x 1536 from the 2048 x 1536 source.
        side = min(im.size)
        left = (im.width - side) // 2
        top = (im.height - side) // 2
        right = left + side
        bottom = top + side
        crop_box = [left, top, right, bottom]

        gray = im.crop((left, top, right, bottom)).convert("L")
        reference = gray.resize((256, 256), resample=Image.Resampling.LANCZOS)
        reference.save(OUTPUT, format="PNG", optimize=False)

    metadata = {
        "source": {
            "identifier": "NASA/JPL-Caltech PIA24428",
            "download_url": "https://assets.science.nasa.gov/content/dam/science/psd/mars/downloadable_items/4/5/45686_1-PIA24428-High-Resolution_Still_Image_of_Perseverances_Landing.png",
            "filename": SOURCE.name,
            "sha256": sha256(SOURCE),
            "format": source_format,
            "mode": source_mode,
            "size": source_size,
        },
        "preprocessing": {
            "crop_rule": "largest centered square crop",
            "crop_box_xyxy": crop_box,
            "crop_size": [side, side],
            "grayscale": "Pillow Image.convert('L')",
            "resize_size": [256, 256],
            "resize_filter": "Lanczos",
            "output_format": "PNG",
            "output_mode": "L",
            "output_filename": OUTPUT.name,
            "output_sha256": sha256(OUTPUT),
        },
        "software": {
            "python": platform.python_version(),
            "pillow": PILLOW_VERSION,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with Image.open(OUTPUT) as out:
        print(json.dumps({
            "output": str(OUTPUT),
            "format": out.format,
            "mode": out.mode,
            "size": out.size,
            "sha256": metadata["preprocessing"]["output_sha256"],
            "crop_box_xyxy": crop_box,
            "metadata": str(METADATA),
        }, indent=2))


if __name__ == "__main__":
    main()
