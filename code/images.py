# Prepares claim images for multimodal model requests.
from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image

from data import IMAGE_EXTS


# Converts one image into a resized JPEG data URI with a content hash.
def prepare_image(path: Path, max_px: int = 1024, quality: int = 85) -> dict | None:
    if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            long_edge = max(im.size)
            if long_edge > max_px:
                scale = max_px / long_edge
                im = im.resize((max(1, round(im.width * scale)),
                                max(1, round(im.height * scale))))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
    except Exception:
        return None
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "image_id": path.stem,
        "data_uri": f"data:image/jpeg;base64,{b64}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


# Prepares all claim images and separates usable and unusable IDs.
def load_claim_images(image_pairs: list[tuple[str, Path]], max_px: int = 1024,
                      quality: int = 85) -> tuple[list[dict], list[str], list[str]]:
    prepared, present_ids, missing_ids = [], [], []
    for image_id, path in image_pairs:
        info = prepare_image(path, max_px, quality)
        if info is None:
            missing_ids.append(image_id)
        else:
            prepared.append(info)
            present_ids.append(info["image_id"])
    return prepared, present_ids, missing_ids
