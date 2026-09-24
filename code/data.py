# Loads claim data and writes prediction rows in the required CSV format.
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from config import DATASET_DIR

INPUT_COLUMNS = ["user_id", "image_paths", "user_claim", "claim_object"]

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# Loads a claims CSV into row dictionaries.
def load_claims(csv_path: str | Path) -> list[dict]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return df.to_dict(orient="records")


# Loads claim history indexed by user ID.
def load_user_history() -> dict[str, dict]:
    path = DATASET_DIR / "user_history.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {row["user_id"]: row for row in df.to_dict(orient="records")}


# Loads general and object-specific evidence requirements.
def load_evidence_requirements() -> dict[str, list[dict]]:
    path = DATASET_DIR / "evidence_requirements.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = df.to_dict(orient="records")
    general = [r for r in rows if r["claim_object"] == "all"]
    by_object: dict[str, list[dict]] = {}
    for obj in ("car", "laptop", "package"):
        specific = [r for r in rows if r["claim_object"] == obj]
        by_object[obj] = specific + general
    return by_object


# Resolves submitted image paths to image IDs and absolute paths.
def resolve_images(image_paths_field: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in (image_paths_field or "").split(";"):
        rel = raw.strip()
        if not rel:
            continue
        abs_path = (DATASET_DIR / rel).resolve()
        out.append((abs_path.stem, abs_path))
    return out


# Writes prediction rows using the required column order and quoting.
def write_output(rows: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in OUTPUT_COLUMNS})
