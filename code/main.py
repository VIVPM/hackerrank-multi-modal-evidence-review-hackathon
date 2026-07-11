"""Entry point: read claims.csv, run the VLM evidence-review pipeline, write output.csv.

Usage:
  python code/main.py --mock --limit 3          # offline dry run (no HF_TOKEN needed)
  python code/main.py --model cheap             # live run with the cheap model
  python code/main.py --model strong --output output.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import DATASET_DIR, REPO_ROOT, Settings
from client import VLMClient
from data import (load_claims, load_evidence_requirements, load_user_history,
                  resolve_images, write_output)
from images import load_claim_images
from postprocess import fallback_row, normalize
from prompts import build_messages


def process_claim(client: VLMClient, claim: dict, history_map: dict,
                  req_map: dict, settings: Settings) -> dict:
    """Run one claim end-to-end; never raises (failures become a safe fallback row)."""
    try:
        history = history_map.get(claim.get("user_id", ""))
        requirements = req_map.get(claim.get("claim_object", ""), [])
        pairs = resolve_images(claim.get("image_paths", ""))
        prepared, present_ids, missing_ids = load_claim_images(
            pairs, settings.max_image_px, settings.jpeg_quality)
        messages = build_messages(claim, history, requirements, prepared, present_ids, missing_ids)
        image_hashes = [p["sha256"] for p in prepared]
        raw = client.complete(messages, image_hashes,
                              mock_ctx={"claim": claim, "present_ids": present_ids})
        return normalize(raw, claim, present_ids, history)
    except Exception as e:  # one bad row must not sink the run
        return fallback_row(claim, f"Processing error: {e}")


def run(settings: Settings, input_csv: Path, output_csv: Path) -> None:
    claims = load_claims(input_csv)
    if settings.limit:
        claims = claims[:settings.limit]
    history_map = load_user_history()
    req_map = load_evidence_requirements()
    client = VLMClient(settings)

    mode = "MOCK" if settings.mock else f"model={settings.model_id} provider={settings.provider}"
    print(f"Processing {len(claims)} claims [{mode}] workers={settings.workers} ...")

    def work(claim):
        return process_claim(client, claim, history_map, req_map, settings)

    rows: list[dict] = [None] * len(claims)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, settings.workers)) as pool:
        for i, row in enumerate(pool.map(work, claims)):
            rows[i] = row
            print(f"  [{i + 1}/{len(claims)}] {row['user_id']} "
                  f"{row['claim_object']} -> {row['claim_status']} ({row['severity']})")

    write_output(rows, output_csv)
    print(f"Wrote {len(rows)} rows -> {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-modal evidence review pipeline")
    p.add_argument("--model", default="cheap", help="alias (cheap|strong) or HF model id")
    p.add_argument("--mock", action="store_true", help="offline rule-based stub (no API)")
    p.add_argument("--limit", type=int, default=None, help="process only first N claims")
    p.add_argument("--sleep", type=float, default=1.5, help="seconds between live calls")
    p.add_argument("--workers", type=int, default=3, help="concurrent workers")
    p.add_argument("--input", default=str(DATASET_DIR / "claims.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(model=args.model, mock=args.mock, limit=args.limit,
                        sleep=args.sleep, workers=args.workers)
    run(settings, Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
