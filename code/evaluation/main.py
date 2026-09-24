# Evaluates pipeline predictions against the labeled sample claims.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from config import DATASET_DIR, Settings
from client import VLMClient
from data import (load_claims, load_evidence_requirements, load_user_history,
                  resolve_images, write_output)
from main import process_claim
from prompts import SYSTEM_PROMPT
from metrics import score

PRICES = {
    "cheap": {"in": 0.10, "out": 0.30},
    "strong": {"in": 0.40, "out": 1.20},
}
IMG_TOKENS = 1000
OUT_TOKENS = 200
TEST_CLAIMS = 44


# Runs one model configuration over the supplied claims.
def run_model(alias: str, claims: list[dict], history_map: dict, req_map: dict,
              args) -> dict:
    settings = Settings(model=alias, mock=args.mock, sleep=args.sleep, workers=args.workers)
    client = VLMClient(settings)
    t0 = time.time()
    preds = [process_claim(client, c, history_map, req_map, settings) for c in claims]
    elapsed = time.time() - t0
    write_output(preds, CODE_DIR / ".cache" / f"eval_preds_{alias}.csv")
    return {"alias": alias, "model_id": settings.model_id, "preds": preds,
            "elapsed": elapsed, "api_calls": client.api_calls, "cache_hits": client.cache_hits}


# Estimates token use, cost, and per-claim latency for a model run.
def estimate_ops(alias: str, claims: list[dict], n_images: int, elapsed: float) -> dict:
    text_tokens = sum((len(SYSTEM_PROMPT) + len(c.get("user_claim", "")) + 500) / 4 for c in claims)
    in_tokens = text_tokens + n_images * IMG_TOKENS
    out_tokens = len(claims) * OUT_TOKENS
    price = PRICES.get(alias, PRICES["cheap"])
    cost = in_tokens / 1e6 * price["in"] + out_tokens / 1e6 * price["out"]
    per_claim_cost = cost / len(claims) if claims else 0.0
    return {
        "in_tokens": int(in_tokens), "out_tokens": int(out_tokens),
        "sample_cost": cost, "per_claim_cost": per_claim_cost,
        "test_cost_projection": per_claim_cost * TEST_CLAIMS,
        "per_claim_latency": elapsed / len(claims) if claims else 0.0,
    }


# Formats a decimal as a percentage.
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# Builds a Markdown evaluation report from model results.
def build_report(results: list[dict], n_claims: int, n_images: int,
                 best_alias: str, is_mock: bool) -> str:
    L = []
    L.append("# Evaluation Report — Multi-Modal Evidence Review\n")
    if is_mock:
        L.append("> NOTE: generated in **--mock** mode (no model calls). Numbers are for "
                 "harness validation only; re-run live for real metrics.\n")
    L.append(f"- Dataset: `dataset/sample_claims.csv` — **{n_claims} labeled claims**, "
             f"**{n_images} images**.\n- Strategy: single VLM call per claim, "
             "claim-restate-then-inspect prompt, deterministic history/rule fusion.\n"
             "- Comparison (Option A): cheap vs strong model, identical prompt.\n")

    L.append("\n## Accuracy on the labeled sample\n")
    fields = ["claim_status_acc", "evidence_standard_met_acc", "valid_image_acc",
              "issue_type_acc", "object_part_acc", "severity_acc",
              "risk_flags_precision", "risk_flags_recall", "risk_flags_f1"]
    header = "| metric | " + " | ".join(r["alias"] for r in results) + " |"
    sep = "|---|" + "|".join("---" for _ in results) + "|"
    L.append(header)
    L.append(sep)
    for f in fields:
        cells = " | ".join(_pct(r["metrics"][f]) for r in results)
        L.append(f"| {f} | {cells} |")

    for r in results:
        L.append(f"\n## Model: `{r['model_id']}` (alias `{r['alias']}`)\n")
        conf = r["metrics"]["claim_status_confusion"]
        L.append("claim_status confusion (rows = gold, cols = predicted):\n")
        labels = ["supported", "contradicted", "not_enough_information"]
        L.append("| gold \\ pred | " + " | ".join(labels) + " |")
        L.append("|---|" + "|".join("---" for _ in labels) + "|")
        for gl in labels:
            L.append(f"| {gl} | " + " | ".join(str(conf[gl][pl]) for pl in labels) + " |")
        o = r["ops"]
        L.append("\n**Operational analysis (this sample run):**\n")
        L.append(f"- model calls (cache misses): {r['api_calls']} ; cache hits: {r['cache_hits']}")
        L.append(f"- images processed: {n_images}")
        L.append(f"- est. input tokens: ~{o['in_tokens']:,} ; est. output tokens: ~{o['out_tokens']:,}")
        L.append(f"- est. sample cost: ~${o['sample_cost']:.4f} "
                 f"(~${o['per_claim_cost']:.4f}/claim)")
        L.append(f"- **projected cost for full test set ({TEST_CLAIMS} claims): "
                 f"~${o['test_cost_projection']:.3f}**")
        L.append(f"- measured runtime: {r['elapsed']:.1f}s "
                 f"(~{o['per_claim_latency']:.2f}s/claim)")

    L.append("\n## Final strategy & recommendation\n")
    best = next(r for r in results if r["alias"] == best_alias)
    L.append(f"Selected **`{best['model_id']}`** (alias `{best_alias}`) for `output.csv`: "
             f"highest claim_status accuracy "
             f"({_pct(best['metrics']['claim_status_acc'])}) at acceptable cost/latency.\n")
    L.append("## Pricing assumptions & rate-limit strategy\n")
    L.append(f"- Prices are **illustrative** (USD/1M tokens): "
             + "; ".join(f"{k}=in ${v['in']}/out ${v['out']}" for k, v in PRICES.items())
             + ". Replace with your HF provider's published rates.\n"
             f"- Image tokens assumed ~{IMG_TOKENS}/image at ~1024px; output ~{OUT_TOKENS}/claim.\n"
             "- TPM/RPM: bounded `--workers` + `--sleep` between live calls + exponential "
             "backoff w/ jitter on 429/5xx. Content-addressed cache avoids repeat calls "
             "across reruns. HF chat endpoint has no native batch API, so throughput = "
             "concurrency + cache.\n")
    return "\n".join(L) + "\n"


# Parses options, evaluates models, and writes the report.
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the pipeline on the labeled sample set")
    p.add_argument("--models", default="cheap,strong", help="comma list of aliases/model ids")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--report", default=str(Path(__file__).resolve().parent / "evaluation_report.md"))
    args = p.parse_args()

    claims = load_claims(DATASET_DIR / "sample_claims.csv")
    if args.limit:
        claims = claims[:args.limit]
    history_map = load_user_history()
    req_map = load_evidence_requirements()
    n_images = sum(len(resolve_images(c.get("image_paths", ""))) for c in claims)

    aliases = [a.strip() for a in args.models.split(",") if a.strip()]
    results = []
    for alias in aliases:
        print(f"=== Evaluating model alias '{alias}' on {len(claims)} sample claims ===")
        r = run_model(alias, claims, history_map, req_map, args)
        r["metrics"] = score(r["preds"], claims)
        r["ops"] = estimate_ops(alias, claims, n_images, r["elapsed"])
        print(f"  claim_status accuracy: {_pct(r['metrics']['claim_status_acc'])}")
        results.append(r)

    best = max(results, key=lambda r: (r["metrics"]["claim_status_acc"],
                                       r["metrics"]["risk_flags_f1"]))
    report = build_report(results, len(claims), n_images, best["alias"], args.mock)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"Wrote report -> {args.report}")


if __name__ == "__main__":
    main()
