"""Scoring helpers: per-field accuracy + set-based risk_flags P/R/F1 + claim_status confusion."""
from __future__ import annotations

# Categorical/boolean fields scored by exact match (justification text is not scored).
CATEGORICAL = ["claim_status", "evidence_standard_met", "valid_image",
               "issue_type", "object_part", "severity"]


def _n(v) -> str:
    return str(v).strip().lower()


def _flagset(v) -> set[str]:
    return {x.strip().lower() for x in str(v or "none").split(";") if x.strip()}


def score(preds: list[dict], golds: list[dict]) -> dict:
    """Return metrics comparing prediction rows to gold rows (matched by position)."""
    n = len(golds)
    assert len(preds) == n, "pred/gold length mismatch"
    out: dict = {"n": n}

    for f in CATEGORICAL:
        correct = sum(1 for p, g in zip(preds, golds) if _n(p.get(f)) == _n(g.get(f)))
        out[f + "_acc"] = correct / n if n else 0.0

    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        ps, gs = _flagset(p.get("risk_flags")), _flagset(g.get("risk_flags"))
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    out["risk_flags_precision"] = prec
    out["risk_flags_recall"] = rec
    out["risk_flags_f1"] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # claim_status confusion: gold -> {pred: count}
    labels = ["supported", "contradicted", "not_enough_information"]
    conf = {gl: {pl: 0 for pl in labels} for gl in labels}
    for p, g in zip(preds, golds):
        gl, pl = _n(g.get("claim_status")), _n(p.get("claim_status"))
        if gl in conf and pl in conf[gl]:
            conf[gl][pl] += 1
    out["claim_status_confusion"] = conf
    return out
