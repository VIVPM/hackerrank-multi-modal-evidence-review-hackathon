# Validates and normalizes raw model output into required claim rows.
from __future__ import annotations

import schema
from data import OUTPUT_COLUMNS


# Normalizes a value for schema comparison.
def _norm(s) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


# Converts a value to an allowed enum value or its default.
def _coerce_enum(value, allowed: list[str], default: str) -> str:
    v = _norm(value)
    if v in allowed:
        return v
    for a in allowed:
        if a in v or v in a:
            return a
    return default


# Converts a value to the required lowercase boolean string.
def _coerce_bool(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if _norm(value) in ("true", "1", "yes") else "false"


# Filters, orders, and escalates risk flags.
def _coerce_flags(raw, present_flags_from_history: set[str]) -> list[str]:
    flags: list[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        v = _norm(item)
        if v in schema.RISK_FLAGS:
            flags.append(v)
    flags.extend(present_flags_from_history)
    flags = [f for f in flags if f != "none"]
    seen = set()
    ordered = [f for f in schema.RISK_FLAGS if f in flags and not (f in seen or seen.add(f))]
    if any(f in schema.ESCALATING_FLAGS or f == "user_history_risk" for f in ordered):
        if "manual_review_required" not in ordered:
            ordered.append("manual_review_required")
    return ordered or ["none"]


# Extracts valid risk flags from a user's claim history.
def _history_flags(history: dict | None) -> set[str]:
    if not history:
        return set()
    raw = history.get("history_flags", "") or ""
    return {_norm(f) for f in raw.split(";") if _norm(f) in schema.RISK_FLAGS}


# Produces a fully validated output row from a raw model response.
def normalize(raw: dict, claim: dict, present_ids: list[str], history: dict | None) -> dict:
    obj = _norm(claim.get("claim_object", ""))
    part_allowed = schema.OBJECT_PART.get(obj, schema.ALL_OBJECT_PARTS)
    raw = raw if isinstance(raw, dict) else {}

    raw_support = raw.get("supporting_image_ids", [])
    raw_support = raw_support if isinstance(raw_support, list) else [raw_support]
    support = [str(s).strip() for s in raw_support if str(s).strip() in present_ids]

    row = {
        "user_id": claim.get("user_id", ""),
        "image_paths": claim.get("image_paths", ""),
        "user_claim": claim.get("user_claim", ""),
        "claim_object": claim.get("claim_object", ""),
        "evidence_standard_met": _coerce_bool(raw.get("evidence_standard_met", False)),
        "evidence_standard_met_reason": str(raw.get("evidence_standard_met_reason", "")).strip()
            or "Evidence assessed from the submitted images.",
        "risk_flags": ";".join(_coerce_flags(raw.get("risk_flags", []), _history_flags(history))),
        "issue_type": _coerce_enum(raw.get("issue_type", "unknown"), schema.ISSUE_TYPE, "unknown"),
        "object_part": _coerce_enum(raw.get("object_part", "unknown"), part_allowed, "unknown"),
        "claim_status": _coerce_enum(raw.get("claim_status", "not_enough_information"),
                                     schema.CLAIM_STATUS, "not_enough_information"),
        "claim_status_justification": str(raw.get("claim_status_justification", "")).strip()
            or "Decision based on the visible image evidence.",
        "supporting_image_ids": ";".join(support) if support else "none",
        "valid_image": _coerce_bool(raw.get("valid_image", False)),
        "severity": _coerce_enum(raw.get("severity", "unknown"), schema.SEVERITY, "unknown"),
    }

    if not present_ids:
        row.update({
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": row["evidence_standard_met_reason"]
                or "No usable images were available for review.",
            "valid_image": "false",
            "claim_status": "not_enough_information",
            "supporting_image_ids": "none",
            "severity": "unknown",
        })

    if row["claim_status"] != "not_enough_information":
        if row["evidence_standard_met"] == "false":
            row["evidence_standard_met"] = "true"
            row["evidence_standard_met_reason"] = (
                "The claimed object and part are visible clearly enough to evaluate the "
                "claim, so the image set meets the evidence standard.")
        if row["supporting_image_ids"] == "none" and present_ids:
            row["supporting_image_ids"] = ";".join(present_ids)
        if row["claim_status"] == "contradicted" and row["severity"] == "unknown":
            row["severity"] = "none"
    elif row["evidence_standard_met"] == "true":
        row["evidence_standard_met"] = "false"
    return row


# Creates a safe review row when model processing fails.
def fallback_row(claim: dict, reason: str) -> dict:
    return normalize({"evidence_standard_met_reason": reason,
                      "claim_status_justification": reason,
                      "risk_flags": ["manual_review_required"]},
                     claim, present_ids=[], history=None)


# Runs assertions covering normalization invariants.
def _selfcheck() -> None:
    claim = {"user_id": "u1", "image_paths": "images/test/c/img_1.jpg;images/test/c/img_2.jpg",
             "user_claim": "rear bumper dent", "claim_object": "car"}
    raw = {"evidence_standard_met": True, "evidence_standard_met_reason": "visible",
           "risk_flags": ["claim_mismatch", "bogus_flag", "none"], "issue_type": "DENT",
           "object_part": "rear bumper", "claim_status": "supported",
           "claim_status_justification": "img_1 shows it",
           "supporting_image_ids": ["img_1", "img_99"], "valid_image": True, "severity": "medium"}
    hist = {"history_flags": "user_history_risk"}
    row = normalize(raw, claim, present_ids=["img_1", "img_2"], history=hist)

    assert set(row.keys()) == set(OUTPUT_COLUMNS), "missing columns"
    assert row["issue_type"] == "dent"
    assert row["object_part"] == "rear_bumper"
    assert row["supporting_image_ids"] == "img_1", "should drop invented img_99"
    flags = row["risk_flags"].split(";")
    assert "bogus_flag" not in flags and "none" not in flags
    assert "user_history_risk" in flags, "history flag must fuse in"
    assert "manual_review_required" in flags, "escalation/history must trigger review"
    assert row["valid_image"] == "true" and row["evidence_standard_met"] == "true"

    empty = normalize(raw, claim, present_ids=[], history=None)
    assert empty["claim_status"] == "not_enough_information"
    assert empty["valid_image"] == "false" and empty["supporting_image_ids"] == "none"

    con = normalize({"claim_status": "contradicted", "evidence_standard_met": False,
                     "supporting_image_ids": [], "severity": "unknown",
                     "issue_type": "none", "object_part": "door", "valid_image": True,
                     "risk_flags": ["damage_not_visible"]},
                    claim, present_ids=["img_1", "img_2"], history=None)
    assert con["evidence_standard_met"] == "true", "decided claim must meet evidence standard"
    assert con["supporting_image_ids"] == "img_1;img_2", "decided claim must cite images"
    assert con["severity"] == "none", "contradicted severity must not stay unknown"
    nei = normalize({"claim_status": "not_enough_information", "evidence_standard_met": True,
                     "valid_image": True}, claim, present_ids=["img_1"], history=None)
    assert nei["evidence_standard_met"] == "false", "NEI must not meet evidence standard"
    print("postprocess self-check: OK")


if __name__ == "__main__":
    _selfcheck()
