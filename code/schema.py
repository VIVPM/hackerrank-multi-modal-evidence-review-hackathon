"""Canonical allowed values (from problem_statement.md). Imported by prompts + postprocess."""
from __future__ import annotations

CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
]

OBJECT_PART = {
    "car": ["front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
            "headlight", "taillight", "fender", "quarter_panel", "body", "unknown"],
    "laptop": ["screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
               "base", "body", "unknown"],
    "package": ["box", "package_corner", "package_side", "seal", "label", "contents",
                "item", "unknown"],
}

RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]

SEVERITY = ["none", "low", "medium", "high", "unknown"]

# Risk flags that should also trigger manual_review_required (postprocess rule).
ESCALATING_FLAGS = {"claim_mismatch", "possible_manipulation", "non_original_image", "wrong_object"}

ALL_OBJECT_PARTS = sorted({p for parts in OBJECT_PART.values() for p in parts})
