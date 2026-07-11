"""VLM client: HF Inference Providers call, content-addressed cache, mock mode, retry."""
from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

from config import Settings
from prompts import output_json_schema

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> dict:
    """Extract a JSON object from model output (tolerates code fences / surrounding prose)."""
    if not text:
        raise ValueError("empty response")
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


# --- mock -----------------------------------------------------------------

_ISSUE_KEYWORDS = [
    ("glass_shatter", ["shatter"]), ("crack", ["crack", "cracked"]),
    ("dent", ["dent", "dented"]), ("scratch", ["scratch", "scrape", "scraped"]),
    ("missing_part", ["missing", "faltan", "came off", "keycap"]),
    ("torn_packaging", ["torn", "phati", "open"]), ("crushed_packaging", ["crush", "dab"]),
    ("water_damage", ["water", "wet"]), ("stain", ["stain", "oil"]),
    ("broken_part", ["broke", "broken", "toot"]),
]
_PART_KEYWORDS = {
    "car": [("front_bumper", ["front bumper", "parachoques"]), ("rear_bumper", ["rear bumper", "back bumper", "trasero"]),
            ("windshield", ["windshield", "windscreen", "front glass"]), ("side_mirror", ["mirror"]),
            ("headlight", ["headlight"]), ("taillight", ["taillight", "back light"]),
            ("hood", ["hood"]), ("door", ["door"]), ("body", ["body", "panel"])],
    "laptop": [("screen", ["screen", "display", "pantalla"]), ("keyboard", ["keyboard", "key", "teclas"]),
               ("trackpad", ["trackpad"]), ("hinge", ["hinge"]), ("lid", ["lid"]),
               ("corner", ["corner"]), ("body", ["body"])],
    "package": [("seal", ["seal"]), ("label", ["label"]), ("contents", ["content", "item", "inside", "product"]),
                ("package_corner", ["corner"]), ("package_side", ["side", "surface"]), ("box", ["box", "package"])],
}
_INJECTION = ["ignore all previous", "ignore previous", "approve", "mark this row", "follow the note",
              "skip manual review", "accept this", "mark supported"]


def _guess(text: str, candidates, default):
    low = text.lower()
    for value, keys in candidates:
        if any(k in low for k in keys):
            return value
    return default


def _mock_response(ctx: dict) -> dict:
    """Deterministic rule-based stub so the pipeline runs end-to-end without an API."""
    claim = ctx["claim"]
    present = ctx["present_ids"]
    obj = claim.get("claim_object", "")
    text = claim.get("user_claim", "")
    has_img = bool(present)
    flags = []
    if any(p in text.lower() for p in _INJECTION):
        flags.append("text_instruction_present")
    return {
        "evidence_standard_met": has_img,
        "evidence_standard_met_reason": "mock: image present" if has_img else "mock: no usable image",
        "risk_flags": flags or ["none"],
        "issue_type": _guess(text, _ISSUE_KEYWORDS, "unknown") if has_img else "unknown",
        "object_part": _guess(text, _PART_KEYWORDS.get(obj, []), "unknown"),
        "claim_status": "supported" if has_img else "not_enough_information",
        "claim_status_justification": "mock heuristic response (no model call).",
        "supporting_image_ids": present[:1],
        "valid_image": has_img,
        "severity": "medium" if has_img else "unknown",
    }


# --- client ---------------------------------------------------------------

class VLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self._hf: Any = None
        self._use_schema = settings.use_schema
        self.api_calls = 0       # real HF calls made (cache misses, non-mock)
        self.cache_hits = 0
        self._cache_path = Path(settings.cache_path)
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache), encoding="utf-8")
        tmp.replace(self._cache_path)

    @property
    def hf(self) -> Any:
        if self._hf is None:
            if not self.s.hf_token:
                raise RuntimeError("HF_TOKEN not set. Add it to .env or use --mock.")
            from huggingface_hub import InferenceClient
            self._hf = InferenceClient(provider=self.s.provider, api_key=self.s.hf_token)  # type: ignore[arg-type]
        return self._hf

    def _key(self, messages: list[dict], image_hashes: list[str]) -> str:
        import hashlib
        texts = []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                texts.append(c)
            else:
                texts.extend(p["text"] for p in c if p.get("type") == "text")
        payload = json.dumps([self.s.model_id, texts, sorted(image_hashes)], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, messages: list[dict], image_hashes: list[str], mock_ctx: dict) -> dict:
        """Return the model's raw field dict (cached). mock_ctx carries claim context for --mock."""
        # Mock bypasses the cache entirely: it is instant and must never poison the
        # on-disk cache that live runs (same model_id) read from.
        if self.s.mock:
            return _mock_response(mock_ctx)

        key = self._key(messages, image_hashes)
        with self._lock:
            if key in self._cache:
                self.cache_hits += 1
                return self._cache[key]

        result = self._call_hf(messages)
        with self._lock:
            self.api_calls += 1
            self._cache[key] = result
            self._save_cache()
        if self.s.sleep:
            time.sleep(self.s.sleep)  # pace request rate (live calls only)
        return result

    def _call_hf(self, messages: list[dict]) -> dict:
        content = self._chat_with_retry(messages)
        try:
            return _parse_json(content)
        except (ValueError, json.JSONDecodeError):
            repair = messages + [{"role": "user",
                                  "content": "Your previous reply was not valid JSON. "
                                             "Reply with ONLY the JSON object, nothing else."}]
            return _parse_json(self._chat_with_retry(repair, allow_schema=False))

    def _chat_with_retry(self, messages: list[dict], allow_schema: bool = True) -> str:
        last = None
        for attempt in range(self.s.max_retries):
            try:
                return self._raw_chat(messages, allow_schema)
            except Exception as e:  # network / 429 / 5xx -> backoff and retry
                last = e
                time.sleep(min(30.0, 1.5 * (2 ** attempt)) + random.uniform(0, 1))
        raise RuntimeError(f"VLM call failed after {self.s.max_retries} attempts: {last}")

    def _raw_chat(self, messages: list[dict], allow_schema: bool) -> str:
        if allow_schema and self._use_schema:
            rf = {"type": "json_schema",
                  "json_schema": {"name": "review", "schema": output_json_schema()}}
            try:
                return self._create(messages, rf)
            except Exception:
                self._use_schema = False  # provider rejected schema; degrade for rest of run
        return self._create(messages, None)

    def _create(self, messages: list[dict], response_format: dict | None) -> str:
        resp = self.hf.chat.completions.create(
            model=self.s.model_id,
            messages=messages,                 # type: ignore[arg-type]
            temperature=self.s.temperature,
            max_tokens=self.s.max_tokens,
            response_format=response_format,    # type: ignore[arg-type]
        )
        return resp.choices[0].message.content
