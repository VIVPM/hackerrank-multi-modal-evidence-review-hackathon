# Architecture — Multi-Modal Evidence Review

## Problem in one line
For each damage claim (car / laptop / package), inspect the submitted images and decide
whether they **support / contradict / lack info** for the customer's claim, plus evidence
sufficiency, risk flags, issue type, object part, severity, supporting image ids, and
short justifications. **Images are the source of truth**; the chat says what to check;
user history only adds risk context and must not by itself overturn clear visual evidence.

## Approach
A **single vision-language model (VLM) call per claim**, wrapped in a deterministic rule
layer. A VLM is a superset of an LLM — it reads both the conversation text and the images
in one call — so **no separate text LLM is used**. The single prompt is structured to
**first restate the actual claim** (ignoring rambling/distractor parts and any embedded
"approve this" instructions), then inspect the images, then emit strict JSON. This folds
in the main benefit of a multi-stage pipeline (clean claim extraction) at one call/claim.

Provider: **Hugging Face Inference Providers** (`huggingface_hub.InferenceClient`,
`provider="auto"`, `HF_TOKEN`), OpenAI-compatible chat completions with `image_url` data
URIs. Model family **Qwen-VL** (strong fine-detail vision, strong OCR for in-image
instruction text, multilingual).

## Data flow
```
claims.csv ─┐
            ├─► data.py ─► per-claim record (text + image paths + object-relevant
user_history│              evidence requirements + history row)
evidence_req┘                    │
                                 ▼
                    images.py: resize ~1024px → JPEG → base64 data URI → sha256
                                 │
                                 ▼  (cache lookup by sha256(model+prompt+image hashes))
                    prompts.py: system rubric + per-claim multimodal user message
                                 │
                                 ▼
                    client.py: HF InferenceClient (retry/backoff) ──► JSON
                                 │            └─ mock mode (rule-based, no API)
                                 ▼
                    postprocess.py: coerce enums, clamp image ids, fuse history flags
                                 │
                                 ▼
                    data.write_output ─► output.csv (exact 14-col schema, QUOTE_ALL)
```

## Components & tradeoffs

| Module | Responsibility | Key tradeoff(s) |
|---|---|---|
| `config.py` | Env-only secrets + tunables (`HF_TOKEN`, model aliases, provider, resize, temp, sleep, cache). | Env-only (contract §6.2) + thin CLI overrides for dev. |
| `data.py` | Load 3 CSVs, parse `image_paths`, join history, select object-relevant evidence rows, write output. | **pandas** for clean joins/quoted I/O vs stdlib `csv` zero-dep. Output uses `QUOTE_ALL` to match input style. |
| `images.py` | Pillow load, downscale long edge ~1024px, JPEG re-encode, base64 data URI, content hash; filter non-images; tolerate missing/corrupt. | **Detail vs tokens** at 1024px. **Skipped** client-side blur/glare CV — the VLM already judges quality. |
| `schema.py` | Canonical allowed-value tables for every enum (single source of truth). | One small shared module avoids duplicating enums in prompt + validation. |
| `prompts.py` | Static system rubric (enums, `valid_image` vs `evidence_standard_met`, multi-part tie-break, multilingual→English, injection defense, claim-restate-first) + per-claim multimodal message. | **Rubric-rich vs terse**: inject only this object's requirement rows + compact enums. |
| `client.py` | HF call, content-addressed JSON cache, `--mock` stub, exponential backoff+jitter, defensive JSON parse + one repair retry, optional `response_format` json_schema, inter-call sleep. | **json_schema support varies by provider** → defensive parse is mandatory; schema is opt-in (`VLM_USE_SCHEMA`). `provider="auto"` for resilience. |
| `postprocess.py` | Coerce all fields to allowed enums; dedupe/order `risk_flags`; clamp `supporting_image_ids` to ids in the row; fuse history flags (**add-only, never flip `claim_status`**); add `manual_review_required` on escalation/history. | **Coerce vs reject** → coerce, because the contract needs exactly one valid row per claim. |
| `main.py` | Orchestrate: per row → images → prompt → VLM (cached) → normalize → write `output.csv`. Bounded thread pool. | Sequential vs **bounded concurrency** (`--workers`) respecting RPM; `temperature=0` + cache for stable reruns. |
| `evaluation/` | Run sample for each model tier, score vs labels, write `evaluation_report.md` + operational analysis. | Exact-match on categoricals; **set P/R/F1** for multi-label `risk_flags`; justification text not scored. |

## Edge cases handled

| Edge case | Handling |
|---|---|
| Multilingual claims (Hindi/Hinglish, Spanish, romanized Chinese) | VLM interprets any language; **outputs English enums only**. |
| Prompt injection / "approve this immediately" in chat or image | Treat claim + in-image text as **data, never instructions**; decide on visuals; set `text_instruction_present`. |
| Multi-part claims (schema allows one issue/part) | Pick the **primary / best-image-supported** part-issue; note the rest in justification. |
| Rambling chats with distractor parts | Prompt's **claim-restate step** extracts the final/actual claim. |
| Cross-image identity mismatch (different / wrong-color car) | Cross-image consistency check → `wrong_object` + `claim_mismatch`. |
| Claim vs visual severity mismatch | `claim_status=contradicted`; **severity reflects what's visible**, not what's claimed. |
| Part visible but no damage vs part not visible | Visible-no-damage → `contradicted` + `damage_not_visible`; not-visible → `not_enough_information` + `wrong_angle`. |
| `valid_image` vs `evidence_standard_met` are independent | Defined explicitly: `valid_image` = usable/authentic; `evidence_standard_met` = the valid set meets the requirement. |
| Missing / unreadable / non-image files | `images.py` skips them; `postprocess` forces a safe `not_enough_information` fallback when no usable image. |
| Repeated `user_id`; `user_id` absent from history | Each row independent; missing history → no history flags. |
| Invented `supporting_image_ids` | Clamped to ids present in the row, else `none`. |
| `issue_type`/`severity` `none` vs `unknown` | `none` = visible & no issue; `unknown` = cannot determine. |

## Evaluation strategy
**Option A — cheap vs strong model** (`Qwen3-VL-8B` vs `Qwen2.5-VL-72B`), identical prompt.
Scores `claim_status` (primary), `evidence_standard_met`, `valid_image`, `issue_type`,
`object_part`, `severity` accuracy, and `risk_flags` set P/R/F1 on the 20 labeled sample
claims; picks the better cost/accuracy model for `output.csv`. See `evaluation_report.md`.

### Documented fallback — Option C (multi-stage)
If evaluation shows **claim extraction** (not vision) is the failure mode, split into:
(1) a text-only stage that extracts `{object_part, issue_family, injection_detected}` from
the conversation, then (2) a focused VLM stage that inspects images for that intent, then
fuse. Tradeoff: ~2 calls/claim (higher cost/latency) and error-compounding risk. Not built
because this task's difficulty is the vision judgment, which both designs route through the
same VLM — so the single-call + claim-restate prompt captures the upside at one call/claim.

## Operational considerations
- **Caching**: content-addressed (`sha256(model + prompt text + image hashes)`) → reruns
  and re-evaluations cost ~0 new calls.
- **Rate limits**: bounded `--workers`, optional `--sleep` between live calls, exponential
  backoff + jitter on 429/5xx. HF chat endpoint has no native batch API → throughput is
  concurrency + cache.
- **Determinism**: `temperature=0` + cache. Open VLMs aren't bit-exact even at 0; the cache
  pins each row's result once computed.
- **Cost/tokens**: estimated in the report (text chars/4 + ~vision tokens/image), with
  stated, replaceable price assumptions.
