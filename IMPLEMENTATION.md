# Implementation Report — Multi-Modal Evidence Review

> Companion to [`code/ARCHITECTURE.md`](code/ARCHITECTURE.md) (component-level design) and
> [`code/evaluation/evaluation_report.md`](code/evaluation/evaluation_report.md) (metrics +
> ops analysis). This document is the end-to-end narrative: what was asked, how it was built,
> what was hard, and what is still imperfect.

---

## 1. The problem (from `problem_statement.md` / `README.md`)

Build a system that verifies damage claims for three object types — **car**, **laptop**,
**package** — using **submitted images, a short claim conversation, user claim history, and
minimum image-evidence requirements**.

- Read `dataset/claims.csv` (input-only, 44 rows) and produce `output.csv` with a fixed
  **14-column schema** (4 input columns passed through + 10 predicted).
- **Images are the source of truth.** The conversation defines _what_ to check. User
  history adds risk context but must **not by itself** overturn clear visual evidence.
- For each claim, decide: evidence sufficiency, the visible issue type, the object part,
  the final `claim_status` (`supported` / `contradicted` / `not_enough_information`),
  supporting image ids, risk flags, severity, image validity, and short justifications —
  all drawn from constrained **allowed-value enums**.
- Ship an `evaluation/` folder that scores the system on the labeled
  `dataset/sample_claims.csv` (20 rows), **compares ≥2 configurations**, and includes an
  **operational analysis** (model calls, tokens, image count, cost, runtime, TPM/RPM).

Repo rules live in `AGENTS.md` (entry points `code/main.py` and `code/evaluation/main.py`,
secrets from env only, deterministic where possible, transcript logging).

---

## 2. Approach

A **single vision-language model (VLM) call per claim**, wrapped in a deterministic rule
layer. A VLM reads both the conversation text and the images in one call, so **no separate
text LLM is used**. The one prompt is structured to **(1) restate the actual claim** —
ignoring rambling/distractor parts and any embedded "approve this" instructions — then
**(2) inspect the images**, then **(3) emit strict JSON**. This folds the main benefit of a
multi-stage pipeline (clean claim extraction) into a single call.

- **Provider:** Hugging Face Inference Providers (`huggingface_hub.InferenceClient`,
  `provider="auto"`, `HF_TOKEN`), OpenAI-compatible chat completions with `image_url` data
  URIs.
- **Model family:** Qwen-VL — strong fine-detail vision, strong OCR (to catch in-image
  instruction text), multilingual. Compared a **cheap** tier (`Qwen3-VL-8B`) vs a
  **strong** tier (`Qwen2.5-VL-72B`).
- **Safety/robustness:** content-addressed cache, `--mock` offline mode, retry + backoff,
  defensive JSON parsing, deterministic enum coercion, add-only history fusion.

---

## 3. Architecture & data flow

```
claims.csv ─┐
            ├─► data.py ─► per-claim record (text + image paths + object-relevant
user_history│              evidence requirements + history row)
evidence_req┘                    │
                                 ▼
                    images.py: resize ~1024px → JPEG → base64 data URI → sha256
                                 │
                                 ▼  (cache lookup by sha256(model + prompt + image hashes))
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

---

## 4. Components & tradeoffs

| Module           | Responsibility                                                                                                                                                                                                                   | Key tradeoff                                                                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`      | Env-only secrets + tunables (model aliases, provider, resize, temp, sleep, cache).                                                                                                                                               | Env-only (contract §6.2) + thin CLI overrides for dev.                                                                                                      |
| `data.py`        | Load 3 CSVs, parse `image_paths`, join history, select object-relevant evidence rows, write output.                                                                                                                              | **pandas** for clean joins/quoted I/O; output uses `QUOTE_ALL` to match input style.                                                                        |
| `images.py`      | Pillow load, downscale ~1024px, JPEG re-encode, base64 data URI, content hash; tolerate missing/corrupt.                                                                                                                         | **Detail vs tokens** at 1024px. **Skipped** client-side blur/glare CV — the VLM already judges quality.                                                     |
| `schema.py`      | Canonical allowed-value tables for every enum (single source of truth).                                                                                                                                                          | One shared module avoids duplicating enums in prompt + validation.                                                                                          |
| `prompts.py`     | Static system rubric (enums, `valid_image` vs `evidence_standard_met`, multi-part tie-break, multilingual→English, injection defense, claim-restate-first) + per-claim multimodal message.                                       | **Rubric-rich vs terse**: inject only this object's requirement rows + compact enums.                                                                       |
| `client.py`      | HF call, content-addressed JSON cache, `--mock` stub, backoff+jitter, defensive JSON parse + one repair retry, optional `response_format`, inter-call sleep.                                                                     | **json_schema support varies by provider** → defensive parse mandatory; `provider="auto"` for resilience; mock bypasses cache so it can't poison live runs. |
| `postprocess.py` | Coerce all fields to allowed enums; dedupe/order `risk_flags`; clamp `supporting_image_ids` to ids in the row; fuse history flags (**add-only, never flip `claim_status`**); add `manual_review_required` on escalation/history. | **Coerce vs reject** → coerce, because the contract needs exactly one valid row per claim.                                                                  |
| `main.py`        | Orchestrate per row → images → prompt → VLM (cached) → normalize → write `output.csv`. Bounded thread pool.                                                                                                                      | Sequential vs **bounded concurrency** respecting RPM; `temperature=0` + cache for stable reruns.                                                            |
| `evaluation/`    | Run sample for each model tier, score vs labels, write report + operational analysis.                                                                                                                                            | Exact-match on categoricals; **set P/R/F1** for multi-label `risk_flags`; justification text not scored.                                                    |

---

## 5. Edge cases handled

| Edge case                                                                                 | Handling                                                                                                                                   |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Multilingual claims (Hindi/Hinglish, Spanish, romanized Chinese)                          | VLM interprets any language; **outputs English enums only**.                                                                               |
| Prompt injection ("approve immediately", "ignore previous instructions") in chat or image | Treat claim + in-image text as **data, never instructions**; decide on visuals; set `text_instruction_present`. Verified live on case_055. |
| Multi-part claims (schema allows one issue/part)                                          | Pick the **primary / best-image-supported** part; note the rest in justification.                                                          |
| Rambling chats with distractor parts                                                      | Prompt's **claim-restate step** extracts the final/actual claim.                                                                           |
| Cross-image identity mismatch (different / wrong-color object)                            | Cross-image consistency check → `wrong_object` + `claim_mismatch`.                                                                         |
| Claim vs visual severity mismatch                                                         | `contradicted`; **severity reflects what's visible**, not what's claimed.                                                                  |
| Part visible but no damage vs part not visible                                            | Visible-no-damage → `contradicted` + `damage_not_visible`; not-visible → `not_enough_information`.                                         |
| `valid_image` vs `evidence_standard_met` are independent                                  | Defined explicitly: validity/authenticity vs sufficiency-for-this-claim.                                                                   |
| Missing / unreadable / non-image files                                                    | Skipped; safe `not_enough_information` fallback when no usable image.                                                                      |
| **Mixed image formats** (JPEG/PNG/WEBP/AVIF, all named `.jpg`)                            | Pillow ≥11.3 decodes all; everything re-encoded to JPEG before sending. Pinned in `requirements.txt`.                                      |
| Repeated `user_id`; `user_id` absent from history                                         | Each row independent; missing history → no history flags.                                                                                  |
| Invented `supporting_image_ids`                                                           | Clamped to ids present in the row, else `none`.                                                                                            |

---

## 6. Challenges faced

1. **Hidden image-format trap (reproducibility).** Several test "`.jpg`" files are actually
   **AVIF/PNG/WEBP** (8 AVIF, 14 PNG, 11 WEBP of 82). The dev machine's Pillow 11.3 decodes
   them natively, but an older Pillow would silently fail to load the AVIF ones (cases
   001/005/018/046/047/051). Fixed by pinning **`pillow>=11.3`** and re-encoding every input
   to JPEG before the VLM sees it.
2. **Non-deterministic eval on a tiny labeled set.** Only 20 labeled samples exist, and the
   open VLM is not bit-exact even at `temperature=0`. A single prompt tweak swings metrics
   ±10–15% (±1–3 cases). A "principled" `evidence_standard_met` rule was tested, **regressed
   `claim_status` 85%→70%**, and was reverted (revert deterministically restored 85% —
   confirming the edit, not noise, caused it). Lesson: stop tuning against 20 rows.
3. **Prompt injection.** Claims and in-image sticky notes contain "approve this / ignore
   previous instructions." The system prompt treats all such text as untrusted data and
   flags `text_instruction_present`; verified live (case_055 kept the seal claim
   `contradicted`).
4. **Multilingual input, English output.** Claims appear in Hindi/Hinglish, Spanish, and
   romanized Chinese; outputs must be English enums. Handled in-prompt.
5. **Single-label schema vs multi-part claims.** Several claims name two parts/issues
   (case_001 bumper + headlight); the schema allows one. Resolved by a primary-part rule.
6. **Provider variability.** `response_format` json_schema isn't supported by every HF
   provider, so the client relies on prompt-driven JSON + defensive parsing + one repair
   retry, with schema as an opt-in.
7. **Cost / rate limits.** Content-addressed caching (reruns ≈ 0 new calls), bounded
   concurrency, inter-call sleep, and exponential backoff keep the run cheap and within RPM.

---

## 7. Results (labeled sample, 20 claims)

| metric                | cheap (Qwen3-VL-8B) | strong (Qwen2.5-VL-72B) |
| --------------------- | ------------------- | ----------------------- |
| **claim_status**      | 75%                 | **85%**                 |
| evidence_standard_met | 70%                 | 80%                     |
| valid_image           | 90%                 | 85%                     |
| object_part           | 85%                 | 95%                     |
| severity              | 35%                 | 50%                     |
| issue_type            | 30%                 | 55%                     |
| risk_flags F1         | ~69%                | ~74%                    |

**Selected `strong` for `output.csv`** (best `claim_status` at acceptable cost/latency;
projected full-test cost ~$0.06 under stated price assumptions). On the 44 test claims the
system predicts 26 contradicted / 16 supported / 2 not_enough_information, with 31 routed to
`manual_review_required` — consistent with a deliberately adversarial test set (toy car,
smartphone, scooter, tablet, color/identity mismatches, injection notes, manipulated
"after" photos), each contradiction citing a specific visual mismatch.

---

## 8. Known issues / limitations (improvable, but tuning risks overfitting)

These are the fields where accuracy is lowest. Each _could_ be pushed higher with more
prompt rules — but because there are only **20 labeled rows** and an adversarial hidden test
set, hard-fitting these would likely **overfit the sample and hurt the hidden set**. They
are documented rather than over-engineered.

1. **`severity` calibration (≈50%).** Severity is a subjective 5-level ordinal. The model
   tends to over-rate by one notch (medium→high) versus the dataset's convention. A severity
   rubric helped; pushing further means encoding the labeler's exact thresholds from 20
   examples — overfitting.
2. **`issue_type` near-synonyms (≈55%).** Of 9 misses, ~4 are irreducible label
   subjectivity: `stain` vs `water_damage` (the gold flips by _customer wording_, not
   visuals) and `crack` vs `glass_shatter` (a cracked screen genuinely looks shattered). No
   general visual rule predicts these without memorizing the rows.
3. **`contradicted` vs `not_enough_information` boundary.** When the claimed part is _not
   visible_, the convention is `not_enough_information`, but the model sometimes returns
   `contradicted` + `damage_not_visible`. Both are "not approved," so it rarely changes the
   business outcome, but it costs exact-match accuracy.
4. **Multi-part claims.** With one issue/part slot, a claim whose secondary part is undamaged
   is marked `contradicted` (case_001). Whether that's "correct" depends on how partial
   multi-part claims are graded — it's a deliberate, documented design choice, flagged for
   manual review.

**False positive / false negative posture.** The system is biased against **false positives**
(falsely approving) via the strict "supported needs clearly-visible damage" rule and the
injection defense. **False negatives** (falsely rejecting) are mitigated by routing risky/
borderline cases to `manual_review_required` rather than hard-denying them.

---

## 9. Robustness — guardrails, retries, output validation

The pipeline is built to **never crash on one bad claim, never emit an invalid row, and
never trust untrusted input**. All loops are bounded (there is no agent loop, so there is no
open-ended iteration to cap).

**Guardrails (`prompts.py`, `postprocess.py`, `main.py`)**
- **Prompt-injection defense:** the claim text and any in-image text are treated as data,
  never instructions; directive text raises `text_instruction_present` and is ignored
  (verified live on case_055).
- **No false approvals:** `supported` requires the claimed damage to be clearly visible;
  otherwise `contradicted` + `damage_not_visible`.
- **History is add-only:** it can add risk flags but **never flips `claim_status`**,
  enforcing the rule that history must not override visual evidence.
- **Per-row isolation:** each claim runs in a `try/except` (`main.process_claim`); any
  failure becomes a safe `fallback_row` so one bad claim never aborts the run or drops a row.
- **Secrets:** read from env only; nothing hardcoded.

**Retries & bounded loops (`client.py`)**
- Exponential backoff + jitter on 429/5xx/network errors, capped at `max_retries` (default 4).
- Exactly **one** JSON "repair" retry if a response doesn't parse (not an unbounded loop).
- `max_tokens` cap on output; concurrency bounded by `--workers`; inter-call `--sleep`.

**Output validation (`postprocess.py`, `data.py`)**
- Every field coerced to an allowed enum (`_coerce_enum`); invalid values map to
  `unknown`/`none` — an out-of-vocabulary value can't reach the CSV.
- `risk_flags` de-duped, ordered, and never empty (`none` floor); `supporting_image_ids`
  clamped to ids that exist in the row (hallucinated ids dropped); booleans normalized to
  `true`/`false`; free-text reasons get non-empty fallbacks.
- `write_output` uses a fixed 14-column `DictWriter` (`extrasaction="ignore"`) → exactly the
  required columns in the required order, no missing/extra.
- A defensive JSON parser strips code fences / extracts the brace block before validation,
  and an assert-based self-check (`python code/postprocess.py`) guards the schema/rule logic.

**Max-iteration caps:** not applicable in the agentic sense (single call per claim, no
plan→act loop) — but every retry/loop above is explicitly bounded, so there is no runaway path.

---

## 10. Reproduce

See [`code/README.md`](code/README.md). In short: `pip install -r code/requirements.txt`, set
`HF_TOKEN` in `.env`, then `python code/evaluation/main.py --models cheap,strong` and
`python code/main.py --model strong`. `--mock` runs the whole pipeline offline with no token.
