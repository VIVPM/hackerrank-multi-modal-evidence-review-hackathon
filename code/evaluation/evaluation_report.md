# Evaluation Report — Multi-Modal Evidence Review

- Dataset: `dataset/sample_claims.csv` — **20 labeled claims**, **29 images**.
- Strategy: single VLM call per claim, claim-restate-then-inspect prompt, deterministic history/rule fusion.
- Comparison (Option A): cheap vs strong model, identical prompt.


## Accuracy on the labeled sample

| metric | cheap | strong |
|---|---|---|
| claim_status_acc | 75.0% | 85.0% |
| evidence_standard_met_acc | 95.0% | 95.0% |
| valid_image_acc | 90.0% | 85.0% |
| issue_type_acc | 30.0% | 55.0% |
| object_part_acc | 85.0% | 95.0% |
| severity_acc | 35.0% | 50.0% |
| risk_flags_precision | 72.2% | 75.7% |
| risk_flags_recall | 66.7% | 71.8% |
| risk_flags_f1 | 69.3% | 73.7% |

## Model: `Qwen/Qwen3-VL-8B-Instruct` (alias `cheap`)

claim_status confusion (rows = gold, cols = predicted):

| gold \ pred | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 10 | 2 | 0 |
| contradicted | 2 | 3 | 0 |
| not_enough_information | 1 | 0 | 2 |

**Operational analysis (this sample run):**

- model calls (cache misses): 0 ; cache hits: 20
- images processed: 29
- est. input tokens: ~60,332 ; est. output tokens: ~4,000
- est. sample cost: ~$0.0072 (~$0.0004/claim)
- **projected cost for full test set (44 claims): ~$0.016**
- measured runtime: 0.6s (~0.03s/claim)

## Model: `Qwen/Qwen2.5-VL-72B-Instruct` (alias `strong`)

claim_status confusion (rows = gold, cols = predicted):

| gold \ pred | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 12 | 0 | 0 |
| contradicted | 2 | 3 | 0 |
| not_enough_information | 1 | 0 | 2 |

**Operational analysis (this sample run):**

- model calls (cache misses): 0 ; cache hits: 20
- images processed: 29
- est. input tokens: ~60,332 ; est. output tokens: ~4,000
- est. sample cost: ~$0.0289 (~$0.0014/claim)
- **projected cost for full test set (44 claims): ~$0.064**
- measured runtime: 0.5s (~0.02s/claim)

## Final strategy & recommendation

Selected **`Qwen/Qwen2.5-VL-72B-Instruct`** (alias `strong`) for `output.csv`: highest claim_status accuracy (85.0%) at acceptable cost/latency.

## Pricing assumptions & rate-limit strategy

- Prices are **illustrative** (USD/1M tokens): cheap=in $0.1/out $0.3; strong=in $0.4/out $1.2. Replace with your HF provider's published rates.
- Image tokens assumed ~1000/image at ~1024px; output ~200/claim.
- TPM/RPM: bounded `--workers` + `--sleep` between live calls + exponential backoff w/ jitter on 429/5xx. Content-addressed cache avoids repeat calls across reruns. HF chat endpoint has no native batch API, so throughput = concurrency + cache.

