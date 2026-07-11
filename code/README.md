# Multi-Modal Evidence Review — Solution

Verifies damage claims (car / laptop / package) by inspecting submitted images with a
vision-language model (VLM) and a deterministic rule layer. For each row in
`dataset/claims.csv` it produces one row in `output.csv`.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design, components, tradeoffs, and edge
cases.

## Setup

```bash
pip install -r code/requirements.txt
```

Add your Hugging Face token to `.env` at the repo root (gitignored; copy from
`.env.example`):

```
HF_TOKEN=hf_xxx
```

A token is **not** needed for `--mock`. Get one at https://huggingface.co/settings/tokens
(Inference Providers access).

## Run

```bash
# Offline dry run — no token, rule-based stub (validates the full pipeline + I/O)
python code/main.py --mock --limit 3

# Live run -> writes output.csv at the repo root
python code/main.py --model cheap          # Qwen3-VL-8B  (alias)
python code/main.py --model strong          # Qwen2.5-VL-72B (alias)
python code/main.py --model Qwen/Qwen2.5-VL-7B-Instruct   # any HF model id

# Evaluate on the labeled sample set (model comparison + operational report)
python code/evaluation/main.py --mock                 # validate harness offline
python code/evaluation/main.py --models cheap,strong  # live comparison
```

Outputs:
- `output.csv` — predictions for all rows in `dataset/claims.csv` (14-column schema).
- `code/evaluation/evaluation_report.md` — metrics, model comparison, operational analysis.

## CLI flags (`main.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `cheap` | alias (`cheap`/`strong`) or any HF model id |
| `--mock` | off | offline rule-based stub (no API) |
| `--limit N` | all | process only the first N claims |
| `--sleep S` | `1.5` | seconds between live calls (skipped on cache hits) |
| `--workers N` | `3` | concurrent workers |
| `--input` / `--output` | claims.csv / output.csv | I/O paths |

## Environment variables

| Var | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token (required for live runs) |
| `HF_PROVIDER` | inference provider, default `auto` |
| `VLM_MODEL_CHEAP` / `VLM_MODEL_STRONG` | override the alias→model mapping |
| `VLM_USE_SCHEMA` | set `1` to request `response_format` json_schema (if your provider supports it) |

## Notes
- **Caching**: VLM responses are cached in `code/.cache/` keyed by model + prompt + image
  hashes, so reruns and evaluations cost ~0 new calls.
- **Determinism**: `temperature=0` + cache. Secrets come from env only; nothing is hardcoded.
- **Robustness**: bad/missing images and failed calls degrade to a safe
  `not_enough_information` row — the run never crashes on one claim.
- **Self-check**: `python code/postprocess.py` runs assert-based validation of the
  schema/normalization/rule logic.

## Layout
```
code/
  main.py            config.py   data.py    images.py
  prompts.py         client.py   schema.py  postprocess.py
  evaluation/        main.py     metrics.py  evaluation_report.md
  ARCHITECTURE.md    README.md   requirements.txt
```
