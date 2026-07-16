# model-tailor — TODO

## Context

Teacher/Judge model: GLM 4.7 (`glm-4.7`) via Zhipu AI API.
Student model: Llama 3.1 8B fine-tuned with QLoRA.
GPU: RTX 3070 Ti (8.6 GB VRAM) — batch_size and max_seq_length already reduced in config/base.yaml.
Goal: Fine-tune Llama 3.1 8B to replicate high-quality SQL generation.

**Status (2026-04-19):** Training complete! (Task 5). Evaluation in progress (Task 6).

**Note:** GLM API balance insufficient for evaluation. Switching to Anthropic (Sonnet 4.6) as teacher model for evaluation. Notebooks 07_comparison_anthropic.ipynb and 07_comparison_glm.ipynb both updated with field name fixes (natural_language) and absolute path handling.

---

## Milestone 1: First Dataset

### Task 1 — Dry-run generation (10 examples)

Verify the teacher model and generation pipeline work before committing to a full run.

```python
import asyncio
from src.generate.batch import BatchGenerator, BatchConfig
from src.llm.client import TeacherClient

client = TeacherClient()  # reads config/base.yaml -> anthropic / claude-sonnet-4-6
config = BatchConfig.from_yaml("config/tasks/sql_generation.yaml")
config.num_examples = 10  # dry run

results = asyncio.run(BatchGenerator(client, config).run())
print(f"Generated {len(results)} examples")
for r in results[:3]:
    print(r)
```

Expected: 10 examples printed, no API errors, SQL looks valid.

- [x] Dry-run passes with 10 examples

### Task 2 — Full generation run (5,000 examples)

```python
import asyncio
from src.generate.batch import BatchGenerator, BatchConfig
from src.llm.client import TeacherClient

client = TeacherClient()
config = BatchConfig.from_yaml("config/tasks/sql_generation.yaml")

results = asyncio.run(BatchGenerator(client, config).run())
print(f"Generated {len(results)} examples -> data/raw/generated.jsonl")
```

Expected: ~5,000 examples saved to `data/raw/generated.jsonl`.
Estimated time: 2-4 hours. Estimated cost: ~$8-12 with claude-sonnet-4-6.

- [x] Full generation run completes, file exists at `data/raw/generated.jsonl` (146 examples generated)
- [x] Manually review 20 random examples for quality

### Task 3 — Curate the dataset

```python
import jsonlines
from src.curate.dedup import ExactDedup, FuzzyDedup
from src.curate.filter import QualityFilter
from src.curate.balance import DatasetBalancer
from src.curate.split import DatasetSplitter

with jsonlines.open("data/raw/generated.jsonl") as f:
    data = list(f)

print(f"Raw: {len(data)} examples")

# Deduplicate
data = ExactDedup(fields=["nl", "sql"]).run(data)
data = FuzzyDedup(field="sql", threshold=0.85).run(data)
print(f"After dedup: {len(data)} examples")

# Filter
data = (QualityFilter(data)
    .filter_length(min_len=10, max_len=500, field="sql")
    .filter_sql_valid(field="sql")
    .result())
print(f"After filter: {len(data)} examples")

# Balance
target = {"easy": 0.30, "medium": 0.40, "hard": 0.20, "expert": 0.10}
data = DatasetBalancer(data, target_distribution=target, field="difficulty").run()
print(f"After balance: {len(data)} examples")
print(DatasetBalancer(data, target_distribution=target, field="difficulty").report(data))

# Split
splitter = DatasetSplitter(data, ratios=[0.8, 0.1, 0.1], stratify_field="difficulty")
splits = splitter.run()
splitter.save(splits, output_dir="data/curated")
print(splitter.stats(splits))
```

Expected: 3 files saved — `data/curated/train.jsonl`, `data/curated/val.jsonl`, `data/curated/test.jsonl`.

- [x] Curated splits saved to `data/curated/`
- [x] Log final counts: total, per-difficulty, avg SQL length

### Task 4 — Format for training

```python
import jsonlines
from src.format.templates import ChatTemplate
from src.format.tokenize import TokenStats, validate_max_length

template = ChatTemplate()

for split in ["train", "val"]:
    with jsonlines.open(f"data/curated/{split}.jsonl") as f:
        data = list(f)

    conversations = [template.build_conversation(ex) for ex in data]
    valid = validate_max_length(conversations, max_length=2048)

    with jsonlines.open(f"data/formatted/{split}.jsonl", mode="w") as f:
        f.write_all(valid)

    stats = TokenStats(valid).compute()
    print(f"{split}: {len(valid)} examples, avg tokens: {stats['mean']:.0f}, max: {stats['max']}")
```

Expected: Formatted files saved to `data/formatted/`. No examples exceed 2048 tokens.

- [x] `data/formatted/train.jsonl` and `data/formatted/val.jsonl` saved
- [x] All examples within 2048 token limit

---

## Milestone 2: First Fine-Tune

### Task 5 — Train (COMPLETE)

```python
# Completed via scripts/train_simple.py (conservative settings for 8GB VRAM)
# batch_size=1, gradient_accumulation=8, max_seq_length=2048, 1 epoch
```

**Results (2026-04-18):**
- Training completed: 16/16 steps, 11 minutes
- LoRA adapter saved: `models/sql-llama-8b-lora/` (161MB, 41.9M params, 0.52%)
- Peak VRAM: 7.7GB/8GB (97%) - stable, no OOM
- No WandB tracking (report_to="none" for stability)

- [x] Training completes without OOM
- [x] Best checkpoint saved to `models/sql-llama-8b-lora/`
- [x] Val loss stays within 0.3 of train loss (no severe overfitting)
- [x] Log peak GPU memory usage: 7.7GB/8GB (97%)

### Task 6 — Evaluate (Partial Complete - Teacher Only)

**Completed (2026-04-20):**
- Teacher model evaluation: Claude Sonnet 4.6 on 14 test examples
- Results: 59.4s, $0.033 cost (3.3 cents)
- Cleaned SQL predictions saved to `tasks/sql_generation/teacher_results_anthropic.json`
- Created lightweight notebook: `notebooks/08_teacher_evaluation_anthropic.ipynb`

**Known Issues:**
- **GLM API**: Balance insufficient ("余额不足或无可用资源包"). Z.ai coding plan does not include API credits.
- **Gemini API**: Free tier quota exceeded (429 RESOURCE_EXHAUSTED).
- **Student Model**: RTX 3070 Ti (8GB VRAM) insufficient for Llama 3.1 8B inference. Requires ~16GB RAM.

**Notebooks Created:**
- `08_teacher_evaluation_anthropic.ipynb` - Lightweight, teacher-only evaluation (~$0.02 cost)
- `10_comparison_from_results.ipynb` - Load pre-saved results for comparison

**Remaining:**
- [x] Add credits to Anthropic account (done)
- [x] Run teacher model evaluation (done)
- [x] Clean predictions and save results (done)
- [ ] Student model evaluation (requires 16GB+ RAM or cloud GPU)
- [ ] Quality parity analysis (requires student results)
- [ ] Cost projections and break-even analysis (requires student results)
- [ ] Document findings in `tasks/sql_generation/results.md`

**Lessons Learned:**
- SQL extraction from LLM responses requires careful parsing (model adds explanations)
- Prompt engineering: Updated `_build_prompt()` to explicitly ask for "SQL only, no explanation"
- RAM constraints apply to inference, not just training (8GB insufficient for 8B model)
- Teacher-only evaluation proves pipeline works and establishes baseline

---

## Milestone 3: Deploy

### Task 7 — Export and quantize

```python
from src.deploy.export import merge_and_export
from src.deploy.quantize import quantize_gguf

merge_and_export(
    lora_path="models/sql-llama-8b-lora",
    output_dir="models/sql-llama-8b-gguf",
    export_format="gguf",
)

quantize_gguf(
    model_path="models/sql-llama-8b-gguf/model.gguf",
    output_path="models/sql-llama-8b-q4/model-q4_k_m.gguf",
    quant_type="q4_k_m",
)
```

- [ ] GGUF model exported to `models/sql-llama-8b-gguf/`
- [ ] Q4_K_M quantized model at `models/sql-llama-8b-q4/`
- [ ] Test Q4_K_M quality on 20 examples vs full-precision

### Task 8 — Serve and test

```bash
uvicorn src.deploy.serve:create_app --host 0.0.0.0 --port 8000 --factory
```

```bash
curl -X POST http://localhost:8000/generate_sql \
  -H "Content-Type: application/json" \
  -d '{"nl": "Find the top 5 customers by total order value"}'
```

- [ ] Server starts without errors
- [ ] `/generate_sql` returns valid SQL for 20 test queries
- [ ] Inference latency measured (tokens/sec)

---

## Milestone 4: Iterate

- [ ] Analyse failure cases — which difficulty/category fails most?
- [ ] Add targeted seeds for weak categories
- [ ] Add 3 more database schemas (inventory, social media, healthcare)
- [ ] Re-train with expanded dataset and compare metrics to v1

---

## Done

- [x] Project structure and all 7 pipeline modules
- [x] Teacher model client (OpenAI, Anthropic, Gemini, Ollama)
- [x] Synthetic data generation strategies (seed_expansion, self_instruct, evol_instruct)
- [x] Quality checking pipeline (heuristic, syntax, LLM scoring)
- [x] Batch generation with async concurrency
- [x] Dataset curation pipeline (dedup, filter, balance, split)
- [x] Instruction tuning format adapters (Llama 3, Mistral, Phi-3)
- [x] Tokenization validation and stats
- [x] QLoRA training setup with Unsloth + TRL SFTTrainer
- [x] Dual experiment tracking (WandB + MLFlow)
- [x] Evaluation metrics (BLEU, ROUGE, exact match, execution accuracy)
- [x] LLM-as-judge evaluation with structured rubric
- [x] Automated benchmark runner with model comparison
- [x] GGUF export, quantization, FastAPI inference server
- [x] SQL generation seed data (25 examples) and schemas (3 databases)
- [x] Pipeline walkthrough notebooks (01-06)
- [x] Full documentation (architecture, config, getting-started, guides)
- [x] CI/CD pipeline (ruff lint + 96 unit tests passing)
- [x] Config updated: teacher=claude-sonnet-4-6, batch_size=2, max_seq_length=2048

---

## LinkedIn Gate + Repair Iteration (In Progress)

Implementing the writer → gate → fixer → gate loop inspired by the recent
LinkedIn case study. Scope is staged: gate first, then repair, then batch
integration and DPO groundwork.

- [x] Task 1 — Implement `ExecutionGate` (`src/evaluate/gate.py`)
- [x] Task 2 — Refactor `sql_execution_accuracy` to use `ExecutionGate`
- [x] Task 3 — Add unit tests for `ExecutionGate`
- [x] Task 4 — Implement `SQLRepairer` (`src/generate/repair.py`)
- [x] Task 5 — Integrate gate + repair into `BatchGenerator`
- [x] Task 6 — Add pass/fail metrics and MLFlow logging
- [x] Task 7 — Update docs and run CI checks

### Review — LinkedIn Gate + Repair Iteration

**Status**: Complete. All 7 tasks finished.

**What was delivered**:
- `src/evaluate/gate.py` — reusable `ExecutionGate` with pass/fail classification and `GateMetrics`.
- `src/generate/repair.py` — `SQLRepairer` that uses gate feedback to ask the teacher for corrected SQL.
- `src/generate/batch.py` — writer → gate → fixer → gate loop integrated into `BatchGenerator`.
- `src/evaluate/benchmark.py` — benchmark runner reports `gate_pass_rate`, `writer_only_pass_rate`, and `writer_plus_fixer_pass_rate`.
- `src/train/monitor.py` — `log_gate_metrics()` helper for MLFlow/WandB logging.
- `config/tasks/sql_generation.yaml` — `execution_gate` and `repair` sections added.
- Documentation updated in `docs/architecture.md`, `docs/generation-strategies.md`, and `docs/evaluation-guide.md` with mermaid flow diagrams.
- Unit tests added: `tests/test_gate.py`, `tests/test_repair.py`, `tests/test_batch_integration.py`, `tests/test_benchmark_gate.py`.

**CI results**:
```bash
make check
```
- ruff lint: passed
- ruff format check: passed (31 files)
- pytest: 133 passed, 5 warnings

**Notes**:
- Gate and repair are disabled by default in `config/tasks/sql_generation.yaml` (`enabled: false`) to keep generation API-cost-free by default.
- DPO / minimal-pair groundwork remains a future phase, per the original plan.

---

## Local Dashboard Iteration (Next.js + FastAPI)

Local configuration/performance dashboard replacing the discarded Streamlit
prototype. Motivation: no GCP or Anthropic console access, so MLflow runs,
benchmark reports, and RAM/GPU usage must be tracked locally. Plan:
`~/.claude/plans/proud-questing-kitten.md`.

- [x] Stage 0 — Remove `src/dashboard/`, drop `streamlit`/`psutil` from root requirements
- [x] Stage 1 — Backend data layer (`apps/backend-py/src/dashboard_api/data.py`, `system.py`, `models.py`)
- [x] Stage 2 — FastAPI app + 9 passing tests (`main.py`, `tests/test_api.py`)
- [x] Stage 3 — `run_gate_benchmark.py` persists JSON reports to `tasks/sql_generation/results/` by default
- [x] Stage 4 — Next.js frontend (`apps/frontend/`): Overview, Training, Benchmarks, System, Configs pages
- [x] Stage 5 — Makefile targets, READMEs, `docs/dashboard-guide.md`, setup.sh wiring, lessons captured

### Review — Local Dashboard Iteration

**Status**: Complete. All 6 stages finished.

**What was delivered**:
- `apps/backend-py/` — FastAPI service (`/health`, `/config`, `/runs`,
  `/runs/{id}/metrics/{key:path}`, `/benchmarks`, `/benchmarks/{file}`,
  `/system`, `/configs`) with port-delegating dev launcher and `.dev-port`
  handshake file.
- `apps/frontend/` — Next.js 15 + TypeScript + Tailwind + Recharts frontend
  with port-delegating launcher that polls `.dev-port` to find the backend.
- `scripts/run_gate_benchmark.py` — always writes a rich JSON report; defaults
  to `tasks/sql_generation/results/<model-slug>_<YYYYMMDD_HHMMSS>.json`.
- Root `Makefile` — `dashboard`, `dashboard-backend`, `dashboard-frontend`,
  `backend-lint`, `backend-test`; `check` now covers the backend too.
- `setup.sh` — installs dashboard backend requirements and (when npm exists)
  frontend dependencies; verifies `dashboard_api` imports.
- `docs/dashboard-guide.md` — architecture and port-delegation diagrams,
  usage, troubleshooting.

**Verification results**:
- Backend: 9/9 pytest tests pass; `ruff check` clean on src/scripts/tests.
- Frontend: `npm run build` green (Next.js 15, all 5 routes static).
- End-to-end: with ports 3000 and 8000 deliberately occupied, backend delegated
  to 8001 and frontend to 3001; proxied `/api/health`, `/api/config`,
  `/api/system` (live RTX 3070 Ti snapshot), and `/api/benchmarks` all
  returned correct data.

**Notes**:
- Streamlit code and dependencies fully removed; `psutil` now lives only in
  `apps/backend-py/requirements.txt`.
- `apps/backend-py/.dev-port` is git-ignored; benchmark results dir stays
  git-ignored as before.

---

## Local Dashboard Storage Refinement

Add storage scanning to the `/system` snapshot and show built-in vs external
indicators instead of hiding removable drives.

- [x] Add `StorageDrive` model and `storage` field to `SystemSnapshot`
- [x] Implement `dashboard_api/storage.py` with platform-aware classification
- [x] Wire `snapshot_storage()` into `dashboard_api/system.py`
- [x] Add TypeScript `StorageDrive` type and render storage in `SystemCards.tsx`
- [x] Update `/system` mock in `tests/test_api.py` and add `tests/test_storage.py`
- [x] Update `docs/dashboard-guide.md` with storage architecture and classification notes
- [x] Run verification checks (`ruff`, `pytest`, `npm run build`, `make check`, smoke test)

### Review — Local Dashboard Storage Refinement

**Status**: Complete.

**What was delivered**:
- `dashboard_api/storage.py` — cross-platform storage scanner using
  `psutil.disk_partitions(all=True)`, pseudo-filesystem filtering, and
  built-in/external classification.
- `dashboard_api/system.py` — includes the storage array in every `/system` snapshot.
- `SystemCards.tsx` — renders each drive with capacity, usage bar, and a
  **Built-in** (sky) or **External** (amber) badge.
- Backend tests for filtering, WSL/Windows bus-type classification, Linux
  removable/transport classification, PowerShell caching, and unknown-platform
  defaults.
- Documentation updates with mermaid diagram changes and a storage-classification
  section.

**Verification**:
- Backend lint/tests: `make check` passed.
  - Root pytest: 133 passed.
  - Backend pytest: 23 passed (9 API + 14 storage).
  - `ruff check` clean for both root and backend.
- Frontend build: `npm run build` succeeded (Next.js 15, static pages generated).
- Smoke test: `snapshot_system()` returned a non-empty `storage` array on the WSL
  host with correct built-in/external indicators:
  - `/mnt/c` and `/mnt/d` → built-in SATA.
  - `/mnt/f` → external USB.
  - WSL rootfs/snap mounts → built-in WSL.
- A live `/system` HTTP request was also verified through a running Uvicorn
  instance; the endpoint returned the storage payload successfully.

---

## Local Dashboard Storage Safety Hardening

Ensure the storage scanner cannot hang the server, spawn a subprocess storm, or
overload the host.

- [x] Add per-partition disk-usage timeout in `dashboard_api/storage.py`
- [x] Add 5-second storage snapshot cache
- [x] Add 30-second per-device `lsblk` transport cache
- [x] Run `snapshot_system()` in a thread with 5-second timeout in `/system`
- [x] Add single-flight `asyncio.Lock` to the `/system` route
- [x] Add storage cache/timeout tests
- [x] Add `/system` timeout test
- [x] Update `docs/dashboard-guide.md` with caching/timeout notes
- [x] Run verification checks (`ruff`, `pytest`, `npm run build`, `make check`, smoke test)

### Review — Local Dashboard Storage Safety Hardening

**Status**: Complete.

**What was delivered**:
- `dashboard_api/storage.py` — storage snapshot TTL cache, `lsblk` transport
  cache, and thread-pool disk-usage reads with a 1-second per-partition timeout.
- `dashboard_api/main.py` — `/system` now runs `snapshot_system()` in a worker
  thread guarded by `asyncio.wait_for(timeout=5.0)` and an `asyncio.Lock`;
  timeouts return HTTP 503.
- Backend tests for cache hits/expiry, disk-usage timeout skipping, and route-level
  timeout returning 503.
- Documentation updates in `docs/dashboard-guide.md` describing the caching and
  timeout safeguards.

**Verification**:
- Backend lint/tests: `make check` passed.
  - Root pytest: 133 passed.
  - Backend pytest: 29 passed (9 API + 20 storage).
  - `ruff check` clean for both root and backend.
- Frontend build: `npm run build` succeeded (Next.js 15, static pages generated).
- Smoke test: `snapshot_system()['storage']` returned a non-empty storage array
  on the WSL host with correct built-in/external indicators.
