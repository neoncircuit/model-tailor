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
