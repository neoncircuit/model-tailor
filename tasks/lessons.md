# model-tailor — Lessons Learned

A living document. Update this as you run experiments and discover what works.

---

## Synthetic Data Generation

### Seed Quality > Seed Quantity
15 well-crafted seed examples produce better generations than 50 sloppy ones. Each seed should be a real, practical query that a user would actually ask. Avoid contrived or textbook-style examples — the teacher model will mimic the style and quality of whatever seeds you give it.

### Three-Layer Quality Checking Saves Money
The quality pipeline runs heuristic checks first (free), then SQL syntax validation (free), then LLM scoring (costs API tokens). This ordering catches ~60-70% of bad examples before any API call is made. If you reverse the order, you'll burn tokens scoring garbage.

### Strategy Selection by Difficulty
- **Easy/Medium**: SeedExpansion works best. It produces high-quality variations that stay close to proven patterns.
- **Hard/Expert**: EvolInstruct is the clear winner. It takes working easy/medium examples and systematically adds complexity (JOINs, subqueries, window functions). SelfInstruct at hard difficulty often produces invalid SQL.
- **Broad Coverage**: SelfInstruct fills gaps that seeds don't cover. Good for generating examples across tables and categories you didn't seed.

### Common Teacher Model Failures
- Generates SQL that references tables/columns not in the schema
- Produces syntactically valid but semantically wrong queries (e.g., `GROUP BY` without aggregation)
- Copies seed examples with trivial modifications (just changes a number or column name)
- Returns malformed JSON — always use a robust parser with fallbacks

### Batch Size Sweet Spot
Asking the teacher for 10 examples per request gives the best quality/throughput tradeoff. At 20+, the model starts generating repetitive or lower-quality examples near the end. At 3-5, the overhead of system prompt tokens dominates cost.

---

## Dataset Curation

### Deduplication is Non-Negotiable
Without dedup, teacher models will produce 15-25% near-duplicates across batches. Exact dedup catches identical copies. Fuzzy dedup (MinHash at 0.85 Jaccard threshold) catches paraphrased duplicates where only a column name or filter value changed. Run both, in that order.

### Distribution Balancing is Critical
Without explicit balancing, generated datasets skew heavily toward easy/medium examples (70-80%). Models trained on this distribution overfit to simple patterns and fail on hard queries. Target distribution that works well for SQL: 30% easy, 40% medium, 20% hard, 10% expert.

### Stratified Splits Prevent Data Leakage
Always use stratified splitting so every difficulty level and SQL category appears proportionally in train, val, and test. Random splits can accidentally put all your expert examples in the training set and none in test, giving you false confidence in evaluation.

### SQL Validation Catches More Than You'd Expect
Running `sqlparse.parse()` catches about 5-10% of generated examples that look valid to the eye but have subtle issues: unbalanced parentheses, trailing commas in SELECT lists, unclosed string literals, or CTE syntax errors. Always validate.

---

## Instruction Tuning Format

### Model Family Templates Matter
Using the wrong chat template silently degrades performance. A Llama 3.1 model trained with Mistral's `[INST]` tokens will work, but you'll lose 5-15% on eval metrics compared to using the correct `<|start_header_id|>` format. Always match the template to the model family.

### System Prompt Consistency
Use the exact same system prompt at training time and inference time. Even small differences (extra whitespace, rephrased instructions) can hurt performance. Define the system prompt once in config and reference it everywhere.

### Context Length Budgeting
For SQL generation, most examples fit comfortably in 512-1024 tokens. Set max context to 2048 for safety. If any examples exceed this, truncate or drop them rather than splitting — split SQL queries are useless for training.

### ShareGPT Format for Compatibility
Converting to ShareGPT format (`{"from": "human", "value": "..."}`) before training ensures compatibility with most fine-tuning frameworks (TRL, Axolotl, LLaMA-Factory). It's a universal interchange format.

---

## Training

### QLoRA Configuration
- **Rank 16, Alpha 16** is the sweet spot for 8B models on single-GPU setups. Rank 8 works for simpler tasks. Rank 32+ gives diminishing returns and increases VRAM usage.
- **Target all projection layers** (q, k, v, o, gate, up, down for Llama). Targeting only attention layers leaves performance on the table.
- **4-bit quantization** (QLoRA) cuts VRAM from ~32GB to ~8-12GB with negligible quality loss. Use Unsloth's optimized implementation, not vanilla bitsandbytes.

### Learning Rate and Scheduling
- **2e-4 with cosine scheduler** is a reliable starting point. If loss plateaus early, try 5e-5.
- **Warmup ratio 0.03** (3% of total steps) prevents early instability.
- **3 epochs** is usually enough for 3,000-5,000 examples. More epochs = overfitting risk.

### Overfitting Detection
Watch the gap between training loss and validation loss. Rules of thumb:
- Gap < 0.1: healthy
- Gap 0.1-0.3: mild overfitting, probably fine
- Gap > 0.3: significant overfitting, stop training or add more data
- Val loss increasing while train loss decreasing: stop immediately

### Gradient Checkpointing
Always enable it. Unsloth's optimized gradient checkpointing (`use_gradient_checkpointing="unsloth"`) cuts VRAM usage by ~40% with only 5-10% training slowdown. Worth it every time.

### Checkpointing Strategy
Save checkpoints every epoch and keep the best one (lowest val loss). Don't just keep the last checkpoint — the best model is often from epoch 2, not epoch 3.

---

## Evaluation

### Execution Accuracy is the Gold Standard for SQL
BLEU and ROUGE measure surface-level text similarity. A query can have 0.9 BLEU but return completely wrong results. Execution accuracy (run both queries against a real DB and compare result sets) is the only metric that truly measures correctness.

### Metric Correlation
From experience with SQL tasks:
- Execution accuracy and exact match correlate at ~0.6-0.7
- BLEU and execution accuracy correlate at ~0.3-0.4
- ROUGE-L and execution accuracy correlate at ~0.3-0.5
- LLM-as-judge scores correlate with execution accuracy at ~0.7-0.8

Bottom line: use execution accuracy as your primary metric, LLM-as-judge for qualitative insights, and BLEU/ROUGE for quick sanity checks only.

### LLM-as-Judge Best Practices
- Use a structured rubric with explicit scoring criteria (correctness, efficiency, readability, edge cases)
- Set temperature to 0.1 for consistency across evaluations
- The judge should see both the prediction and the reference — this is scoring, not blind generation
- Average over at least 50 examples to get stable scores
- GPT-4o-mini is cost-effective for judging; GPT-4o is better but 10x the cost

### Per-Category Evaluation is Essential
Overall accuracy can mask severe weaknesses. A model might score 75% overall but only 20% on window function queries. Always break down metrics by:
- Difficulty level (easy/medium/hard/expert)
- SQL category (select, aggregate, join, subquery, window_function)
- Schema (which database the query targets)

---

## Deployment

### Quantization Tradeoffs
| Format | Size (8B model) | Quality Loss | Use Case |
|--------|-----------------|-------------|----------|
| F16 | ~16 GB | None | Development, benchmarking |
| Q8_0 | ~8 GB | Negligible | Production with GPU |
| Q5_K_M | ~5.5 GB | Minimal | Production, balanced |
| Q4_K_M | ~4.5 GB | Small (1-3%) | Production, resource-constrained |

Q4_K_M is the default recommendation. The 1-3% quality loss is acceptable for most production use cases, and the 4x size reduction means you can run on machines with 8GB RAM.

### GGUF vs HuggingFace for Inference
- **GGUF (llama.cpp)**: Faster for single-request latency, lower memory overhead, works on CPU. Use for serving.
- **HuggingFace transformers**: Better for batched inference, easier integration with Python ecosystem, GPU-optimized. Use for evaluation and batch processing.

### FastAPI Server Configuration
- Set `n_gpu_layers=-1` for full GPU offloading with GGUF models
- Use low temperature (0.1) for SQL generation — you want deterministic outputs
- Stop on semicolon for SQL tasks to prevent rambling
- Health check endpoint at `/health` for load balancers and monitoring

---

## Cost

### Generation Costs (as of 2025)
| Provider | Model | Cost per 1M input tokens | Cost per 1M output tokens |
|----------|-------|-------------------------|--------------------------|
| OpenAI | GPT-4o | $2.50 | $10.00 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |
| Anthropic | Claude Sonnet | $3.00 | $15.00 |
| Google | Gemini 2.0 Flash | $0.10 | $0.40 |
| Google | Gemini 2.5 Pro | $1.25 | $10.00 |
| Ollama | Any local model | $0 | $0 |

### Budget Estimates
- **5,000 examples with GPT-4o-mini**: ~$2-5 total (generation + quality scoring)
- **5,000 examples with GPT-4o**: ~$15-30 total (higher quality, 10x cost)
- **Training on RunPod A100 (40GB)**: ~$1.50/hr, training takes 1-3 hours = $2-5
- **Total pipeline cost**: $5-35 depending on teacher model choice

### The ROI Case
- GPT-4o costs ~$12.50/M tokens (blended input/output)
- A fine-tuned 8B model via GGUF on a $50/month VPS costs ~$0.00 per token
- At 1M tokens/month usage, the fine-tuned model pays for itself in the first month
- At 10M tokens/month, you save $125/month = $1,500/year

### Using Ollama to Cut Costs to Zero
If you have a local GPU (RTX 3090/4090 or better), use Ollama as the teacher model for generation. Quality will be lower than GPT-4, but for simpler tasks (basic SQL, templated emails) it's often good enough. The entire pipeline then costs $0 in API fees.

---

## Project Standards

### CLAUDE.md as the Single Source of Truth
Having a CLAUDE.md at the project root keeps every contributor (human and AI) aligned on conventions. Key rules that paid off:
- **Plan before coding**: Non-trivial tasks go through a plan → approve → implement cycle. This avoids wasted effort from misunderstood requirements.
- **Google-style docstrings everywhere**: Standardizing on one format (Args/Returns/Raises) makes the codebase grep-friendly and self-documenting. The consistency matters more than the specific style.
- **Mermaid diagrams in every doc**: Text-only architecture descriptions are hard to follow. A single flowchart at the top of each doc gives readers an instant mental model before they dive into details.
- **Dual experiment tracking (WandB + MLFlow)**: WandB excels at real-time monitoring dashboards. MLFlow excels at model registry and comparison across runs. Using both covers the full experiment lifecycle without compromise.
- **CI/CD from day one**: Even a simple lint + test pipeline catches regressions before they compound. `make check` locally mirrors what CI runs, so there are no surprises on push.

---

## NLP Library Initialization (2026-04-20)

### ML Library Imports Are Slow (Expected Behavior)
Importing `torch` + `transformers` takes 1-2 minutes on first load. This is normal and unavoidable - these libraries load many components.

**Breakdown of import times:**
- `torch` + `transformers`: ~1m47s (main bottleneck)
- `nltk.translate.bleu_score`: ~10s
- `rouge_score`: ~7s
- Other imports: few seconds
- **Total: ~2 minutes**

**Fix Applied (2026-04-20):**
1. Removed unnecessary `nltk.data.find()` check from `metrics.py`
2. Added progress indicators to all comparison notebooks to show import progress
3. NLTK downloads now show progress (removed `quiet=True`)

**Usage Pattern:** The imports will be slow on first run, but subsequent runs in the same kernel session will be faster (cached).

```python
# This will take ~2 minutes on first run (normal)
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluate.metrics import evaluate_batch
```

**Note:** This delay is expected behavior for ML libraries and cannot be significantly optimized.

---

## Data Format and Field Names (2026-04-19)

### Field Name Consistency is Critical
When generating synthetic data, choose descriptive field names and use them consistently throughout the pipeline. Changing from `nl` to `natural_language` mid-project causes KeyErrors across notebooks, evaluation scripts, and judge code.

**Fix:** Use descriptive names from the start:
- `nl` → `natural_language` (clearer intent)
- `prompt` → `user_query` or `question` (avoids ambiguity)

**Pattern:** When renaming fields, grep across the entire codebase:
```bash
grep -r "old_field_name" --include="*.py" --include="*.ipynb" .
```

### Absolute Paths Prevent Notebook Errors
Jupyter notebooks inherit their working directory from where Jupyter was launched. Relative paths like `"data/curated/test.jsonl"` fail when running from different directories.

**Fix:** Add a `find_project_root()` function to notebooks and use absolute paths:
```python
def find_project_root() -> Path:
    MARKER_FILES = ["CLAUDE.md", "pyproject.toml", ".git"]
    current_path = Path.cwd()
    for parent in [current_path] + list(current_path.parents):
        for marker in MARKER_FILES:
            if (parent / marker).exists():
                return parent
    raise RuntimeError(f"Cannot find project root from {current_path}")

PROJECT_ROOT = find_project_root()
TEST_DATA_PATH = PROJECT_ROOT / "data" / "curated" / "test.jsonl"
CONFIG_PATH = PROJECT_ROOT / "config" / "base.yaml"
```

This works regardless of where Jupyter is started.

---

## API Provider and Billing (2026-04-19)

### Z.ai Coding Plan ≠ GLM API Credits
Zhipu AI's coding assistant subscription (Z.ai) is separate from the GLM API credits. Having a coding plan does NOT include API tokens for programmatic access.

**Error:** `Error code: 429 - {'error': {'code': '1113', 'message': '余额不足或无可用资源包,请充值。'}}`

**Solution:** Switch to a provider with active credits (OpenAI, Anthropic, or add GLM API balance).

### Provider-Agnostic Pipeline Design
The `TeacherClient` supports multiple providers (openai, anthropic, gemini, glm, ollama). Switching is as simple as editing `config/base.yaml`:

```yaml
teacher:
  provider: anthropic  # Change from 'glm'
  model: claude-sonnet-4-6
```

All downstream code works identically — no changes needed in generation, evaluation, or judge scripts.

---

## Hardware Constraints Beyond Training (2026-04-19)

### VRAM Constraints Apply to Inference Too
RTX 3070 Ti (8GB VRAM) can train Llama 3.1 8B with QLoRA (4-bit quantization, gradient checkpointing), but inference still requires ~8GB VRAM. Loading the model for evaluation fails with:

```
ValueError: Some modules are dispatched on the CPU or the disk...
```

**Workarounds:**
1. **CPU-only inference** (slow but functional):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       torch_dtype=torch.float32,
       device_map="cpu",  # Force CPU
   )
   ```

2. **Skip student evaluation** — Evaluate teacher only and document hardware limitation

3. **Use a smaller student model** — Llama 3.2 3B fits comfortably in 8GB VRAM

**Rule of thumb:** If training barely fits VRAM, inference will also struggle. Plan for CPU fallback or smaller models.

---

## Debugging and Troubleshooting (2026-04-18)

### MLFlow Connection Handling
MLFlow's `set_experiment()` throws an exception when the server isn't running, which kills training before it starts. The fix: wrap MLFlow setup in a try/except and continue without it if the server is unavailable. Training works fine with just WandB tracking.

**Pattern:**
```python
def setup_mlflow(config_path: str = "config/base.yaml") -> None:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not connect to MLFlow: {e}. Training will continue without MLFlow tracking.")
```

### SFTTrainer API Changes (TRL v0.12+)
The `SFTTrainer` constructor changed from `tokenizer=` to `processing_class=`. Code that uses the old parameter silently fails with confusing type errors. Always check the current API signature when upgrading TRL.

**Migration:**
```python
# Old (TRL < 0.12)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)

# New (TRL >= 0.12)
trainer = SFTTrainer(model=model, processing_class=tokenizer, ...)
```

### TrainingRunner Expects Dataset Objects
`TrainingRunner` doesn't accept file paths directly — you need to load JSONL into HuggingFace `Dataset` objects first. The `scripts/train_sql_model.py` helper script handles this correctly.

**Correct pattern:**
```python
train_ds = load_formatted_data("data/formatted/train.jsonl")  # Returns Dataset
runner = TrainingRunner(model, tokenizer, train_dataset=train_ds, ...)
```

### Silent Training Failures
When background processes exit with code 0 but produce no output, the issue is usually:
1. Import order (Unsloth must be imported before TRL/transformers)
2. API incompatibility (wrong parameter names)
3. Missing MLFlow server (handled now with try/except)

Always run training with `stderr redirected to stdout` and capture full error traces when debugging.

### API Provider Switching
You can switch the teacher model provider at any time by editing `config/base.yaml`:
```yaml
teacher:
  provider: glm        # Change from 'anthropic' to 'glm'
  model: glm-4.7       # Use GLM model instead of claude-sonnet-4-6
```

This is useful for:
- Evaluation (Task 6): Use GLM as the judge instead of Anthropic
- Future data generation: Use GLM instead of Claude for cost savings

The pipeline is provider-agnostic — all code works the same way regardless of which provider you choose.

---

## System Stability (2026-04-18)

### BSoD Risk from Concurrent Training Processes
Running multiple training processes simultaneously on RTX 3070 Ti (8GB VRAM) can cause system crashes with blue screen of death. This happens when VRAM is exhausted and the system can't recover.

**Prevention:**
- Always check `ps aux | grep -E "python.*train"` before starting new training
- Use `pkill -f "python.*train"` to clean up orphaned processes
- Run only ONE training instance at a time
- Monitor VRAM with `nvidia-smi` during training

**Safe Training Settings for 8GB VRAM:**
```python
TrainingArguments(
    per_device_train_batch_size=1,        # Conservative
    gradient_accumulation_steps=8,         # Maintain effective batch size
    max_seq_length=2048,                   # SQL rarely exceeds this
    bf16=True,                             # Ampere supports bfloat16
    gradient_checkpointing=True,           # Saves VRAM
)
```

### Data Format for Causal Language Modeling
Causal LM models require a `labels` field in the dataset. The labels are typically identical to `input_ids` (the model learns to predict the next token).

**Common Error:** `ValueError: The model did not return a loss from the inputs`

**Fix:**
```python
def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")
    result["labels"] = result["input_ids"].clone()  # Required for causal LM
    result.pop("text", None)  # Remove text field to avoid dimension issues
    return result
```

### VRAM Management on RTX 3070 Ti
The RTX 3070 Ti has 8GB VRAM. Here's how to stay within budget:

| Component | VRAM Usage | Total |
|-----------|------------|-------|
| Base model (4-bit) | ~5.5 GB | 5.5 GB |
| LoRA adapter | ~0.5 GB | 6.0 GB |
| Batch size 2 | ~1.5 GB | 7.5 GB |
| Batch size 1 | ~0.8 GB | 6.8 GB |
| Overhead | ~0.5 GB | - |

**Safe configuration:** batch_size=1, gradient_accumulation=8, max_seq_length=2048

This keeps total VRAM at ~7.3GB / 8GB (91%) — safe but efficient.

## Prompt Engineering for Output Format (2026-04-20)

### LLM Models Add Explanations Even When Not Asked
Claude Sonnet 4.6 naturally adds explanation text after SQL responses, even with simple prompts like "Convert this question to SQL."

**Problem:** Responses like:
```sql
SELECT * FROM users;
```

### Explanation:
- **SELECT**: Selects data
```

This breaks naive SQL extraction that takes "everything after the first line."

**Fix:** Explicitly request the output format in the prompt:
```python
"Output ONLY the SQL query, with no explanation or markdown formatting.\n\n"
```

**Pattern:** Always specify output format constraints in prompts when:
- The response will be parsed programmatically
- You need clean data without post-processing
- The model tends to be verbose

**Implementation:** Updated `src/evaluate/benchmark.py::_build_prompt()` to include "no explanation or markdown formatting" instruction.

---

## SQL Extraction from Markdown Code Blocks (2026-04-20)

### Multiple Extraction Strategies Needed
Different models format SQL differently. Robust extraction handles:
1. Clean SQL (no markdown)
2. SQL in ```sql code blocks
3. SQL with closing ``` but additional text after
4. Unclosed code blocks

**Aggressive cleanup approach:**
```python
def extract_sql(response: str) -> str:
    # Split on ``` and take everything before it
    if '```' in response:
        response = response.split('```')[0].strip()
    return response.rstrip(';').strip()
```

**Why this works:** Even if the closing ``` is missing or followed by other content, splitting on ``` gives us the SQL-only portion.

**Pattern:** For LLM-generated code, always handle markdown formatting as potentially malformed. Models may:
- Forget closing ```
- Add text after code blocks
- Use inconsistent formatting (```sql vs ```)
- Mix explanations with code

---

## Teacher-Only Evaluation Strategy (2026-04-20)

### Partial Evaluation is Valid When Resources Are Limited
When hardware constraints prevent full teacher-student comparison, teacher-only evaluation still provides value:

**What you get:**
- Baseline quality metrics (what the ideal output looks like)
- API cost and latency benchmarks
- Validation that data generation and curation worked
- Working pipeline for future full evaluation

**What you miss:**
- Student model quality parity (% of teacher)
- Cost-benefit analysis
- Deployment recommendations

**Valid use case:** "This lightweight version is deliberately intended to prove that everything is working. However, any regular evaluation will require investment in more RAM or cloud GPU."

**Documentation:** Created separate notebooks:
- `08_teacher_evaluation_anthropic.ipynb` - Teacher-only (runs anywhere, ~$0.02)
- `09_student_evaluation.ipynb` - Student-only (requires 16GB+ RAM)
- `10_comparison_from_results.ipynb` - Compare pre-saved results

This split approach allows:
- Progress on limited hardware
- Parallel work (teacher on laptop, student on cloud GPU)
- Modular evaluation pipeline



## Prompt Engineering for Output Format (2026-04-20)

### LLM Models Add Explanations Even When Not Asked
Claude Sonnet 4.6 naturally adds explanation text after SQL responses, even with simple prompts.

**Problem:** Responses include explanation text after SQL:
```sql
SELECT * FROM users;
```

### Explanation:
```

**Fix:** Explicitly request the output format:
```python
"Output ONLY the SQL query, with no explanation or markdown formatting."
```

**Implementation:** Updated `src/evaluate/benchmark.py::_build_prompt()`.

---

## SQL Extraction from Markdown Code Blocks (2026-04-20)

### Aggressive Cleanup Works Best
When models add explanations after code blocks, simple regex extraction fails.

**Robust approach:**
```python
if '```' in response:
    response = response.split('```')[0].strip()
return response.rstrip(';').strip()
```

This handles unclosed blocks and additional text after the closing marker.

---

## Teacher-Only Evaluation Strategy (2026-04-20)

### Partial Evaluation is Valid When Resources Are Limited
When hardware constraints prevent full comparison, teacher-only evaluation still provides:
- Baseline quality metrics
- API cost and latency benchmarks  
- Pipeline validation

**Split approach:**
- `08_teacher_evaluation_anthropic.ipynb` - Teacher-only (~$0.02, runs anywhere)
- `09_student_evaluation.ipynb` - Student-only (requires 16GB+ RAM)
- `10_comparison_from_results.ipynb` - Compare pre-saved results

This allows progress on limited hardware with clear documentation of constraints.
