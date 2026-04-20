# model-tailor Status Report (2026-04-18)

## Current Status

**Milestone 1 (First Dataset): COMPLETE**
- Task 1 (Dry-run): Complete - 10 examples generated successfully
- Task 2 (Full generation): Complete - 146 examples generated
- Task 3 (Curation): Complete - Deduplicated, filtered, balanced, and split into train/val/test
- Task 4 (Formatting): Complete - Llama 3.1 chat template applied, validated for 2048 token limit

**Milestone 2 (First Fine-Tune): COMPLETE**
- Task 5 (Training): COMPLETE - QLoRA training finished successfully
  - Dataset: 123 train examples, 16 val examples, 14 test examples
  - Config: batch_size=1, max_seq_length=2048, gradient_accumulation=8, 1 epoch
  - Result: LoRA adapter saved (161MB, 41.9M params, 0.52% of 8B model)
  - VRAM: Stable at 97% (7.7GB/8GB) - no OOM errors
  - Training time: 11 minutes

**Milestone 3 (Evaluation): READY TO START**
- Task 6 (Evaluation): Two comparison notebooks ready
  - `notebooks/07_comparison_glm.ipynb` - For GLM API users (no Anthropic key needed)
  - `notebooks/07_comparison_anthropic.ipynb` - For Anthropic API users
  - Both compare teacher vs student with quality metrics, cost analysis, and recommendations

## Recent Accomplishments (2026-04-18)

1. **Training Completed Successfully** - `scripts/train_simple.py`
   - Fixed data loading to handle text format in JSONL
   - Added labels field for causal language modeling
   - Conservative settings (batch_size=1) to prevent OOM on 8GB VRAM
   - Model saved to `models/sql-llama-8b-lora/`

2. **Comparison Notebooks Created** - Two provider-specific versions
   - `notebooks/07_comparison_glm.ipynb` - GLM 4.6 vs Llama (GLM API only)
   - `notebooks/07_comparison_anthropic.ipynb` - Sonnet 4.6 vs Llama (Anthropic API)
   - Both include quality metrics, LLM judge evaluation, cost analysis, visualizations

3. **GLM 4.7 Integration** - `src/llm/client.py`
   - Added GLM provider support
   - Updated config to use GLM 4.7 as teacher/judge model
   - OpenAI-compatible API integration

4. **Safety Improvements**
   - Added BSoD prevention feedback to memory
   - Conservative VRAM settings to avoid system crashes
   - Proper process cleanup

## API Usage Status

### GLM 4.7 (Zhipu AI)
- **Current usage**: Teacher model for evaluation (Task 6)
- **Config**: `config/base.yaml` - provider=glm, model=glm-4.7
- **Ready**: Comparison notebook configured to use GLM as judge

### Local Inference
- **Student model**: Llama 3.1 8B + LoRA adapter
- **Location**: `models/sql-llama-8b-lora/`
- **Cost**: $0.00 per query (after one-time training cost)

## Data Summary

| File | Records | Size | Status |
|------|---------|------|--------|
| `data/raw/generated.jsonl` | 146 | 127 KB | Complete |
| `data/curated/train.jsonl` | 123 | 57 KB | Complete |
| `data/curated/val.jsonl` | 16 | 8.6 KB | Complete |
| `data/curated/test.jsonl` | 14 | 6.4 KB | Complete |
| `data/formatted/train.jsonl` | 123 | 90 KB | Complete |
| `data/formatted/val.jsonl` | 16 | 13 KB | Complete |

## Model Summary

| Component | Details |
|-----------|---------|
| Base Model | unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit |
| LoRA Adapter | 41.9M params (0.52%) |
| Adapter Size | 161 MB |
| Training Data | 123 examples (1 epoch) |
| Training Time | 11 minutes |
| Peak VRAM | 7.7 GB / 8 GB (97%) |

## Next Steps

1. **Run comparison notebook** - Choose based on your API access:
   - GLM users: `notebooks/07_comparison_glm.ipynb`
   - Anthropic users: `notebooks/07_comparison_anthropic.ipynb`
   - Compares teacher vs student on test set
   - Generates quality metrics and cost analysis

2. **Review results** - Analyze quality parity and cost-effectiveness
   - Target: >75% quality retention vs teacher
   - Calculate break-even point for deployment

3. **Decide on deployment** (Milestone 3)
   - Task 7: Export and quantize to GGUF
   - Task 8: Deploy FastAPI server for inference

## Files Updated This Session

- `scripts/train_simple.py` - Created, fixed data loading and labels
- `notebooks/07_comparison_glm.ipynb` - GLM-specific comparison notebook
- `notebooks/07_comparison_anthropic.ipynb` - Anthropic-specific comparison notebook
- `src/llm/client.py` - Added GLM provider support
- `config/base.yaml` - Updated to use GLM 4.7
- `tasks/todo.md` - Updated task statuses with both notebook options
- `tasks/lessons.md` - Added system stability lessons
- `memory/feedback_bsod_risk.md` - Added BSoD prevention feedback
- `docs/notebook-guide.md` - Created comprehensive guide for notebook usage
- `docs/documentation-update-2026-04-18.md` - Comprehensive update summary
- `docs/configuration.md` - Added GLM provider documentation
- `README.md` - Added notebook guide as first documentation link
- `requirements.txt` - Added visualization libraries
- `home/jamestjy/.claude/plans/quiet-dancing-quilt.md` - Marked training complete
