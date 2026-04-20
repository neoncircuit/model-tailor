# Documentation Update Summary (2026-04-18)

## Overview

Comprehensive documentation update to reflect the current state of the model-tailor project after completing Task 5 (Training) and preparing for Task 6 (Evaluation).

---

## Files Updated

### Core Configuration

1. **`requirements.txt`**
   - Added visualization dependencies: `matplotlib>=3.9.0`, `seaborn>=0.13.0`
   - Required for comparison notebooks

2. **`config/base.yaml`**
   - Already configured with `provider: glm` and `model: glm-4.7`
   - Conservative VRAM settings: `batch_size: 2`, `max_seq_length: 2048`
   - Safe for RTX 3070 Ti (8.6 GB VRAM)

### Documentation Updates

3. **`docs/configuration.md`**
   - Updated `teacher.provider` options to include `glm`
   - Added GLM provider documentation section
   - Added `GLM_API_KEY` to environment variables table

4. **`docs/status-report-2026-04-18.md`**
   - Updated training status to COMPLETE
   - Documented comparison notebooks (GLM and Anthropic versions)
   - Updated "Files Updated This Session" section
   - Updated "Next Steps" with both notebook options

5. **`tasks/lessons.md`**
   - Added "System Stability (2026-04-18)" section
   - Documented BSoD prevention from concurrent training processes
   - Added causal LM data format requirements (labels field)
   - Added VRAM management guidelines for RTX 3070 Ti

6. **`tasks/todo.md`**
   - Updated Task 5 to COMPLETE
   - Updated Task 6 with both notebook options (GLM/Anthropic)
   - Added clear provider requirements for each notebook

### New Files Created

7. **`notebooks/07_comparison_glm.ipynb`**
   - GLM 4.6 as teacher model
   - GLM 4.7 as LLM judge
   - GLM-specific cost analysis
   - Saves to `comparison_results_glm.json`

8. **`notebooks/07_comparison_anthropic.ipynb`**
   - Claude Sonnet 4.6 as teacher model
   - GLM 4.7 as LLM judge
   - Sonnet-specific cost analysis
   - Saves to `comparison_results_anthropic.json`

### Memory Updates

9. **`memory/feedback_bsod_risk.md`**
   - New memory file documenting BSoD risk
   - Rules for preventing concurrent training processes
   - VRAM safety guidelines

10. **`memory/MEMORY.md`**
    - Added index entry for BSoD risk feedback

### Plan Updates

11. **`home/jamestjy/.claude/plans/quiet-dancing-quilt.md`**
    - Marked training as COMPLETE
    - Updated next steps for evaluation

---

## Current Project State

### Completed (Milestone 1 & 2)
- Task 1: Dry-run generation (10 examples) ✓
- Task 2: Full generation (146 examples) ✓
- Task 3: Dataset curation ✓
- Task 4: Formatting for training ✓
- Task 5: Training (QLoRA complete) ✓

### Ready to Start (Milestone 3)
- Task 6: Evaluation (two notebooks ready)
- Task 7: Export and quantize
- Task 8: Deploy and serve

### Model Artifacts
- **Student Model**: `models/sql-llama-8b-lora/`
  - Adapter weights: 161 MB
  - Trainable params: 41.9M (0.52% of 8B)
  - Training time: 11 minutes
  - Peak VRAM: 7.7GB / 8GB (97%)

### Dataset
- Train: 123 examples
- Val: 16 examples
- Test: 14 examples
- All formatted for Llama 3.1 chat template

---

## Provider Configuration

### Current (GLM)
```yaml
teacher:
  provider: glm
  model: glm-4.7
```

### Alternative (Anthropic)
```yaml
teacher:
  provider: anthropic
  model: claude-sonnet-4-6
```

### Supported Providers
- `openai` - GPT-4o, GPT-4o-mini
- `anthropic` - Claude Sonnet 4.6, Claude Haiku 4.5
- `gemini` - Gemini 2.0 Flash, Gemini 2.5 Pro
- `glm` - GLM 4.6, GLM 4.7, GLM 4 Plus
- `ollama` - Local models (free)

---

## Next Steps for User

1. Choose evaluation notebook based on API access:
   - GLM: `notebooks/07_comparison_glm.ipynb`
   - Anthropic: `notebooks/07_comparison_anthropic.ipynb`

2. Run the notebook to:
   - Compare teacher vs student quality
   - Analyze cost-effectiveness
   - Determine break-even point
   - Get deployment recommendations

3. Based on results, proceed to:
   - Task 7: Export and quantize model
   - Task 8: Deploy FastAPI server

---

## Verification Checklist

- [x] All documentation updated with GLM support
- [x] Requirements.txt includes visualization libraries
- [x] Configuration documentation includes GLM provider
- [x] Lessons learned include system stability guidelines
- [x] Todo.md reflects current project state
- [x] Comparison notebooks separated by provider
- [x] Memory updated with BSoD prevention
- [x] Status report current
- [x] No stale references to old configurations

---

## Notes

- `.env.example` already includes `GLM_API_KEY`
- `.gitignore` properly excludes all artifacts
- All provider switches documented in configuration.md
- Training script (`scripts/train_simple.py`) ready for future runs
- BSoD risk documented for future reference
