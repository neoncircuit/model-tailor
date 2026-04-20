"""Synthetic training data generation module.

Generates NL->SQL training pairs using a teacher LLM (GPT-4) through
multiple prompting strategies, with few-shot management, quality control,
and batched parallel execution.

Typical usage:
    from src.generate.batch import run_generation
    result = run_generation("config/tasks/sql_generation.yaml")

Or component-by-component:
    from src.generate.strategies import SeedExpansion, SelfInstruct, EvolInstruct
    from src.generate.few_shot import SeedLoader, FewShotFormatter
    from src.generate.quality import QualityChecker
    from src.generate.batch import BatchGenerator
"""

from src.generate.batch import BatchConfig, BatchGenerator, BatchResult, run_generation
from src.generate.few_shot import FewShotFormatter, SeedLoader
from src.generate.quality import QualityChecker, QualityResult
from src.generate.strategies import (
    EvolInstruct,
    GeneratedExample,
    GenerationStrategy,
    SeedExpansion,
    SelfInstruct,
    get_strategy,
)

__all__ = [
    # Strategies
    "GenerationStrategy",
    "SeedExpansion",
    "SelfInstruct",
    "EvolInstruct",
    "GeneratedExample",
    "get_strategy",
    # Few-shot
    "SeedLoader",
    "FewShotFormatter",
    # Quality
    "QualityChecker",
    "QualityResult",
    # Batch
    "BatchGenerator",
    "BatchConfig",
    "BatchResult",
    "run_generation",
]
