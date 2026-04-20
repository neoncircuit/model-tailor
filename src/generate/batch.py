"""Batching and parallel execution for synthetic data generation.

Orchestrates multiple generation strategies, runs them concurrently,
applies quality filtering, and saves results to JSONL files.

Usage:
    client = TeacherClient()
    generator = BatchGenerator(client, config_path="config/tasks/sql_generation.yaml")
    results = asyncio.run(generator.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from tqdm import tqdm

from src.generate.few_shot import SeedLoader
from src.generate.quality import QualityChecker
from src.generate.strategies import (
    GeneratedExample,
    get_strategy,
)
from src.llm.client import TeacherClient

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for a batch generation run."""

    num_examples: int = 5000
    batch_size: int = 10
    max_concurrency: int = 5
    strategies: list[str] = field(default_factory=lambda: ["seed_expansion", "self_instruct"])
    strategy_weights: dict[str, float] = field(default_factory=dict)
    difficulty_distribution: dict[str, float] = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.4, "hard": 0.2, "expert": 0.1}
    )
    seed_file: str = ""
    schema_file: str = ""
    output_dir: str = "data/raw"
    output_filename: str = "generated.jsonl"
    run_quality_check: bool = True
    temperature: float = 0.7

    @classmethod
    def from_yaml(cls, config_path: str) -> BatchConfig:
        """Load configuration from a task YAML file.

        Args:
            config_path: Path to a YAML file with 'generation' and
                optional 'curation' sections.

        Returns:
            A populated BatchConfig instance.
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        gen = cfg.get("generation", {})
        curation = cfg.get("curation", {})

        return cls(
            num_examples=gen.get("num_examples", 5000),
            batch_size=gen.get("batch_size", 10),
            max_concurrency=gen.get("max_concurrency", 5),
            strategies=gen.get("strategies", ["seed_expansion", "self_instruct"]),
            strategy_weights=gen.get("strategy_weights", {}),
            difficulty_distribution=gen.get(
                "difficulty_distribution",
                {"easy": 0.3, "medium": 0.4, "hard": 0.2, "expert": 0.1},
            ),
            seed_file=gen.get("seed_file", ""),
            schema_file=gen.get("schema_file", ""),
            output_dir=gen.get("output_dir", "data/raw"),
            output_filename=gen.get("output_filename", "generated.jsonl"),
            run_quality_check=curation.get("require_valid_sql", True),
            temperature=gen.get("temperature", 0.7),
        )


@dataclass
class GenerationJob:
    """A single generation task to be executed."""

    strategy_name: str
    num_examples: int
    difficulty: str
    schema: str
    seeds: list[dict] | None = None


@dataclass
class BatchResult:
    """Results from a complete batch generation run."""

    examples: list[GeneratedExample]
    total_generated: int
    total_passed_quality: int
    duration_seconds: float
    output_path: str | None = None


class BatchGenerator:
    """Orchestrates generation across strategies with concurrency and quality control.

    Splits the target number of examples across strategies and difficulty
    levels, runs generation jobs concurrently using asyncio, applies
    quality filtering, and writes results to JSONL.

    Args:
        client: TeacherClient instance for LLM calls.
        config: BatchConfig or path to a YAML config file.
        quality_checker: Optional pre-configured QualityChecker.
    """

    def __init__(
        self,
        client: TeacherClient,
        config: BatchConfig | str | None = None,
        quality_checker: QualityChecker | None = None,
    ) -> None:
        """Initialize the batch generator.

        Args:
            client: TeacherClient instance for LLM calls.
            config: A BatchConfig object, a path to a YAML config file, or
                ``None`` to use defaults.
            quality_checker: Optional pre-configured QualityChecker. When
                ``None``, a default checker is created from the client.
        """
        self.client = client

        if isinstance(config, str):
            self.config = BatchConfig.from_yaml(config)
        elif config is not None:
            self.config = config
        else:
            self.config = BatchConfig()

        self.quality_checker = quality_checker or QualityChecker(
            client=client,
            run_llm_check=self.config.run_quality_check,
        )

        # Load schema
        self._schema = ""
        if self.config.schema_file:
            schema_path = Path(self.config.schema_file)
            if schema_path.exists():
                self._schema = schema_path.read_text(encoding="utf-8")
            else:
                logger.warning("Schema file not found: %s", schema_path)

        # Load seeds
        self._seeds: list[dict] = []
        if self.config.seed_file:
            seed_path = Path(self.config.seed_file)
            if seed_path.exists():
                loader = SeedLoader(seed_path)
                loader.load()
                self._seeds = loader.to_dicts()
                logger.info("Loaded %d seed examples", len(self._seeds))
            else:
                logger.warning("Seed file not found: %s", seed_path)

    def _plan_jobs(self) -> list[GenerationJob]:
        """Split total examples into individual jobs by strategy and difficulty.

        Returns:
            A list of GenerationJob objects covering the configured strategies
            and difficulty distribution.
        """
        jobs: list[GenerationJob] = []
        total = self.config.num_examples

        # Determine how many examples each strategy should produce
        strategies = self.config.strategies
        weights = self.config.strategy_weights

        if weights:
            # Use explicit weights
            weight_sum = sum(weights.get(s, 1.0) for s in strategies)
            strategy_counts = {
                s: max(1, int(total * weights.get(s, 1.0) / weight_sum)) for s in strategies
            }
        else:
            # Split evenly
            per_strategy = total // len(strategies)
            strategy_counts = {s: per_strategy for s in strategies}
            # Give remainder to the first strategy
            remainder = total - sum(strategy_counts.values())
            if remainder > 0:
                strategy_counts[strategies[0]] += remainder

        # Split each strategy's count by difficulty
        diff_dist = self.config.difficulty_distribution
        for strategy_name, count in strategy_counts.items():
            for difficulty, weight in diff_dist.items():
                n = max(1, int(count * weight))
                jobs.append(
                    GenerationJob(
                        strategy_name=strategy_name,
                        num_examples=n,
                        difficulty=difficulty,
                        schema=self._schema,
                        seeds=self._seeds if self._seeds else None,
                    )
                )

        logger.info(
            "Planned %d jobs across %d strategies and %d difficulty levels",
            len(jobs),
            len(strategies),
            len(diff_dist),
        )
        return jobs

    async def _run_job(
        self,
        job: GenerationJob,
        semaphore: asyncio.Semaphore,
        progress: tqdm,
    ) -> list[GeneratedExample]:
        """Run a single generation job in a thread pool.

        LLM calls are synchronous, so each job is offloaded to a thread via
        ``run_in_executor``.

        Args:
            job: The generation job to execute.
            semaphore: Asyncio semaphore controlling maximum concurrency.
            progress: tqdm progress bar to update after each batch.

        Returns:
            A list of GeneratedExample objects produced by the job. Returns
            an empty list if the job raises an exception.
        """
        async with semaphore:
            strategy = get_strategy(
                job.strategy_name,
                client=self.client,
                temperature=self.config.temperature,
            )

            # Run the synchronous strategy.generate() in a thread pool
            loop = asyncio.get_event_loop()
            try:
                examples = await loop.run_in_executor(
                    None,
                    lambda: strategy.generate(
                        num_examples=job.num_examples,
                        schema=job.schema,
                        seeds=job.seeds,
                        difficulty=job.difficulty,
                    ),
                )
            except Exception:
                logger.exception(
                    "Job failed: strategy=%s difficulty=%s",
                    job.strategy_name,
                    job.difficulty,
                )
                examples = []

            progress.update(len(examples))
            return examples

    async def run(self) -> BatchResult:
        """Execute the full generation pipeline.

        1. Plan jobs across strategies and difficulty levels.
        2. Run jobs concurrently with bounded concurrency.
        3. Apply quality filtering.
        4. Save results to JSONL.

        Returns:
            BatchResult with all generated and filtered examples.
        """
        start_time = time.time()

        jobs = self._plan_jobs()
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        progress = tqdm(
            total=self.config.num_examples,
            desc="Generating examples",
            unit="ex",
        )

        # Launch all jobs
        tasks = [self._run_job(job, semaphore, progress) for job in jobs]
        results_nested = await asyncio.gather(*tasks)
        progress.close()

        # Flatten results
        all_examples: list[GeneratedExample] = []
        for batch in results_nested:
            all_examples.extend(batch)

        total_generated = len(all_examples)
        logger.info("Generated %d raw examples", total_generated)

        # Quality filtering
        if self.config.run_quality_check:
            logger.info("Running quality checks...")
            example_dicts = [
                {"natural_language": ex.natural_language, "sql": ex.sql} for ex in all_examples
            ]
            quality_results = self.quality_checker.check_batch(example_dicts, schema=self._schema)
            filtered = [ex for ex, qr in zip(all_examples, quality_results) if qr.passed]
            logger.info("Quality filter: %d / %d passed", len(filtered), total_generated)
        else:
            filtered = all_examples

        # Save to JSONL
        output_path = self._save_results(filtered)

        duration = time.time() - start_time
        logger.info(
            "Batch generation complete: %d examples in %.1fs, saved to %s",
            len(filtered),
            duration,
            output_path,
        )

        return BatchResult(
            examples=filtered,
            total_generated=total_generated,
            total_passed_quality=len(filtered),
            duration_seconds=duration,
            output_path=str(output_path) if output_path else None,
        )

    def run_sync(self) -> BatchResult:
        """Synchronous wrapper around run() for convenience.

        Returns:
            BatchResult with all generated and filtered examples.
        """
        return asyncio.run(self.run())

    def _save_results(self, examples: list[GeneratedExample]) -> Path | None:
        """Write generated examples to a JSONL file.

        Args:
            examples: GeneratedExample objects to persist.

        Returns:
            The Path to the written file, or ``None`` if the list was empty.
        """
        if not examples:
            logger.warning("No examples to save")
            return None

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.config.output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "natural_language": ex.natural_language,
                    "sql": ex.sql,
                    "difficulty": ex.difficulty,
                    "category": ex.category,
                    "strategy": ex.strategy,
                }
                if ex.metadata:
                    record["metadata"] = ex.metadata
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved %d examples to %s", len(examples), output_path)
        return output_path


def run_generation(
    config_path: str = "config/tasks/sql_generation.yaml",
    base_config_path: str = "config/base.yaml",
) -> BatchResult:
    """Top-level entry point: load config, create client, run batch generation.

    Args:
        config_path: Path to task-specific generation config YAML.
        base_config_path: Path to base config with teacher model settings.

    Returns:
        BatchResult with all generated examples and metadata.
    """
    client = TeacherClient(config_path=base_config_path)
    config = BatchConfig.from_yaml(config_path)
    generator = BatchGenerator(client=client, config=config)
    return generator.run_sync()
