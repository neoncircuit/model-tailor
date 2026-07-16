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

from src.evaluate.gate import ExecutionGate, GateMetrics
from src.generate.few_shot import SeedLoader
from src.generate.quality import QualityChecker
from src.generate.repair import SQLRepairer
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

    # Execution gate
    run_execution_gate: bool = False
    execution_gate_timeout: float = 5.0
    execution_gate_allow_empty_result: bool = True

    # Repair / fixer
    repair_enabled: bool = False
    repair_max_attempts: int = 2
    repair_batch_size: int = 5

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
        gate_cfg = gen.get("execution_gate", {})
        repair_cfg = gen.get("repair", {})

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
            run_execution_gate=gate_cfg.get("enabled", False),
            execution_gate_timeout=gate_cfg.get("timeout", 5.0),
            execution_gate_allow_empty_result=gate_cfg.get("allow_empty_result", True),
            repair_enabled=repair_cfg.get("enabled", False),
            repair_max_attempts=repair_cfg.get("max_attempts", 2),
            repair_batch_size=repair_cfg.get("repair_batch_size", 5),
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
    total_passed_gate: int = 0
    total_repaired: int = 0
    total_rejected: int = 0
    gate_metrics: dict[str, float] = field(default_factory=dict)
    rejected_path: str | None = None


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

        # Execution gate and repairer are created lazily so that generation
        # can still run when the gate is disabled.
        self._gate: ExecutionGate | None = None
        self._repairer: SQLRepairer | None = None
        self._init_gate_and_repair()

    def _init_gate_and_repair(self) -> None:
        """Create the execution gate and repairer if enabled in config."""
        if self.config.repair_enabled:
            # Repair implies the gate must run.
            self.config.run_execution_gate = True

        if not self.config.run_execution_gate:
            return

        if not self._schema or not self.config.schema_file:
            logger.warning(
                "Execution gate/repair enabled but no schema file provided. "
                "Disabling gate and repair for this run."
            )
            self.config.run_execution_gate = False
            self.config.repair_enabled = False
            return

        self._gate = ExecutionGate(
            schema_sql_path=self.config.schema_file,
            timeout=self.config.execution_gate_timeout,
            allow_empty_result=self.config.execution_gate_allow_empty_result,
        )

        if self.config.repair_enabled:
            self._repairer = SQLRepairer(
                client=self.client,
                gate=self._gate,
                max_attempts=self.config.repair_max_attempts,
            )

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

    def _record_for_gate(self, example: GeneratedExample) -> dict:
        """Build the plain dict the execution gate expects from a GeneratedExample.

        Args:
            example: A generated example.

        Returns:
            Dict with ``natural_language``, ``sql`` and an optional
            ``target_sql`` key.
        """
        record = {
            "natural_language": example.natural_language,
            "sql": example.sql,
        }
        target_sql = example.metadata.get("target_sql") if example.metadata else None
        if target_sql:
            record["target_sql"] = target_sql
        return record

    def _annotate_gate_metadata(
        self,
        example: GeneratedExample,
        final_result: object,
        was_repaired: bool,
        attempts: int,
        original_sql: str | None,
        history: list[dict] | None = None,
    ) -> None:
        """Attach gate/repair metadata to a generated example.

        Args:
            example: The example whose metadata dict will be updated in place.
            final_result: The final ``ExecutionResult`` from the gate.
            was_repaired: Whether a repair was attempted.
            attempts: Number of repair attempts made.
            original_sql: The original SQL before repair, if any.
            history: Optional repair history from the ``SQLRepairer``.
        """
        if example.metadata is None:
            example.metadata = {}
        example.metadata["gate_status"] = final_result.status
        example.metadata["gate_error"] = final_result.error_message
        example.metadata["was_repaired"] = was_repaired
        example.metadata["repair_attempts"] = attempts
        example.metadata["original_sql"] = original_sql
        if history is not None:
            example.metadata["repair_history"] = history

    def _run_gate_and_repair(
        self,
        examples: list[GeneratedExample],
    ) -> tuple[list[GeneratedExample], list[GeneratedExample], GateMetrics]:
        """Apply the execution gate and optional repairer to filtered examples.

        Args:
            examples: Examples that passed quality filtering.

        Returns:
            A tuple of (accepted examples, rejected examples, gate metrics).
            Accepted examples passed the gate either on the first try or after
            a successful repair.
        """
        if self._gate is None:
            # Should not happen because _init_gate_and_repair guards this, but
            # keep the path safe.
            return examples, [], GateMetrics(total=len(examples))

        records = [self._record_for_gate(ex) for ex in examples]
        initial_results = self._gate.check_batch(records)

        metrics = GateMetrics(total=len(examples))
        accepted: list[GeneratedExample] = []
        rejected: list[GeneratedExample] = []

        for example, record, initial_result in zip(examples, records, initial_results):
            if initial_result.passed:
                metrics.passed_first_try += 1
                metrics.final_passed += 1
                self._annotate_gate_metadata(
                    example,
                    initial_result,
                    was_repaired=False,
                    attempts=0,
                    original_sql=None,
                )
                accepted.append(example)
                continue

            metrics.failed_first_try += 1
            final_result = initial_result
            attempts = 0
            original_sql = record.get("sql", "").strip()
            repair_history: list[dict] | None = None

            if self._repairer is not None:
                repair_result = self._repairer.repair(record, schema=self._schema)
                final_result = repair_result.final_result
                attempts = repair_result.attempts
                repair_history = repair_result.history
                metrics.repair_attempts += attempts

                if final_result.passed:
                    metrics.repair_success += 1
                    metrics.final_passed += 1
                    example.sql = repair_result.repaired_sql
                    self._annotate_gate_metadata(
                        example,
                        final_result,
                        was_repaired=True,
                        attempts=attempts,
                        original_sql=original_sql,
                        history=repair_history,
                    )
                    accepted.append(example)
                    continue

                metrics.repair_failed += 1

            metrics.final_failed += 1
            self._annotate_gate_metadata(
                example,
                final_result,
                was_repaired=self._repairer is not None,
                attempts=attempts,
                original_sql=original_sql,
                history=repair_history,
            )
            rejected.append(example)

        for result in initial_results:
            metrics.failure_by_type[result.status] = (
                metrics.failure_by_type.get(result.status, 0) + 1
            )

        return accepted, rejected, metrics

    async def run(self) -> BatchResult:
        """Execute the full generation pipeline.

        1. Plan jobs across strategies and difficulty levels.
        2. Run jobs concurrently with bounded concurrency.
        3. Apply quality filtering.
        4. Apply execution gate and optional repair.
        5. Save accepted and rejected examples to JSONL.

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
            quality_passed = [ex for ex, qr in zip(all_examples, quality_results) if qr.passed]
            logger.info("Quality filter: %d / %d passed", len(quality_passed), total_generated)
        else:
            quality_passed = all_examples

        # Execution gate and optional repair
        if self.config.run_execution_gate:
            accepted, rejected, gate_metrics = self._run_gate_and_repair(quality_passed)
            logger.info(
                "Gate filter: %d accepted, %d rejected (repair success: %d)",
                len(accepted),
                len(rejected),
                gate_metrics.repair_success,
            )
        else:
            accepted = quality_passed
            rejected = []
            gate_metrics = None

        # Save to JSONL
        output_path = self._save_results(accepted)
        rejected_path = self._save_results(rejected, filename="rejected.jsonl")

        duration = time.time() - start_time
        logger.info(
            "Batch generation complete: %d examples in %.1fs, saved to %s",
            len(accepted),
            duration,
            output_path,
        )

        return BatchResult(
            examples=accepted,
            total_generated=total_generated,
            total_passed_quality=len(quality_passed),
            duration_seconds=duration,
            output_path=str(output_path) if output_path else None,
            total_passed_gate=len(accepted),
            total_repaired=gate_metrics.repair_success if gate_metrics else 0,
            total_rejected=len(rejected),
            gate_metrics=gate_metrics.to_flat_dict() if gate_metrics else {},
            rejected_path=str(rejected_path) if rejected_path else None,
        )

    def run_sync(self) -> BatchResult:
        """Synchronous wrapper around run() for convenience.

        Returns:
            BatchResult with all generated and filtered examples.
        """
        return asyncio.run(self.run())

    def _save_results(
        self,
        examples: list[GeneratedExample],
        filename: str | None = None,
    ) -> Path | None:
        """Write generated examples to a JSONL file.

        Args:
            examples: GeneratedExample objects to persist.
            filename: Optional output filename. Defaults to the configured
                ``output_filename``.

        Returns:
            The Path to the written file, or ``None`` if the list was empty.
        """
        if not examples:
            return None

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = filename or self.config.output_filename
        output_path = output_dir / output_filename

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
