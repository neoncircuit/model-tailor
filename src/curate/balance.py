"""Distribution balancing for curated datasets.

Adjusts the number of examples per category (e.g. difficulty level, SQL type)
so the final dataset follows a target distribution, using upsampling
(duplicating underrepresented records) and/or downsampling (trimming
overrepresented records).
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from typing import Literal

logger = logging.getLogger(__name__)

Record = dict[str, object]


class DatasetBalancer:
    """Balance a dataset across categories.

    Args:
        field: The record field that holds the category label
            (e.g. ``"difficulty"`` or ``"category"``).
        target: Desired distribution as category -> proportion. Values must be
            non-negative and will be normalised to sum to 1.
        strategy: Resampling strategy. ``"upsample"`` only duplicates
            underrepresented records (total size can grow). ``"downsample"``
            only trims overrepresented records (total size can shrink).
            ``"both"`` applies both strategies to hit the target proportions
            as closely as possible.
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If *target* is ``None`` or its proportions do not sum to
            a positive number.

    Example::

        >>> balancer = DatasetBalancer(
        ...     field="difficulty",
        ...     target={"easy": 0.3, "medium": 0.4, "hard": 0.2, "expert": 0.1},
        ... )
        >>> balanced = balancer.run(data)
    """

    def __init__(
        self,
        field: str = "difficulty",
        target: dict[str, float] | None = None,
        strategy: Literal["upsample", "downsample", "both"] = "both",
        seed: int | None = 42,
    ) -> None:
        self.field = field
        self.strategy = strategy
        self.seed = seed

        if target is None:
            raise ValueError("A target distribution dict is required")

        total = sum(target.values())
        if total <= 0:
            raise ValueError("target proportions must sum to a positive number")
        # Normalise so proportions sum to 1.
        self.target: dict[str, float] = {k: v / total for k, v in target.items()}

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _group_by(data: list[Record], field: str) -> dict[str, list[Record]]:
        """Group records by the value of a given field.

        Args:
            data: Input records.
            field: Record field whose value is used as the grouping key.

        Returns:
            Mapping of category string to the list of records in that category.
        """
        groups: dict[str, list[Record]] = defaultdict(list)
        for rec in data:
            key = str(rec.get(field, "__unknown__"))
            groups[key].append(rec)
        return dict(groups)

    def _compute_target_counts(
        self,
        groups: dict[str, list[Record]],
    ) -> dict[str, int]:
        """Decide how many records each category should have after balancing.

        Args:
            groups: Mapping of category string to its current records.

        Returns:
            Mapping of category string to the desired record count.
        """
        current_counts = {cat: len(recs) for cat, recs in groups.items()}
        n_total = sum(current_counts.values())

        if self.strategy == "downsample":
            # The bottleneck is the category most over-represented relative
            # to the target.  Scale so no category needs upsampling.
            # Find the smallest n such that for every category:
            #   target[cat] * n <= current_counts[cat]
            ratios = []
            for cat, prop in self.target.items():
                if prop > 0 and cat in current_counts:
                    ratios.append(current_counts[cat] / prop)
            if not ratios:
                return current_counts
            scale = min(ratios)
            return {
                cat: min(current_counts.get(cat, 0), max(1, round(prop * scale)))
                for cat, prop in self.target.items()
                if cat in current_counts
            }

        elif self.strategy == "upsample":
            # Scale up so the smallest category (relative to target) stays
            # at its current size -- everything else gets upsampled.
            ratios = []
            for cat, prop in self.target.items():
                if prop > 0 and cat in current_counts:
                    ratios.append(current_counts[cat] / prop)
            if not ratios:
                return current_counts
            scale = max(ratios)
            return {
                cat: max(current_counts.get(cat, 0), max(1, round(prop * scale)))
                for cat, prop in self.target.items()
                if cat in current_counts
            }

        else:  # "both"
            # Use total dataset size and distribute according to target.
            return {
                cat: max(1, round(prop * n_total))
                for cat, prop in self.target.items()
                if cat in current_counts
            }

    @staticmethod
    def _resize_group(
        records: list[Record],
        target_n: int,
        rng: random.Random,
    ) -> list[Record]:
        """Return exactly *target_n* records from *records*.

        If ``target_n <= len(records)`` samples without replacement
        (downsample). If ``target_n > len(records)`` keeps all originals
        then samples with replacement to fill the gap (upsample).

        Args:
            records: The records belonging to one category.
            target_n: Desired number of records after resizing.
            rng: Random number generator for reproducibility.

        Returns:
            A list of exactly *target_n* records.
        """
        n = len(records)
        if target_n <= 0:
            return []
        if target_n <= n:
            return rng.sample(records, target_n)
        # Upsampling: keep all originals + random duplicates.
        extras = rng.choices(records, k=target_n - n)
        out = list(records) + extras
        rng.shuffle(out)
        return out

    # -- public API -------------------------------------------------------

    def run(self, data: list[Record]) -> list[Record]:
        """Balance *data* and return the resampled list.

        Records whose category is not listed in ``target`` are **dropped**
        with a warning.

        Args:
            data: Input records.

        Returns:
            Balanced dataset.
        """
        rng = random.Random(self.seed)
        groups = self._group_by(data, self.field)

        # Warn about categories present in data but absent from target.
        unknown = set(groups) - set(self.target)
        if unknown:
            count_unknown = sum(len(groups[c]) for c in unknown)
            logger.warning(
                "DatasetBalancer: dropping %d records with categories not in target: %s",
                count_unknown,
                unknown,
            )

        target_counts = self._compute_target_counts(groups)

        result: list[Record] = []
        for cat, desired in target_counts.items():
            original = groups.get(cat, [])
            resized = self._resize_group(original, desired, rng)
            result.extend(resized)
            logger.info("DatasetBalancer [%s]: %d -> %d records", cat, len(original), len(resized))

        rng.shuffle(result)
        logger.info(
            "DatasetBalancer: total %d -> %d records (strategy=%s)",
            len(data),
            len(result),
            self.strategy,
        )
        return result

    # -- diagnostics ------------------------------------------------------

    def report(self, data: list[Record]) -> dict[str, object]:
        """Return a summary dict describing the category distribution.

        Useful for inspecting data before and after balancing.

        Args:
            data: Records to summarize.

        Returns:
            Dictionary with keys ``"field"``, ``"total"``, ``"counts"``,
            ``"distribution"``, and ``"target"``.
        """
        counts = Counter(str(rec.get(self.field, "__unknown__")) for rec in data)
        total = sum(counts.values())
        distribution = {cat: round(n / total, 4) if total else 0.0 for cat, n in counts.items()}
        return {
            "field": self.field,
            "total": total,
            "counts": dict(counts),
            "distribution": distribution,
            "target": self.target,
        }
