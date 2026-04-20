"""Train / validation / test splitting for curated datasets.

Provides ``DatasetSplitter`` with configurable ratios, optional stratified
splitting that preserves category distributions, and JSONL persistence.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

Record = dict[str, object]

# Default split proportions.
DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


class DatasetSplitter:
    """Split a dataset into train / val / test (or arbitrary named splits).

    Args:
        ratios: Mapping of split name to proportion. Defaults to
            ``{"train": 0.8, "val": 0.1, "test": 0.1}``. Values are
            normalised to sum to 1.
        stratify_field: If given, perform stratified splitting so each split
            preserves the distribution of this field (e.g. ``"difficulty"``).
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If *ratios* sum to a non-positive number.

    Example::

        >>> splitter = DatasetSplitter(stratify_field="difficulty")
        >>> splits = splitter.run(data)
        >>> splitter.save(splits, "data/curated")
    """

    def __init__(
        self,
        ratios: dict[str, float] | None = None,
        stratify_field: str | None = None,
        seed: int | None = 42,
    ) -> None:
        raw = ratios or dict(DEFAULT_RATIOS)
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("ratios must sum to a positive number")
        self.ratios: dict[str, float] = {k: v / total for k, v in raw.items()}
        self.split_names: list[str] = list(self.ratios.keys())
        self.stratify_field = stratify_field
        self.seed = seed

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _distribute(records: list[Record], proportions: list[float]) -> list[list[Record]]:
        """Partition *records* into sub-lists according to *proportions*.

        The last bucket absorbs any rounding remainder so no records are lost.

        Args:
            records: The records to partition.
            proportions: Desired fraction for each bucket (should sum to 1).

        Returns:
            List of sub-lists, one per proportion entry.
        """
        n = len(records)
        buckets: list[list[Record]] = []
        start = 0
        for i, p in enumerate(proportions):
            if i == len(proportions) - 1:
                # Last bucket gets everything that is left.
                buckets.append(records[start:])
            else:
                end = start + round(p * n)
                buckets.append(records[start:end])
                start = end
        return buckets

    # -- public API -------------------------------------------------------

    def run(self, data: list[Record]) -> dict[str, list[Record]]:
        """Split *data* and return a dict of split-name -> records.

        Args:
            data: Full dataset to split.

        Returns:
            Dictionary keyed by split name (e.g. ``"train"``, ``"val"``,
            ``"test"``), with each value being the list of records in
            that split.
        """
        rng = random.Random(self.seed)
        proportions = [self.ratios[name] for name in self.split_names]

        if self.stratify_field is None:
            # Simple random split.
            shuffled = list(data)
            rng.shuffle(shuffled)
            buckets = self._distribute(shuffled, proportions)
            splits = {name: bucket for name, bucket in zip(self.split_names, buckets)}
        else:
            # Stratified split: partition within each category, then merge.
            groups: dict[str, list[Record]] = defaultdict(list)
            for rec in data:
                key = str(rec.get(self.stratify_field, "__unknown__"))
                groups[key].append(rec)

            # Initialise empty splits.
            splits: dict[str, list[Record]] = {name: [] for name in self.split_names}

            for cat, recs in sorted(groups.items()):
                rng.shuffle(recs)
                buckets = self._distribute(recs, proportions)
                for name, bucket in zip(self.split_names, buckets):
                    splits[name].extend(bucket)

            # Shuffle each split so categories are interleaved.
            for name in self.split_names:
                rng.shuffle(splits[name])

        for name in self.split_names:
            logger.info("DatasetSplitter: %s = %d records", name, len(splits[name]))

        return splits

    # -- persistence ------------------------------------------------------

    def save(
        self,
        splits: dict[str, list[Record]],
        output_dir: str | Path,
        prefix: str = "",
    ) -> dict[str, Path]:
        """Write each split as a JSONL file under *output_dir*.

        Args:
            splits: Output of :meth:`run`.
            output_dir: Directory to write into (created if it does not exist).
            prefix: Optional filename prefix (e.g. ``"curated_"``).

        Returns:
            Mapping of split name to the written file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for name, records in splits.items():
            filename = f"{prefix}{name}.jsonl" if prefix else f"{name}.jsonl"
            filepath = out / filename
            with open(filepath, "w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            paths[name] = filepath
            logger.info("DatasetSplitter: wrote %d records to %s", len(records), filepath)

        return paths

    # -- diagnostics ------------------------------------------------------

    @staticmethod
    def stats(splits: dict[str, list[Record]], field: str | None = None) -> dict[str, object]:
        """Return a summary of split sizes and optional category distributions.

        Args:
            splits: Output of :meth:`run`.
            field: If given, also report per-split distribution of this field.

        Returns:
            Summary statistics including total count, per-split counts and
            ratios, and optional per-split category distributions.
        """
        total = sum(len(recs) for recs in splits.values())
        summary: dict[str, object] = {
            "total": total,
            "splits": {},
        }

        for name, records in splits.items():
            info: dict[str, object] = {
                "count": len(records),
                "ratio": round(len(records) / total, 4) if total else 0.0,
            }
            if field is not None:
                counts = Counter(str(r.get(field, "__unknown__")) for r in records)
                n = len(records)
                info["distribution"] = {
                    cat: {"count": c, "ratio": round(c / n, 4) if n else 0.0}
                    for cat, c in sorted(counts.items())
                }
            summary["splits"][name] = info

        return summary
