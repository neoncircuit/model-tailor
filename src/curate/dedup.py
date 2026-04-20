"""Deduplication utilities for curated datasets.

Provides exact and fuzzy (MinHash-based) deduplication over NL/SQL pairs.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Literal

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

# Type alias for the data records flowing through the pipeline.
Record = dict[str, object]


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters.

    Args:
        text: Input string to tokenize.

    Returns:
        List of lowercased alphanumeric tokens.
    """
    return re.findall(r"\w+", text.lower())


def _record_key(record: Record, field: str) -> str:
    """Return the string value of *field* in *record*, or empty string.

    Args:
        record: A data record dictionary.
        field: Key to look up in the record.

    Returns:
        The string value of the field, or an empty string if absent.
    """
    value = record.get(field, "")
    return value if isinstance(value, str) else str(value)


# ---------------------------------------------------------------------------
# Exact deduplication
# ---------------------------------------------------------------------------


class ExactDedup:
    """Remove records whose target field(s) are exact duplicates.

    Args:
        fields: Which fields to consider when deciding if two records match.
            By default ``["nl", "sql"]`` -- a record is a duplicate only if
            *both* its NL and SQL strings appeared together before.
        keep: Which occurrence to keep when duplicates are found.
            Must be ``"first"`` or ``"last"``.
    """

    def __init__(
        self,
        fields: list[str] | None = None,
        keep: Literal["first", "last"] = "first",
    ) -> None:
        self.fields = fields or ["nl", "sql"]
        self.keep = keep

    # -- helpers ----------------------------------------------------------

    def _fingerprint(self, record: Record) -> str:
        """SHA-256 hex digest of the concatenated field values.

        Args:
            record: A data record dictionary.

        Returns:
            Hex digest string uniquely identifying the record's field values.
        """
        combined = "\x00".join(_record_key(record, f) for f in self.fields)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # -- public API -------------------------------------------------------

    def run(self, data: list[Record]) -> list[Record]:
        """Return *data* with exact duplicates removed.

        Args:
            data: Input records.

        Returns:
            De-duplicated records, original order preserved.
        """
        seen: set[str] = set()
        out: list[Record] = []

        items = data if self.keep == "first" else reversed(data)

        for record in items:
            fp = self._fingerprint(record)
            if fp not in seen:
                seen.add(fp)
                out.append(record)

        if self.keep == "last":
            out.reverse()

        n_removed = len(data) - len(out)
        logger.info("ExactDedup: removed %d exact duplicates from %d records", n_removed, len(data))
        return out


# ---------------------------------------------------------------------------
# Fuzzy deduplication (MinHash + LSH)
# ---------------------------------------------------------------------------


class FuzzyDedup:
    """MinHash-based fuzzy deduplication using Jaccard similarity.

    Args:
        field: The record field to compare (default ``"nl"``).
        threshold: Jaccard similarity threshold above which two records are
            considered duplicates (default ``0.85``).
        num_perm: Number of permutations for MinHash (default ``128``).
            Higher values give more accurate estimates at the cost of memory.

    Raises:
        ValueError: If *threshold* is not in the interval ``(0, 1]``.
    """

    def __init__(
        self,
        field: str = "nl",
        threshold: float = 0.85,
        num_perm: int = 128,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.field = field
        self.threshold = threshold
        self.num_perm = num_perm

    # -- helpers ----------------------------------------------------------

    def _make_minhash(self, text: str) -> MinHash:
        """Build a MinHash signature for *text*.

        Args:
            text: Input string to hash.

        Returns:
            A MinHash object representing the token-level signature.
        """
        m = MinHash(num_perm=self.num_perm)
        for token in _tokenize(text):
            m.update(token.encode("utf-8"))
        return m

    # -- public API -------------------------------------------------------

    def run(self, data: list[Record]) -> list[Record]:
        """Return *data* with near-duplicates removed.

        The first occurrence (by input order) of each cluster is kept.

        Args:
            data: Input records.

        Returns:
            De-duplicated records, original order preserved.
        """
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        signatures: list[MinHash] = []
        keep_flags: list[bool] = [True] * len(data)

        for idx, record in enumerate(data):
            text = _record_key(record, self.field)
            mh = self._make_minhash(text)
            signatures.append(mh)

            # Query LSH *before* inserting so we only compare against earlier records.
            candidates = lsh.query(mh)
            if candidates:
                # This record is a near-duplicate of an earlier one -- mark for removal.
                keep_flags[idx] = False
            else:
                # Use the index as the key (must be str or bytes for datasketch).
                lsh.insert(str(idx), mh)

        out = [rec for rec, keep in zip(data, keep_flags) if keep]
        n_removed = len(data) - len(out)
        logger.info(
            "FuzzyDedup(field=%s, threshold=%.2f): removed %d near-duplicates from %d records",
            self.field,
            self.threshold,
            n_removed,
            len(data),
        )
        return out
