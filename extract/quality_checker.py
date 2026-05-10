"""
QualityChecker - Extraction result quality assessment and deduplication.

Pure-Python module with no external dependencies and no LLM calls.
Used by ExtractionEngine to clean up extraction results before CSV output.
"""

from typing import List, Dict


# Values treated as "empty" when counting field fill rate
EMPTY_MARKERS = {"无", "未提及", "N/A", "-", "--"}


def _is_empty(value) -> bool:
    """Check if a value should be treated as empty/missing."""
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s in EMPTY_MARKERS


class QualityChecker:
    """Assesses extraction record quality and detects duplicates.

    Designed to be instantiated once and reused across multiple extraction
    batches.  All methods are pure functions (no side effects on state).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def field_fill_rate(self, record: dict, fields: list[str]) -> float:
        """Return the fraction of *fields* whose values are non-empty.

        "Empty" means: None, "" (or whitespace-only), "无", "未提及",
        "N/A", "-", "--".

        Parameters
        ----------
        record : dict
            A single extraction record.
        fields : list[str]
            The field names to check.

        Returns
        -------
        float
            non_empty_count / len(fields).  Returns 0.0 when *fields* is
            empty to avoid ZeroDivisionError.
        """
        if not fields:
            return 0.0

        non_empty = sum(
            1 for f in fields if not _is_empty(record.get(f))
        )
        return non_empty / len(fields)

    def check_sparsity(
        self,
        records: list[dict],
        fields: list[str],
        threshold: float = 0.3,
    ) -> list[int]:
        """Find records whose fill rate falls below *threshold*.

        Parameters
        ----------
        records : list[dict]
            Extraction records to inspect.
        fields : list[str]
            Field names used for the fill-rate calculation.
        threshold : float
            Records with ``field_fill_rate < threshold`` are marked for
            deletion (default 0.3).

        Returns
        -------
        list[int]
            Indices (in the original *records* order) of records that
            should be removed.
        """
        result: list[int] = []
        for idx, record in enumerate(records):
            rate = self.field_fill_rate(record, fields)
            if rate < threshold:
                result.append(idx)
        return result

    @staticmethod
    def records_equal(
        record_a: dict, record_b: dict, fields: list[str]
    ) -> bool:
        """Return True if two records are considered equal.

        Two records are "equal" when, for every field in *fields* where
        **both** have a non-empty value, those values match.  Fields
        where one or both values are empty do not count as a conflict:

        * A="PEAI", B="PEAI"   → agree
        * A="", B="PEAI"       → no conflict (A is missing that value)
        * A="PEAI", B=""       → no conflict (B is missing that value)
        * A="PEAI", B="FAI"    → conflict → return False
        """
        for f in fields:
            va = record_a.get(f)
            vb = record_b.get(f)
            if _is_empty(va) or _is_empty(vb):
                continue  # at least one side is empty → no conflict
            if va != vb:
                return False
        return True

    @staticmethod
    def record_contains(
        record_a: dict, record_b: dict, fields: list[str]
    ) -> bool:
        """Return True if *record_a* is a strict superset of *record_b*.

        *record_a* contains *record_b* when:

        1. For every field where B has a non-empty value, A has the
           **same** value.
        2. A has **at least one** field where A has a value and B does
           not (so A is strictly richer).
        3. If B has a value that A does not → A does NOT contain B.
        4. If A and B have different values on any field → A does NOT
           contain B.
        """
        # Condition 1, 3, 4: for every field where B has a value,
        # A must have the same value.
        for f in fields:
            vb = record_b.get(f)
            if _is_empty(vb):
                continue  # B doesn't claim anything on this field
            va = record_a.get(f)
            if _is_empty(va):
                return False  # B has a value, A doesn't → A does NOT contain B
            if va != vb:
                return False  # different values → A does NOT contain B

        # Condition 2: A must have at least one field where A has a value
        # and B does not.
        for f in fields:
            va = record_a.get(f)
            vb = record_b.get(f)
            if not _is_empty(va) and _is_empty(vb):
                return True  # found a field where A has value but B doesn't

        return False  # identical records → not a strict superset

    def check_duplicates(
        self, records: list[dict], fields: list[str]
    ) -> list[int]:
        """Find duplicate records using equality and containment checks.

        Compares every pair (i, j with i < j).  Priority rules:

        * ``records_equal(i, j)`` → delete **j** (the later one).
        * ``record_contains(i, j)`` → delete **j** (j is a subset of i).
        * ``record_contains(j, i)`` → delete **i** (i is a subset of j).

        Once an index is marked for deletion it is skipped in subsequent
        comparisons.

        Parameters
        ----------
        records : list[dict]
            Records to deduplicate.
        fields : list[str]
            Field names used for comparison.

        Returns
        -------
        list[int]
            Sorted list of indices to delete.
        """
        n = len(records)
        deleted: set[int] = set()

        for i in range(n):
            if i in deleted:
                continue
            for j in range(i + 1, n):
                if j in deleted:
                    continue
                if self.records_equal(records[i], records[j], fields):
                    deleted.add(j)
                elif self.record_contains(records[i], records[j], fields):
                    deleted.add(j)  # j is a subset of i
                elif self.record_contains(records[j], records[i], fields):
                    deleted.add(i)  # i is a subset of j

        return sorted(deleted)

    def run_all_checks(
        self,
        records: list[dict],
        fields: list[str],
        sparse_threshold: float = 0.3,
    ) -> dict:
        """Run sparsity check then duplicate check, returning a report.

        1. Run :meth:`check_sparsity` on the original list and remove
           those records.
        2. Run :meth:`check_duplicates` on the remaining records.
        3. Map duplicate indices back to their original positions.

        Parameters
        ----------
        records : list[dict]
            Original extraction records.
        fields : list[str]
            Field names to use for both checks.
        sparse_threshold : float
            Fill-rate threshold for sparsity deletion (default 0.3).

        Returns
        -------
        dict
            .. code-block:: python

                {
                    "sparse_deleted": [int, ...],      # original indices
                    "duplicate_deleted": [int, ...],   # original indices
                    "total_deleted": int,
                    "sparse_rate": {int: float, ...},  # index → fill_rate
                }
        """
        # ---- step 1: sparsity -------------------------------------------
        sparse_deleted = self.check_sparsity(records, fields, sparse_threshold)
        sparse_set = set(sparse_deleted)

        # Compute fill rates for the report.
        sparse_rate: dict[int, float] = {}
        for idx in sparse_deleted:
            sparse_rate[idx] = self.field_fill_rate(records[idx], fields)

        # ---- step 2: build remaining list --------------------------------
        remaining: list[dict] = []
        new_to_orig: dict[int, int] = {}  # position in *remaining* → original index
        for orig_idx, record in enumerate(records):
            if orig_idx not in sparse_set:
                new_to_orig[len(remaining)] = orig_idx
                remaining.append(record)

        # ---- step 3: duplicate check on remaining ------------------------
        dup_in_remaining = self.check_duplicates(remaining, fields)
        duplicate_deleted = [new_to_orig[nr] for nr in dup_in_remaining]

        return {
            "sparse_deleted": sparse_deleted,
            "duplicate_deleted": duplicate_deleted,
            "total_deleted": len(sparse_deleted) + len(duplicate_deleted),
            "sparse_rate": sparse_rate,
        }
