"""
Unit tests for extract/quality_checker.py.

Run:
    python platform_init/test/prompt/test_quality_checker.py
"""

import sys
import unittest

sys.path.insert(0, r"D:\PycharmProjects\SDL_agent")

from extract.quality_checker import QualityChecker, _is_empty, EMPTY_MARKERS


# ======================================================================
# _is_empty helper
# ======================================================================
class TestIsEmpty(unittest.TestCase):
    """Verify the low-level empty-value detection used by all methods."""

    def test_none(self):
        self.assertTrue(_is_empty(None))

    def test_empty_string(self):
        self.assertTrue(_is_empty(""))

    def test_whitespace_only(self):
        self.assertTrue(_is_empty("   "))
        self.assertTrue(_is_empty("\t\n "))

    def test_markers(self):
        for m in ["无", "未提及", "N/A", "-", "--"]:
            with self.subTest(marker=m):
                self.assertTrue(_is_empty(m), f"'{m}' should be empty")

    def test_legit_values(self):
        self.assertFalse(_is_empty("PEAI"))
        self.assertFalse(_is_empty("22%"))
        self.assertFalse(_is_empty("0"))  # zero is a real value
        self.assertFalse(_is_empty(" organic "))  # after strip → "organic"


# ======================================================================
# field_fill_rate
# ======================================================================
class TestFieldFillRate(unittest.TestCase):
    def setUp(self):
        self.qc = QualityChecker()

    def test_all_filled(self):
        record = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, ["name", "type", "PCE"]), 1.0
        )

    def test_half_filled(self):
        record = {"name": "PEAI", "type": "", "PCE": "22%", "notes": ""}
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, ["name", "type", "PCE", "notes"]), 0.5
        )

    def test_none_filled(self):
        record = {"name": "", "type": "无", "PCE": "未提及"}
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, ["name", "type", "PCE"]), 0.0
        )

    def test_empty_markers_treated_as_empty(self):
        """Every standard sentinel value should count as empty."""
        record = {
            "name": "PEAI",
            "type": "无",
            "PCE": "未提及",
            "stability": "N/A",
            "solvent": "-",
            "temp": "--",
        }
        # Only "name" is non-empty → 1/6
        rate = self.qc.field_fill_rate(
            record, ["name", "type", "PCE", "stability", "solvent", "temp"]
        )
        self.assertAlmostEqual(rate, 1.0 / 6.0)

    def test_missing_key_treated_as_empty(self):
        record = {"name": "PEAI"}
        rate = self.qc.field_fill_rate(record, ["name", "type", "PCE"])
        self.assertAlmostEqual(rate, 1.0 / 3.0)  # only name present

    def test_none_value(self):
        record = {"name": "PEAI", "type": None}
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, ["name", "type"]), 0.5
        )

    def test_whitespace_only_values(self):
        record = {"name": "PEAI", "type": "  ", "PCE": "\t"}
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, ["name", "type", "PCE"]),
            1.0 / 3.0,
        )

    def test_zero_is_not_empty(self):
        record = {"name": "PEAI", "PCE": 0}  # 0 is a real measured value
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, ["name", "PCE"]), 1.0
        )

    def test_empty_fields_list(self):
        record = {"name": "PEAI"}
        self.assertAlmostEqual(
            self.qc.field_fill_rate(record, []), 0.0
        )


# ======================================================================
# check_sparsity
# ======================================================================
class TestCheckSparsity(unittest.TestCase):
    def setUp(self):
        self.qc = QualityChecker()

    def test_no_sparse_records(self):
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "FAI", "type": "organic", "PCE": "20%"},
        ]
        fields = ["name", "type", "PCE"]
        self.assertEqual(self.qc.check_sparsity(records, fields, threshold=0.3), [])

    def test_one_sparse_record(self):
        """Record 1 has only 1/4 fields → 0.25 < 0.3 → sparse."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "good"},
            {"name": "FAI", "type": "", "PCE": "", "notes": ""},
        ]
        fields = ["name", "type", "PCE", "notes"]
        self.assertEqual(
            self.qc.check_sparsity(records, fields, threshold=0.3), [1]
        )

    def test_fill_rate_at_threshold_is_not_sparse(self):
        """1/3 = 0.333... which is >= 0.3 → NOT sparse."""
        records = [
            {"name": "PEAI", "type": "", "PCE": ""},
        ]
        fields = ["name", "type", "PCE"]
        # fill_rate = 1/3 ≈ 0.333 >= 0.3 → not deleted
        self.assertEqual(
            self.qc.check_sparsity(records, fields, threshold=0.3), []
        )

    def test_custom_threshold(self):
        """With threshold=0.5, 1/3 = 0.33 < 0.5 → sparse."""
        records = [
            {"name": "PEAI", "type": "", "PCE": ""},
        ]
        fields = ["name", "type", "PCE"]
        self.assertEqual(
            self.qc.check_sparsity(records, fields, threshold=0.5), [0]
        )

    def test_threshold_zero(self):
        """threshold=0.0 deletes records that are completely empty (0/3 < 0)."""
        records = [
            {"name": "PEAI", "type": "", "PCE": ""},     # 1/3 = 0.33 → keep
            {"name": "", "type": "", "PCE": ""},          # 0/3 = 0.0 → NOT < 0.0 → keep!
        ]
        fields = ["name", "type", "PCE"]
        self.assertEqual(
            self.qc.check_sparsity(records, fields, threshold=0.0), []
        )

    def test_threshold_one(self):
        """threshold=1.0 deletes all non-perfect records."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},  # 3/3 → keep
            {"name": "FAI", "type": "", "PCE": "20%"},           # 2/3 → delete
        ]
        fields = ["name", "type", "PCE"]
        self.assertEqual(
            self.qc.check_sparsity(records, fields, threshold=1.0), [1]
        )

    def test_multiple_sparse(self):
        records = [
            {"name": "", "type": "", "PCE": ""},           # 0/3
            {"name": "FAI", "type": "organic", "PCE": "20%"},  # 3/3
            {"name": "", "type": "", "PCE": ""},           # 0/3
        ]
        fields = ["name", "type", "PCE"]
        self.assertEqual(
            self.qc.check_sparsity(records, fields, threshold=0.3), [0, 2]
        )


# ======================================================================
# records_equal
# ======================================================================
class TestRecordsEqual(unittest.TestCase):
    def test_identical_records(self):
        a = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        b = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_different_values_on_one_field(self):
        a = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        b = {"name": "FAI", "type": "organic", "PCE": "22%"}
        self.assertFalse(
            QualityChecker.records_equal(a, b, ["name", "type", "PCE"])
        )

    def test_one_side_empty_no_conflict(self):
        """A has empty type, B has 'organic' → no conflict on type."""
        a = {"name": "PEAI", "type": "", "PCE": "22%"}
        b = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_second_side_empty_no_conflict(self):
        a = {"name": "PEAI", "type": "organic", "PCE": ""}
        b = {"name": "PEAI", "type": "", "PCE": "22%"}
        # type: B empty → skip. PCE: A empty → skip. Only name compared.
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_both_empty_on_field(self):
        a = {"name": "PEAI", "type": "", "PCE": ""}
        b = {"name": "PEAI", "type": "无", "PCE": "N/A"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_empty_marker_on_both_sides(self):
        """Both have empty markers (possibly different ones)."""
        a = {"name": "PEAI", "type": "无", "PCE": "N/A"}
        b = {"name": "PEAI", "type": "N/A", "PCE": "无"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_cross_missing_info(self):
        """A has name+type, B has name+PCE. No conflicting fields."""
        a = {"name": "PEAI", "type": "organic", "PCE": ""}
        b = {"name": "PEAI", "type": "", "PCE": "22%"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_conflict_on_different_field_across_records(self):
        """Even though they share one field in common, the conflict matters."""
        a = {"name": "REAGENT-A", "type": "organic", "PCE": ""}
        b = {"name": "REAGENT-A", "type": "inorganic", "PCE": "22%"}
        # type differs and both non-empty → conflict
        self.assertFalse(
            QualityChecker.records_equal(a, b, ["name", "type", "PCE"])
        )

    def test_subset_fields_only(self):
        """Only specified fields are compared; extra keys are ignored."""
        a = {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "xyz"}
        b = {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "abc"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type", "PCE"]))

    def test_both_completely_empty(self):
        a = {"name": "", "type": ""}
        b = {"name": "无", "type": "N/A"}
        self.assertTrue(QualityChecker.records_equal(a, b, ["name", "type"]))


# ======================================================================
# record_contains
# ======================================================================
class TestRecordContains(unittest.TestCase):
    def test_a_has_extra_field(self):
        """A has all of B's values plus extra → A contains B."""
        a = {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "stable"}
        b = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        self.assertTrue(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE", "notes"])
        )

    def test_identical_not_contains(self):
        """Identical records → NOT a strict superset."""
        a = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        b = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )

    def test_b_has_value_a_does_not(self):
        """B has a field value that A is missing → A does NOT contain B."""
        a = {"name": "PEAI", "type": "", "PCE": "22%"}
        b = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )

    def test_different_values(self):
        a = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        b = {"name": "PEAI", "type": "inorganic", "PCE": "22%"}
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )

    def test_a_completely_empty_cannot_contain_anything(self):
        a = {"name": "", "type": ""}
        b = {"name": "PEAI", "type": "organic"}
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type"])
        )

    def test_b_completely_empty(self):
        """B is empty → A has at least one value → A contains B."""
        a = {"name": "PEAI", "type": "organic"}
        b = {"name": "", "type": ""}
        self.assertTrue(
            QualityChecker.record_contains(a, b, ["name", "type"])
        )

    def test_missing_key_in_b_treated_as_empty(self):
        """B missing a key entirely → treated as empty → A contains B if richer."""
        a = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        b = {"name": "PEAI"}
        # "type" and "PCE" keys not in B → get() returns None → empty
        self.assertTrue(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )

    def test_no_extra_field_so_not_contains(self):
        """A has same info as B, no extra → not a strict superset."""
        a = {"name": "PEAI", "type": "organic", "PCE": ""}
        b = {"name": "PEAI", "type": "organic", "PCE": ""}
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )

    def test_strictness_requires_extra_field(self):
        """A has all B's values but no extra fields → not strict superset."""
        a = {"name": "PEAI", "type": "organic"}
        b = {"name": "PEAI", "type": "organic"}
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type"])
        )

    def test_a_has_all_b_values_and_more(self):
        """Classic superset: A has everything B has plus more."""
        a = {"name": "PEAI", "type": "organic", "PCE": "22%"}
        b = {"name": "PEAI", "PCE": "22%"}
        # B has name and PCE → A matches both. A also has type → strict superset.
        self.assertTrue(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )

    def test_not_contains_when_b_has_different_value_on_non_overlapping(self):
        """A and B share name, differ on disjoint field sets."""
        a = {"name": "PEAI", "type": "organic", "PCE": ""}
        b = {"name": "PEAI", "type": "", "PCE": "22%"}
        # B has PCE → A empty → A does NOT contain B
        self.assertFalse(
            QualityChecker.record_contains(a, b, ["name", "type", "PCE"])
        )


# ======================================================================
# check_duplicates
# ======================================================================
class TestCheckDuplicates(unittest.TestCase):
    def setUp(self):
        self.qc = QualityChecker()

    def test_no_duplicates(self):
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "FAI", "type": "organic", "PCE": "20%"},
            {"name": "MAI", "type": "inorganic", "PCE": "18%"},
        ]
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type", "PCE"]), []
        )

    def test_equal_records_delete_later(self):
        """Two equal records → delete the later one (index 1)."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
        ]
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type", "PCE"]), [1]
        )

    def test_three_records_first_and_third_equal(self):
        """Records[0] == records[2] → delete index 2."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "FAI", "type": "organic", "PCE": "20%"},
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
        ]
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type", "PCE"]), [2]
        )

    def test_a_contains_b_delete_b(self):
        """A is richer → B is a subset → delete B (index 1)."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "stable"},
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
        ]
        self.assertEqual(
            self.qc.check_duplicates(
                records, ["name", "type", "PCE", "notes"]
            ),
            [1],
        )

    def test_b_richer_than_a(self):
        """B has extra data that A lacks. records_equal fires first
        (both agree on shared fields, A's missing fields are empty →
        no conflict), so the later record (index 1) is deleted by the
        equality rule. This means containment is never reached when
        the subset/superset pair has no conflicting shared values —
        which is correct per the spec's priority ordering."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "stable"},
        ]
        self.assertEqual(
            self.qc.check_duplicates(
                records, ["name", "type", "PCE", "notes"]
            ),
            [1],
        )

    def test_already_deleted_record_skipped(self):
        """Record 1 deleted by equality with 0 → skip 1-vs-2 comparison."""
        records = [
            {"name": "X", "type": "A"},
            {"name": "X", "type": "A"},   # dup of 0, will be deleted
            {"name": "X", "type": "A"},   # also dup of 0 (and 1), but 1 already deleted
        ]
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type"]), [1, 2]
        )

    def test_chain_contains(self):
        """A contains B, B contains C → delete B and C."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "good", "ref": "A"},
            {"name": "PEAI", "type": "organic", "PCE": "22%", "notes": "good"},
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
        ]
        fields = ["name", "type", "PCE", "notes", "ref"]
        self.assertEqual(
            self.qc.check_duplicates(records, fields), [1, 2]
        )

    def test_equal_priority_over_contains(self):
        """records_equal wins over record_contains. Two records that look
        equal by equality rule (ignoring empty values) → delete later one,
        even if the later one has extra data."""
        records = [
            {"name": "PEAI", "type": "", "PCE": "22%"},       # earlier
            {"name": "PEAI", "type": "organic", "PCE": "22%"},  # later, richer
        ]
        # records_equal: type field: A empty → skip → True → delete index 1
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type", "PCE"]), [1]
        )


# ======================================================================
# run_all_checks
# ======================================================================
class TestRunAllChecks(unittest.TestCase):
    def setUp(self):
        self.qc = QualityChecker()

    def test_all_clean(self):
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "FAI", "type": "organic", "PCE": "20%"},
        ]
        result = self.qc.run_all_checks(records, ["name", "type", "PCE"])
        self.assertEqual(result["sparse_deleted"], [])
        self.assertEqual(result["duplicate_deleted"], [])
        self.assertEqual(result["total_deleted"], 0)
        self.assertEqual(result["sparse_rate"], {})

    def test_sparse_only(self):
        records = [
            {"name": "", "type": "", "PCE": ""},           # 0/3 → sparse
            {"name": "FAI", "type": "organic", "PCE": "20%"},  # 3/3 → keep
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE"], sparse_threshold=0.3
        )
        self.assertEqual(result["sparse_deleted"], [0])
        self.assertEqual(result["duplicate_deleted"], [])
        self.assertEqual(result["total_deleted"], 1)
        self.assertAlmostEqual(result["sparse_rate"][0], 0.0)

    def test_duplicates_only_no_sparse(self):
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            {"name": "PEAI", "type": "organic", "PCE": "22%"},  # dup of 0
            {"name": "FAI", "type": "organic", "PCE": "20%"},
        ]
        result = self.qc.run_all_checks(records, ["name", "type", "PCE"])
        self.assertEqual(result["sparse_deleted"], [])
        self.assertEqual(result["duplicate_deleted"], [1])
        self.assertEqual(result["total_deleted"], 1)

    def test_sparse_first_then_duplicates_index_mapping(self):
        """Sparse records are removed first; duplicate indices are mapped
        back to original positions."""
        records = [
            # 0: sparse → deleted in step 1
            {"name": "", "type": "", "PCE": ""},
            # 1: good
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            # 2: good, duplicate of 1
            {"name": "PEAI", "type": "organic", "PCE": "22%"},
            # 3: good
            {"name": "FAI", "type": "organic", "PCE": "20%"},
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE"], sparse_threshold=0.3
        )
        self.assertEqual(result["sparse_deleted"], [0])
        # After removing index 0, remaining = [1(orig), 2(orig), 3(orig)]
        # remaining[0]=orig[1], remaining[1]=orig[2], remaining[2]=orig[3]
        # duplicates: remaining[0] == remaining[1] → delete remaining[1] → orig[2]
        self.assertEqual(result["duplicate_deleted"], [2])
        self.assertEqual(result["total_deleted"], 2)

    def test_sparse_and_dup_no_overlap(self):
        """Sparse records are distinct from duplicate records."""
        records = [
            {"name": "", "type": "", "PCE": ""},                # 0: sparse
            {"name": "PEAI", "type": "organic", "PCE": "22%"},  # 1: keep
            {"name": "PEAI", "type": "organic", "PCE": "22%"},  # 2: dup of 1
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE"], sparse_threshold=0.3
        )
        self.assertEqual(result["sparse_deleted"], [0])
        self.assertEqual(result["duplicate_deleted"], [2])
        self.assertEqual(result["total_deleted"], 2)

    def test_custom_sparse_threshold(self):
        """With threshold=0.5, records with 1/3 filled are sparse."""
        records = [
            {"name": "PEAI", "type": "", "PCE": ""},             # 1/3 ≈ 0.33 < 0.5 → sparse
            {"name": "FAI", "type": "organic", "PCE": "20%"},   # 3/3 → keep
            {"name": "FAI", "type": "organic", "PCE": "20%"},   # dup of 1
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE"], sparse_threshold=0.5
        )
        self.assertEqual(result["sparse_deleted"], [0])
        self.assertEqual(result["duplicate_deleted"], [2])
        self.assertEqual(result["total_deleted"], 2)

    def test_sparse_rate_includes_all_sparse_indices(self):
        """Report includes fill rate for every sparse-deleted record."""
        records = [
            {"name": "", "type": "", "PCE": ""},             # 0/3 = 0.0
            {"name": "X", "type": "", "PCE": ""},            # 1/3 ≈ 0.33 (>= 0.3, keep)
            {"name": "Y", "type": "", "PCE": "", "notes": ""},  # 1/4 = 0.25
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE", "notes"], sparse_threshold=0.3
        )
        # Record 0: 0/4 = 0.0  < 0.3 → sparse
        # Record 1: 1/4 = 0.25 < 0.3 → sparse
        # Record 2: 1/4 = 0.25 < 0.3 → sparse
        self.assertEqual(result["sparse_deleted"], [0, 1, 2])
        self.assertAlmostEqual(result["sparse_rate"][0], 0.0)
        self.assertAlmostEqual(result["sparse_rate"][1], 0.25)
        self.assertAlmostEqual(result["sparse_rate"][2], 0.25)

    def test_empty_input(self):
        """Empty list of records should produce empty results."""
        result = self.qc.run_all_checks([], ["name", "type", "PCE"])
        self.assertEqual(result["sparse_deleted"], [])
        self.assertEqual(result["duplicate_deleted"], [])
        self.assertEqual(result["total_deleted"], 0)
        self.assertEqual(result["sparse_rate"], {})

    def test_all_sparse_none_left_for_duplicates(self):
        records = [
            {"name": "", "type": "", "PCE": ""},
            {"name": "无", "type": "N/A", "PCE": "未提及"},
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE"], sparse_threshold=0.3
        )
        self.assertEqual(result["sparse_deleted"], [0, 1])
        self.assertEqual(result["duplicate_deleted"], [])
        self.assertEqual(result["total_deleted"], 2)


# ======================================================================
# Integration / edge-case tests
# ======================================================================
class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.qc = QualityChecker()

    def test_single_record(self):
        """Single record is never duplicate."""
        records = [{"name": "PEAI", "type": "organic"}]
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type"]), []
        )

    def test_all_same_empty_record(self):
        """Two completely empty records are 'equal' → delete later."""
        records = [
            {"name": "", "type": ""},
            {"name": "无", "type": "N/A"},
        ]
        self.assertEqual(
            self.qc.check_duplicates(records, ["name", "type"]), [1]
        )

    def test_string_vs_int_comparison(self):
        """Values are compared as-is from the dict; 22 != "22"."""
        a = {"name": "PEAI", "PCE": 22}
        b = {"name": "PEAI", "PCE": "22"}
        self.assertFalse(
            QualityChecker.records_equal(a, b, ["name", "PCE"])
        )

    def test_record_contains_with_empty_b(self):
        """A non-empty, B empty → A contains B."""
        self.assertTrue(
            QualityChecker.record_contains(
                {"name": "PEAI"}, {"name": ""}, ["name"]
            )
        )

    def test_record_contains_both_empty(self):
        """Both empty → A does NOT contain B (no strict extra field)."""
        self.assertFalse(
            QualityChecker.record_contains(
                {"name": ""}, {"name": ""}, ["name"]
            )
        )

    def test_duplicate_same_after_sparse_filter(self):
        """A sparse record is removed first, so its identical non-sparse copy
        survives without being tagged as duplicate."""
        records = [
            {"name": "PEAI", "type": "organic", "PCE": "22%"},   # 0: keep
            {"name": "", "type": "", "PCE": ""},                  # 1: sparse
            {"name": "", "type": "", "PCE": ""},                  # 2: sparse (& dup of 1)
        ]
        result = self.qc.run_all_checks(
            records, ["name", "type", "PCE"], sparse_threshold=0.3
        )
        # Both 1 and 2 are sparse, removed in step 1.
        # Remaining = [0], no duplicates.
        self.assertEqual(result["sparse_deleted"], [1, 2])
        self.assertEqual(result["duplicate_deleted"], [])
        self.assertEqual(result["total_deleted"], 2)


# ======================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
