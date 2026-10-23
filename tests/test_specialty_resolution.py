"""Regression tests for scope resolution, provenance and evaluation support."""

import unittest

import pandas as pd

from src.config import PipelineConfig
from src.data_preprocessing import prepare_data, split_data


def frame(rows):
    result = pd.DataFrame(rows, columns=["medical_specialty", "transcription"])
    result["source_row"] = range(len(result))
    return result


class SpecialtyTests(unittest.TestCase):
    def fixture(self):
        return frame([
            ("Surgery", "bladder surgery"), ("Urology", "bladder surgery"),
            ("Urology", "BLADDER  surgery"),
            ("Urology", "renal ambiguity"), ("Nephrology", "renal ambiguity"),
            ("Surgery", "generic document"), ("Office Notes", "generic document"),
            ("Neurology", "neural document one"), ("Neurology", "neural document two"),
            ("Urology", "urinary document two"), ("Nephrology", "rare renal document"),
            ("Urology", ""),
        ])

    def test_scope_resolution_and_lineage(self):
        raw = self.fixture()
        kept, _, _, removed, _, audit, catalog = prepare_data(raw, PipelineConfig(min_class_groups=1))
        self.assertEqual(set(kept.source_row), {1, 7, 8, 9, 10})
        self.assertFalse(kept.raw_group.duplicated().any())
        self.assertFalse(kept.label_ambiguous.any())
        self.assertEqual(set(kept.source_row) | set(removed.source_row), set(raw.source_row))
        self.assertFalse(set(kept.source_row) & set(removed.source_row))
        self.assertFalse(removed.source_row.duplicated().any())
        self.assertEqual(len(catalog), len(raw) - 1)
        decisions = audit["_specialty_exports"]["decisions"]
        surgery = decisions.loc[decisions.resolved_label.eq("Urology") & decisions.source_record_count.eq(3)].iloc[0]
        self.assertEqual(surgery.representative_source_row, 1)
        self.assertEqual(audit["specialty_resolution"]["ambiguous_target_labels"], 1)

    def test_rare_label_filter_cannot_resolve_conflict(self):
        kept, _, _, removed, _, audit, _ = prepare_data(self.fixture(), PipelineConfig(min_class_groups=2))
        self.assertEqual(set(kept.medical_specialty), {"Urology", "Neurology"})
        self.assertNotIn(3, set(kept.source_row))
        self.assertEqual(set(removed.loc[removed.reason.eq("ambiguous_target_labels"), "source_row"]), {3, 4})
        self.assertIn(10, set(audit["_specialty_exports"]["resolved"].source_row))

    def test_top_k_cannot_resolve_conflict(self):
        kept, _, _, removed, _, _, _ = prepare_data(self.fixture(), PipelineConfig(min_class_groups=1, top_k=2))
        self.assertFalse({3, 4} & set(kept.source_row))
        self.assertEqual(set(removed.loc[removed.reason.eq("ambiguous_target_labels"), "source_row"]), {3, 4})

    def test_order_does_not_change_representative(self):
        raw = self.fixture()
        config = PipelineConfig(min_class_groups=1)
        first = prepare_data(raw, config)[0].sort_values("source_row").reset_index(drop=True)
        second = prepare_data(raw.sample(frac=1, random_state=7), config)[0].sort_values("source_row").reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)

    def test_cleaner_collision_is_separate_from_raw_ambiguity(self):
        raw = frame([("Urology", "case: stone"), ("Neurology", "case stone"),
                     ("Urology", "urinary distinct"), ("Neurology", "neural distinct")])
        kept, _, _, removed, _, audit, _ = prepare_data(raw, PipelineConfig(min_class_groups=1))
        self.assertEqual(set(kept.source_row), {2, 3})
        self.assertEqual(set(removed.reason), {"cleaning_label_collision"})
        self.assertEqual(audit["specialty_resolution"].get("ambiguous_target_labels", 0), 0)
        self.assertEqual(audit["specialty_resolution"]["cleaning_collision_rows"], 2)

    def test_unknown_labels_require_scope_review(self):
        with self.assertRaisesRegex(ValueError, "absent from specialty scope"):
            prepare_data(frame([("Unknown specialty", "some text")]), PipelineConfig(min_class_groups=1))

    def test_actual_holdout_support_is_checked(self):
        raw = frame([(label, f"case {label} {i}") for label in ("Urology", "Neurology") for i in range(30)])
        data = prepare_data(raw, PipelineConfig(min_class_groups=1))[0]
        with self.assertRaisesRegex(ValueError, "fewer than 20"):
            split_data(data, min_holdout_groups=20)
        result = split_data(data, min_holdout_groups=1)
        self.assertEqual(result.groupby("raw_group").split.nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
