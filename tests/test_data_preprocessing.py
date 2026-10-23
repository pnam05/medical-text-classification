"""Data correctness tests, including leakage and contradictory labels."""

import unittest

import pandas as pd

from src.config import PipelineConfig
from src.data_preprocessing import (
    MedicalTextCleaner, build_vectorizer, ensure_wordnet, normalized_text,
    prepare_data, split_data, text_hash, clean_text_accuracy_ceiling,
)


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_wordnet(PipelineConfig().nltk_data_dir)
        cls.clean = MedicalTextCleaner(keep_numbers=False, profile="legacy")

    def test_negations_and_contractions(self):
        tokens = self.clean("No fever. Isn't ill; can't walk; won't eat. Without pain, negative test; denies cough.").split()
        self.assertIn("no", tokens)
        self.assertEqual(tokens.count("not"), 3)
        for token in ("without", "negative", "denies"):
            self.assertIn(token, tokens)

    def test_medical_alphanumerics_and_number_option(self):
        text = "B12 T4 SpO2 5-FU 123 98.6"
        cleaned = self.clean(text).split()
        self.assertEqual(cleaned, ["b12", "t4", "spo2", "5-fu"])
        self.assertIn("123", MedicalTextCleaner(keep_numbers=True)(text).split())

    def test_real_lemmatization_and_html(self):
        self.assertEqual(self.clean("<b>The</b> patients have fractures &amp; diagnoses."),
                         "patient fracture diagnosis")

    def test_lemmatization_must_not_invent_negation(self):
        self.assertEqual(self.clean("NOS diagnosis"), "nos diagnosis")

    def test_missing_empty_and_unicode(self):
        self.assertEqual(self.clean(pd.NA), "")
        self.assertEqual(self.clean("the and 123 !!!"), "")
        self.assertIn("not", self.clean("isn’t symptomatic").split())

    def test_audit_and_conflicting_label_policy(self):
        raw = pd.DataFrame({
            "source_row": range(8),
            "transcription": ["The chest pains", "the chest pains", "The chest pains", "Knee fractures",
                              "Kidney stones", "", "the and 123", "normal lungs"],
            "medical_specialty": ["A", "A", "B", "B", "C", "A", "A", "A"],
        })
        kept, _, _, removed, conflicts, audit, catalog = prepare_data(raw, PipelineConfig(
            benchmark_policy="source_labels", top_k=2, min_class_groups=1, cleaning_profile="legacy", keep_numbers=False))
        self.assertEqual(len(kept) + len(removed), len(raw))
        self.assertEqual(set(removed.reason), {"missing_text_or_label", "empty_after_cleaning",
                                             "outside_top_k", "duplicate_text_and_label"})
        self.assertEqual(audit["final_duplicates"]["conflicting_label_groups"], 1)
        self.assertEqual(set(conflicts.medical_specialty), {"A", "B"})
        dropped, _, _, dropped_rows, _, audit, _ = prepare_data(
            raw, PipelineConfig(benchmark_policy="source_labels", top_k=2, min_class_groups=1, conflict_policy="drop",
                                cleaning_profile="legacy", keep_numbers=False))
        self.assertEqual(audit["final_duplicates"]["conflicting_label_groups"], 0)
        self.assertEqual(len(dropped) + len(dropped_rows), len(raw))
        self.assertIn("conflicting_labels", set(dropped_rows.reason))

    def test_schema_independent_normalization(self):
        self.assertEqual(normalized_text("  CHEST\n pain "), "chest pain")

    def test_conservative_preserves_clinical_context(self):
        clean = MedicalTextCleaner()
        result = clean("<b>No</b> change in US. NOS: 2.5mg, BP 120/80, SpO2 98%, glucose <5 and >10.")
        for token in ["no", "in", "us", "nos", "2.5mg", "120/80", "spo2", "98%", "<5", ">10"]:
            self.assertIn(token, result.split())
        self.assertEqual(clean("patients have diagnoses"), "patients have diagnoses")

    def test_support_uses_unique_texts_and_preserves_catalog(self):
        rows = [("A", f"alpha case{i}") for i in range(3)]
        rows += [("B", f"beta case{i}") for i in range(3)]
        rows += [("Rare", "same rare text")] * 30
        raw = pd.DataFrame(rows, columns=["medical_specialty", "transcription"])
        raw["source_row"] = range(len(raw))
        kept, _, _, removed, _, audit, catalog = prepare_data(raw, PipelineConfig(benchmark_policy="source_labels", min_class_groups=3))
        self.assertEqual(set(kept.medical_specialty), {"A", "B"})
        self.assertEqual(len(catalog), len(raw))
        self.assertEqual(len(kept) + len(removed), len(raw))
        rare = next(r for r in audit["class_support"] if r["medical_specialty"] == "Rare")
        self.assertEqual(rare["independent_groups"], 1)
        self.assertFalse(rare["selected"])

    def test_lossy_cleaning_collision_is_not_source_duplicate(self):
        raw = pd.DataFrame({"source_row": range(3), "medical_specialty": ["A", "A", "B"],
                            "transcription": ["dose 5", "dose 10", "other text"]})
        kept, _, _, removed, _, _, _ = prepare_data(raw, PipelineConfig(
            benchmark_policy="source_labels", min_class_groups=1, cleaning_profile="legacy", keep_numbers=False))
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept.loc[kept.medical_specialty.eq("A"), "text_group"].nunique(), 1)
        self.assertEqual(len(removed), 0)


class SplitAndFeatureTests(unittest.TestCase):
    def test_identical_input_accuracy_ceiling(self):
        data = pd.DataFrame({"text_group": ["x", "x", "y"], "medical_specialty": ["A", "B", "A"]})
        self.assertAlmostEqual(clean_text_accuracy_ceiling(data), 2 / 3)

    def fixture(self):
        rows = []
        for index in range(50):
            for label in ("A", "B", "C"):
                # Several exact texts with different labels must stay together.
                text = f"shared {index}" if index % 4 == 0 else f"specialty {label} {index}"
                rows.append({"medical_specialty": label, "clean_text": text,
                             "text_group": text_hash(text), "raw_group": text_hash(text)})
        return pd.DataFrame(rows)

    def test_splits_reproducible_and_disjoint(self):
        data = self.fixture()
        result = split_data(data)
        pd.testing.assert_frame_equal(result, split_data(data))
        self.assertEqual(result.groupby("text_group")["split"].nunique().max(), 1)
        self.assertEqual(set(result["split"]), {"train", "validation", "test"})
        for _, frame in result.groupby("split"):
            self.assertEqual(set(frame.medical_specialty), {"A", "B", "C"})

    def test_too_few_groups_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "at least 5"):
            split_data(self.fixture().head(6))

    def test_tfidf_does_not_learn_holdout_words(self):
        vectorizer = build_vectorizer(PipelineConfig(min_df=1))
        vectorizer.fit_transform(["chest pain b12", "no fever t4"])
        matrix = vectorizer.transform(["holdoutonly b12"])
        self.assertNotIn("holdoutonly", vectorizer.vocabulary_)
        self.assertIn("chest pain", vectorizer.vocabulary_)
        self.assertIn("b12", vectorizer.vocabulary_)
        self.assertEqual(matrix.nnz, 1)


if __name__ == "__main__":
    unittest.main()
