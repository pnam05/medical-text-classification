"""End-to-end preprocessing deliverables: python -m scripts.run_preprocessing --help."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.preprocessing import LabelEncoder

from src.config import PipelineConfig
from src.data_preprocessing import (
    STOP_WORDS, NEGATIONS, build_vectorizer, ensure_wordnet, load_data, prepare_data, split_data,
    clean_text_accuracy_ceiling,
)
from src.eda import generate_eda
from src.reporting import write_report


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict:
    start = time.perf_counter()
    if config.min_df < 1 or config.max_features < 2:
        raise ValueError("min_df must be >= 1 and max_features must be >= 2.")
    print("[1/6] Loading CSV and cleaning resources...", flush=True)
    raw = load_data(config.input_path)
    if config.cleaning_profile == "legacy":
        ensure_wordnet(config.nltk_data_dir, config.download_nltk)
    print("[2/6] Cleaning, auditing duplicates and selecting labels...", flush=True)
    processed, selected_before, stages, removed, conflicts, audit, catalog = prepare_data(raw, config)
    processed = split_data(processed, config.random_state, config.min_holdout_groups)
    encoder = LabelEncoder().fit(processed["medical_specialty"])
    processed["label_id"] = encoder.transform(processed["medical_specialty"])
    output = config.output_dir
    for directory in ("data", "features", "artifacts", "tables", "figures"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    processed.to_csv(output / "data" / "processed.csv", index=False)
    specialty_exports = audit.pop("_specialty_exports", None)
    if specialty_exports is not None:
        decisions = specialty_exports["decisions"]
        decisions["representative_source_row"] = decisions.representative_source_row.astype("Int64")
        decisions["in_benchmark"] = decisions.representative_source_row.isin(processed.source_row)
        decisions["benchmark_split"] = decisions.representative_source_row.map(
            processed.set_index("source_row")["split"]).fillna("not_in_benchmark")
        decisions.to_csv(output / "tables" / "specialty_group_decisions.csv", index=False)
        resolved = specialty_exports["resolved"].copy()
        resolved["benchmark_split"] = resolved.text_group.map(
            processed.drop_duplicates("text_group").set_index("text_group")["split"]).fillna("not_in_benchmark")
        resolved["in_benchmark"] = resolved.source_row.isin(processed.source_row)
        resolved.to_csv(output / "data" / "resolved_specialties.csv", index=False)
        resolved.loc[~resolved.in_benchmark].to_csv(output / "data" / "resolved_low_support.csv", index=False)
        for decision, name in [("ambiguous_target_labels", "ambiguous_specialties"),
                               ("no_target_label", "no_target_specialty")]:
            catalog.loc[catalog.raw_group.isin(decisions.loc[decisions.decision.eq(decision), "raw_group"])].to_csv(
                output / "data" / f"{name}.csv", index=False)
        specialty_exports["cleaning_collisions"].to_csv(output / "tables" / "cleaning_label_collisions.csv", index=False)
        (output / "artifacts" / "specialty_scope.json").write_text(
            json.dumps(audit["specialty_scope"], ensure_ascii=False, indent=2), encoding="utf-8")
    # Assign the same partition to catalog copies sharing a benchmark group.
    # Never use the full catalog for training: it contains validation/test texts.
    assignments = processed.drop_duplicates("text_group").set_index("text_group")["split"]
    catalog["benchmark_split"] = catalog.text_group.map(assignments).fillna("not_in_benchmark")
    catalog["in_benchmark"] = catalog.source_row.isin(processed.source_row)
    catalog.to_csv(output / "data" / "cleaned_all.csv", index=False)
    catalog.loc[~catalog.medical_specialty.isin(audit["selected_specialties"])].to_csv(
        output / "data" / "rare_or_excluded_classes.csv", index=False)
    raw.merge(removed, on=["source_row", "medical_specialty"], how="inner").to_csv(
        output / "data" / "excluded_from_benchmark.csv", index=False)
    pd.DataFrame(audit["class_support"]).to_csv(output / "tables" / "class_eligibility.csv", index=False)
    support = pd.DataFrame(audit["class_support"])
    pd.DataFrame([{"min_class_groups": n, "classes": int(support.independent_groups.ge(n).sum()),
                   "rows": int(support.loc[support.independent_groups.ge(n), "rows"].sum())}
                  for n in sorted({5, 10, 20, 30, 50, 100, 150, 200, config.min_class_groups})]).to_csv(output / "tables" / "threshold_coverage.csv", index=False)
    removed.to_csv(output / "tables" / "removed_rows.csv", index=False)
    conflicts.to_csv(output / "tables" / "conflicting_labels.csv", index=False)
    review = catalog.loc[catalog.label_ambiguous_all_labels].groupby("text_group").agg(
        labels=("medical_specialty", lambda s: json.dumps(sorted(set(s)), ensure_ascii=False)),
        source_rows=("source_row", lambda s: json.dumps(s.tolist())),
        record_count=("source_row", "size"),
        text_excerpt=("transcription", lambda s: str(s.iloc[0])[:500]),
    ).reset_index()
    review.to_csv(output / "tables" / "source_label_conflicts.csv", index=False)
    if specialty_exports is not None:
        review = decisions.loc[decisions.decision.eq("ambiguous_target_labels")].copy()
    review.to_csv(output / "tables" / "label_review_queue.csv", index=False)
    examples = processed.groupby("medical_specialty", sort=True).head(2)
    examples[["source_row", "medical_specialty", "transcription", "clean_text"]].to_csv(
        output / "tables" / "cleaning_examples.csv", index=False)
    joblib.dump(encoder, output / "artifacts" / "label_encoder.joblib")
    label_mapping = {str(label): int(index) for index, label in enumerate(encoder.classes_)}
    (output / "artifacts" / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "artifacts" / "stopwords.json").write_text(json.dumps({
        "removed_words": sorted(STOP_WORDS) if config.cleaning_profile == "legacy" else [],
        "protected_negations": sorted(NEGATIONS), "profile": config.cleaning_profile,
    }, indent=2), encoding="utf-8")
    splits = {}
    print("[3/6] Exporting disjoint train / validation / test...", flush=True)
    for name in ("train", "validation", "test"):
        frame = processed.loc[processed["split"].eq(name)].copy()
        splits[name] = frame
        frame.to_csv(output / "data" / f"{name}.csv", index=False)
        np.save(output / "features" / f"y_{name}.npy", frame["label_id"].to_numpy(dtype=np.int64))
        np.save(output / "features" / f"source_rows_{name}.npy", frame["source_row"].to_numpy(dtype=np.int64))
    pd.DataFrame([{"split": key, "rows": len(frame), "unique_groups": frame["text_group"].nunique(),
                   "percent": 100 * len(frame) / len(processed)} for key, frame in splits.items()]).to_csv(
        output / "tables" / "split_summary.csv", index=False)
    from sklearn.model_selection import StratifiedGroupKFold
    cv_train = splits["train"]
    cv_assignments = cv_train[["source_row", "text_group", "medical_specialty"]].copy()
    cv_assignments["cv_fold"] = -1
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.random_state)
    for fold, (_, val_indices) in enumerate(cv.split(cv_train, cv_train.medical_specialty, cv_train.text_group)):
        cv_assignments.iloc[val_indices, cv_assignments.columns.get_loc("cv_fold")] = fold
    cv_assignments.to_csv(output / "data" / "train_cv_folds.csv", index=False)
    pd.crosstab(cv_assignments.medical_specialty, cv_assignments.cv_fold).to_csv(
        output / "tables" / "train_cv_class_counts.csv")
    # Optional weights, learned only from training label frequencies. Do not
    # implicitly apply them or oversample the validation/test partitions.
    train_counts = cv_train.label_id.value_counts()
    weights = {str(int(label)): float(len(cv_train) / (len(train_counts) * count))
               for label, count in train_counts.items()}
    (output / "artifacts" / "train_class_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    ambiguity = processed.groupby("split").agg(rows=("source_row", "size"),
                                               ambiguous_rows=("label_ambiguous", "sum"))
    ambiguity["deterministic_clean_text_accuracy_ceiling"] = pd.Series({
        name: clean_text_accuracy_ceiling(frame) for name, frame in splits.items()})
    ambiguity.to_csv(output / "tables" / "label_ambiguity_by_split.csv")
    print("[4/6] Fitting unigram and unigram+bigram TF-IDF on TRAIN only...", flush=True)
    feature_summary = {}
    for name, ngrams in [("unigram", (1, 1)), ("unigram_bigram", (1, 2))]:
        vectorizer = build_vectorizer(config, ngrams)
        try:
            x_train = vectorizer.fit_transform(splits["train"]["clean_text"])
        except ValueError as error:
            raise ValueError("TF-IDF vocabulary is empty or too small; check cleaning and --min-df.") from error
        joblib.dump(vectorizer, output / "artifacts" / f"tfidf_{name}.joblib")
        pd.DataFrame({"feature_index": range(len(vectorizer.get_feature_names_out())),
                      "term": vectorizer.get_feature_names_out(), "idf": vectorizer.idf_}).to_csv(
            output / "features" / f"vocabulary_{name}.csv", index=False)
        feature_summary[name] = {}
        for split, frame in splits.items():
            matrix = x_train if split == "train" else vectorizer.transform(frame["clean_text"])
            save_npz(output / "features" / f"X_{split}_{name}.npz", matrix)
            feature_summary[name][split] = {
                "shape": list(matrix.shape), "nonzero": int(matrix.nnz),
                "sparsity": float(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
                "zero_vector_rows": int(np.count_nonzero(matrix.getnnz(axis=1) == 0)),
            }
    print("[5/6] Rendering before / after EDA figures and tables...", flush=True)
    eda_summary = generate_eda(raw, selected_before, processed, stages, conflicts,
                               vectorizer, x_train, output, config.random_state)
    if specialty_exports is not None:
        from src.eda import generate_specialty_eda
        generate_specialty_eda(decisions, support, processed, output)
    class_counts = processed["medical_specialty"].value_counts()
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "input_sha256": hashlib.sha256(config.input_path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "raw_classes": int(raw["medical_specialty"].nunique()),
        "processed_rows": len(processed), "removed_rows": len(removed),
        "retained_fraction_of_valid": len(processed) / len(catalog),
        "ambiguous_benchmark_rows": int(processed.label_ambiguous.sum()),
        "deterministic_clean_text_accuracy_ceiling": clean_text_accuracy_ceiling(processed),
        "class_imbalance_ratio": float(class_counts.max() / class_counts.min()),
        **audit, "split_rows": {name: len(frame) for name, frame in splits.items()},
        "leakage_checks": {"raw_groups_disjoint": True, "clean_groups_disjoint": True,
                           "tfidf_fit_partition": "train"},
        "features": feature_summary, "eda": eda_summary,
        "versions": {name: importlib.metadata.version(name) for name in [
            "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "nltk", "scipy", "joblib"]},
        "python": platform.python_version(),
    }
    if len(processed) + len(removed) != len(raw):
        raise AssertionError("Removal audit does not reconcile with input row count.")
    if removed.source_row.duplicated().any() or set(processed.source_row) & set(removed.source_row):
        raise AssertionError("Source rows must have exactly one final disposition.")
    if specialty_exports is not None:
        assert not processed.raw_group.duplicated().any()
        assert not processed.label_ambiguous.any()
    print("[6/6] Writing Vietnamese Markdown / HTML reports...", flush=True)
    write_report(output, summary, processed, stages, removed)
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {len(raw)} -> {len(processed)} records; splits={summary['split_rows']}", flush=True)
    print(f"Report: {(output / 'report.html').resolve()}", flush=True)
    return summary


def parse_args() -> PipelineConfig:
    default = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default.input_path, dest="input_path")
    parser.add_argument("--output-dir", type=Path, default=default.output_dir)
    parser.add_argument("--top-k", type=int, default=default.top_k, help="Optional ablation; default keeps all eligible classes.")
    parser.add_argument("--min-class-groups", type=int, default=default.min_class_groups)
    parser.add_argument("--min-holdout-groups", type=int, default=default.min_holdout_groups)
    parser.add_argument("--benchmark-policy", choices=["specialty_single_label", "source_labels"], default=default.benchmark_policy)
    parser.add_argument("--specialty-scope-path", type=Path, default=default.specialty_scope_path)
    parser.add_argument("--cleaning-profile", choices=["conservative", "legacy"], default=default.cleaning_profile)
    parser.add_argument("--seed", type=int, default=default.random_state, dest="random_state")
    parser.add_argument("--min-df", type=int, default=default.min_df)
    parser.add_argument("--max-features", type=int, default=default.max_features)
    parser.add_argument("--keep-numbers", action=argparse.BooleanOptionalAction, default=default.keep_numbers)
    parser.add_argument("--conflict-policy", choices=["group", "drop"], default=default.conflict_policy)
    parser.add_argument("--download-nltk", action="store_true", help="Download WordNet if missing.")
    parser.add_argument("--nltk-data-dir", type=Path, default=default.nltk_data_dir)
    return PipelineConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_pipeline(parse_args())
