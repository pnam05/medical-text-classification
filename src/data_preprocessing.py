"""Reusable cleaning, duplicate auditing, grouped splits and TF-IDF.

Run the complete workflow with ``python -m scripts.run_preprocessing``.
No fitting, downloading or file writing happens during import.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.config import PipelineConfig
from src.specialty_resolution import resolve_specialty_groups

NEGATIONS = frozenset({
    "no", "not", "nor", "neither", "never", "none", "without", "negative",
    "deny", "denies", "denied", "denying", "absent", "absence", "cannot",
})
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS
# Keep digits attached to medical tokens: t4, b12, h1n1, spo2, 5-fu.
TOKEN_RE = re.compile(r"(?:[<>]=?|[≤≥])?\d+(?:[./:-]\d+)*(?:%|[a-z]+)?(?![a-z0-9-])|[a-z0-9]+(?:[-/][a-z0-9]+)*(?:%)?")


def ensure_wordnet(data_dir: Path, download: bool = False) -> None:
    """Fail explicitly if genuine lemmatization cannot be performed."""
    data_dir = Path(data_dir)
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))
    try:
        wordnet.ensure_loaded()
    except LookupError as error:
        if not download:
            raise RuntimeError(
                "Missing NLTK WordNet. Run python -m scripts.run_preprocessing "
                "--download-nltk once (requires Internet)."
            ) from error
        data_dir.mkdir(parents=True, exist_ok=True)
        if not nltk.download("wordnet", download_dir=str(data_dir), quiet=True):
            raise RuntimeError("WordNet download failed; check the network and retry.")
        wordnet.ensure_loaded()


class MedicalTextCleaner:
    """Conservative token normalization, with optional legacy linguistic cleaning.

    Default preserves numbers, stopwords and word forms. Legacy uses noun-only
    WordNet, not a POS tagger or a medical dictionary. Neither is lossless NLP.
    """

    def __init__(self, keep_numbers: bool = True, profile: str = "conservative"):
        if profile not in {"conservative", "legacy"}:
            raise ValueError("profile must be conservative or legacy")
        self.keep_numbers = keep_numbers
        self.profile = profile
        self.lemmatizer = WordNetLemmatizer()

    @lru_cache(maxsize=100_000)
    def lemma(self, token: str) -> str:
        if token in NEGATIONS or any(char.isdigit() for char in token):
            return token
        lemma = self.lemmatizer.lemmatize(token, pos="n")
        # WordNet interprets clinical NOS as plural "nos" -> "no". Do not
        # introduce a negation that was absent from the original token.
        return token if lemma in NEGATIONS else lemma

    def __call__(self, value: str) -> str:
        if pd.isna(value):
            return ""
        text = unicodedata.normalize("NFKC", html.unescape(str(value))).lower()
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        # Match actual HTML tags, not clinical comparisons such as '<5 ... >10'.
        text = re.sub(r"</?[a-z][a-z0-9]*(?:\s+[^<>]*?)?\s*/?>", " ", text)
        text = re.sub(r"\bwon't\b", "will not", text)
        text = re.sub(r"\bcan't\b", "can not", text)
        text = re.sub(r"\bcannot\b", "can not", text)
        text = re.sub(r"n['’]t\b", " not", text)
        text = re.sub(r"['’](?:s|re|ve|ll|d|m)\b", " ", text)
        tokens = []
        for token in TOKEN_RE.findall(text):
            if not self.keep_numbers and not any(c.isalpha() for c in token):
                continue
            if self.profile == "legacy" and token in STOP_WORDS:
                continue
            lemma = self.lemma(token) if self.profile == "legacy" else token
            if lemma and (self.profile != "legacy" or lemma not in STOP_WORDS):
                tokens.append(lemma)
        return " ".join(tokens)


def normalized_text(value: str) -> str:
    """Conservative key: Unicode, case and whitespace only."""
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).lower().split())


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"transcription", "medical_specialty"}
    if missing := required - set(data.columns):
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    reserved = {"source_row", "raw_group", "clean_text", "text_group", "split", "label_id"}
    if overlap := reserved & set(data.columns):
        raise ValueError(f"Input contains reserved pipeline columns: {sorted(overlap)}")
    # Zero-based data record number, NOT the physical CSV line (texts span lines).
    data.insert(0, "source_row", range(len(data)))
    data["medical_specialty"] = data["medical_specialty"].astype("string").str.strip()
    data["transcription"] = data["transcription"].astype("string")
    return data


def duplicate_summary(data: pd.DataFrame, key: str) -> dict:
    sizes = data.groupby(key).size()
    labels = data.groupby(key)["medical_specialty"].nunique()
    conflicting = labels[labels.gt(1)].index
    return {
        "unique_text_groups": int(len(sizes)),
        "duplicate_groups": int(sizes.gt(1).sum()),
        "duplicate_excess_rows": int((sizes - 1).sum()),
        "conflicting_label_groups": int(len(conflicting)),
        "rows_in_conflicting_groups": int(data[key].isin(conflicting).sum()),
    }


def clean_text_accuracy_ceiling(data: pd.DataFrame) -> float:
    """Empirical top-1 accuracy bound for deterministic functions of clean_text.

    Identical input can produce only one prediction. This is a dataset
    consistency diagnostic, not an achieved score, generalization estimate,
    bound on Macro-F1, or bound for models receiving additional information.
    """
    if data.empty:
        raise ValueError("Cannot compute consistency for an empty dataset")
    counts = data.groupby(["text_group", "medical_specialty"]).size()
    return float(counts.groupby(level=0).max().sum() / len(data))


def prepare_data(raw: pd.DataFrame, config: PipelineConfig) -> tuple:
    """Resolve raw-group labels within scope, then select by independent support.

    source_labels preserves the previous baseline as an explicit alternative.
    Neither policy invents a target or resolves ambiguity by dropping rare classes.
    """
    if config.top_k is not None and config.top_k < 2:
        raise ValueError("top_k must be at least 2 for classification.")
    if config.min_class_groups < 1:
        raise ValueError("min_class_groups must be >= 1")
    if config.conflict_policy not in {"group", "drop"}:
        raise ValueError("conflict_policy must be group or drop.")
    if config.benchmark_policy not in {"source_labels", "specialty_single_label"}:
        raise ValueError("Unknown benchmark_policy")
    specialty_mode = config.benchmark_policy == "specialty_single_label"
    if specialty_mode and config.conflict_policy != "group":
        raise ValueError("conflict_policy applies only to source_labels; specialty mode resolves raw groups first.")
    data = raw.copy()
    removed = []
    stages = [{"stage": "Raw input", "rows": len(data)}]

    def discard(mask: pd.Series, reason: str, stage: str) -> None:
        nonlocal data
        records = data.loc[mask, ["source_row", "medical_specialty"]].copy()
        records["reason"] = reason
        removed.append(records)
        data = data.loc[~mask].copy()
        stages.append({"stage": stage, "rows": len(data)})

    invalid = (data["transcription"].fillna("").str.strip().eq("")
               | data["medical_specialty"].fillna("").eq(""))
    discard(invalid, "missing_text_or_label", "Valid text and label")
    data["raw_group"] = data["transcription"].map(normalized_text).map(text_hash)
    raw_duplicates = duplicate_summary(data, "raw_group")
    valid_before = data.copy()
    if specialty_mode:
        decisions, scope = resolve_specialty_groups(valid_before, config.specialty_scope_path)
    cleaner = MedicalTextCleaner(config.keep_numbers, config.cleaning_profile)
    # Clean each unique raw string only once; common duplicated texts are expensive.
    mapping = {text: cleaner(text) for text in data["transcription"].unique()}
    data["clean_text"] = data["transcription"].map(mapping)
    discard(data["clean_text"].eq(""), "empty_after_cleaning", "Nonempty cleaned text")
    data["text_group"] = data["clean_text"].map(text_hash)
    # Preserve ALL usable records, including rare labels, before benchmark selection.
    catalog = data.copy()
    # Cleaning may collapse distinct raw strings; both representations must be isolated.
    clean_duplicates = duplicate_summary(data, "text_group")
    group_labels = data.groupby("text_group")["medical_specialty"].nunique()
    conflict_ids = group_labels[group_labels.gt(1)].index
    conflicts = data.loc[data["text_group"].isin(conflict_ids), [
        "source_row", "medical_specialty", "text_group", "raw_group"]].copy()
    specialty_audit = {}
    if specialty_mode:
        statuses = decisions.set_index("raw_group")["decision"]
        catalog["scope_decision"] = catalog.raw_group.map(statuses)
        discard(data.raw_group.map(statuses).eq("no_target_label"),
                "no_target_label", "Groups with target specialty")
        discard(data.raw_group.map(statuses).eq("ambiguous_target_labels"),
                "ambiguous_target_labels", "Exactly one target specialty per raw group")
        representatives = decisions.representative_source_row.dropna().astype(int)
        discard(~data.source_row.isin(representatives),
                "duplicate_resolved_document", "One original target-labelled row per raw group")
        # A cleaner collision is NOT evidence that the raw documents are duplicates.
        clean_labels = data.groupby("text_group").medical_specialty.nunique()
        collision_mask = data.text_group.isin(clean_labels[clean_labels.gt(1)].index)
        collision_rows = data.loc[collision_mask].copy()
        discard(collision_mask, "cleaning_label_collision", "Quarantine conflicting cleaner collisions")
        resolved = data.copy()
        specialty_audit = {
            "specialty_scope": scope,
            "specialty_resolution": {
                "raw_groups": len(decisions),
                **{str(k): int(v) for k, v in decisions.decision.value_counts().items()},
                "cleaning_collision_rows": len(collision_rows),
                "resolved_usable_documents": len(resolved),
            },
            # Transient export frames: runner removes these before JSON serialization.
            "_specialty_exports": {
                "decisions": decisions, "resolved": resolved, "cleaning_collisions": collision_rows,
            },
        }
    elif config.conflict_policy == "drop":
        discard(data["text_group"].isin(conflict_ids), "conflicting_labels", "Remove conflicts")
    # A lossy cleaner can collapse different source texts: group these together,
    # but do not discard them as source duplicates.
    discard(data.duplicated(["raw_group", "medical_specialty"]),
            "duplicate_text_and_label", "Deduplicate text + label")
    support = data.groupby("medical_specialty").agg(
        rows=("source_row", "size"), independent_groups=("text_group", "nunique")
    ).reindex(sorted(scope["included_labels"] if specialty_mode else valid_before.medical_specialty.unique()), fill_value=0).rename_axis(
        "medical_specialty").reset_index().sort_values(
        ["independent_groups", "medical_specialty"], ascending=[False, True])
    eligible = support.loc[support.independent_groups.ge(config.min_class_groups)]
    if config.top_k is not None:
        if len(eligible) < config.top_k:
            raise ValueError(f"Only {len(eligible)} classes meet min_class_groups; cannot select top {config.top_k}.")
        eligible = eligible.head(config.top_k)
    selected = eligible.medical_specialty.tolist()
    if len(selected) < 2:
        raise ValueError("Fewer than 2 eligible classes. Lower min_class_groups or review conflict policy.")
    support["selected"] = support.medical_specialty.isin(selected)
    support["reason"] = support.apply(lambda r: "selected" if r.selected else (
        "insufficient_independent_groups" if r.independent_groups < config.min_class_groups else "outside_top_k"), axis=1)
    discard(~data.medical_specialty.isin(support.loc[support.independent_groups.ge(config.min_class_groups), "medical_specialty"]),
            "insufficient_independent_groups", "Sufficient independent texts")
    if config.top_k is not None:
        discard(~data.medical_specialty.isin(selected), "outside_top_k", "Optional Top K")
    selected_before = valid_before.loc[valid_before.medical_specialty.isin(selected)].copy()
    # Full-data conflicts remain in the catalog even when a label is outside the benchmark.
    selected_group_labels = data.groupby("text_group")["medical_specialty"].nunique()
    data["label_ambiguous"] = data.text_group.isin(selected_group_labels[selected_group_labels.gt(1)].index)
    all_group_labels = catalog.groupby("text_group")["medical_specialty"].nunique()
    catalog["label_ambiguous_all_labels"] = catalog.text_group.isin(all_group_labels[all_group_labels.gt(1)].index)
    data["label_ambiguous_all_labels"] = data.text_group.isin(all_group_labels[all_group_labels.gt(1)].index)
    data["raw_word_count"] = data["transcription"].str.split().str.len()
    data["clean_word_count"] = data["clean_text"].str.split().str.len()
    data["raw_char_count"] = data["transcription"].str.len()
    data["clean_char_count"] = data["clean_text"].str.len()
    data = data.reset_index(drop=True)
    audit = {
        "selected_specialties": selected,
        "class_support": support.to_dict(orient="records"),
        "catalog_rows": len(catalog),
        "raw_duplicates_all_labels": raw_duplicates,
        "clean_duplicates_all_labels_before_dedup": clean_duplicates,
        "final_duplicates": duplicate_summary(data, "text_group"),
        **specialty_audit,
    }
    return (data, selected_before, pd.DataFrame(stages),
            pd.concat(removed, ignore_index=True), conflicts, audit, catalog)


def split_data(data: pd.DataFrame, seed: int = 42, min_holdout_groups: int = 1) -> pd.DataFrame:
    """Approximate stratified 60/20/20 split with strictly disjoint text groups."""
    group_counts = data.groupby("medical_specialty")["text_group"].nunique()
    if group_counts.min() < 5:
        raise ValueError("Each class needs at least 5 distinct texts for grouped 5-fold splitting. "
                         f"Counts: {group_counts.to_dict()}")
    result = data.copy()
    if min_holdout_groups < 1:
        raise ValueError("min_holdout_groups must be >= 1")
    result["split"] = "train"
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(splitter.split(
        data, data["medical_specialty"], groups=data["text_group"]
    )):
        if fold < 2:
            result.loc[result.index[indices], "split"] = "test" if fold == 0 else "validation"
    for split in ("train", "validation", "test"):
        if set(result.loc[result["split"].eq(split), "medical_specialty"]) != set(group_counts.index):
            raise ValueError(f"{split} is missing a class. Change seed or collect more distinct texts.")
    for key in ("text_group", "raw_group"):
        if result.groupby(key)["split"].nunique().max() != 1:
            raise AssertionError(f"Data leakage: {key} appears across splits.")
    for name in ("validation", "test"):
        counts = result.loc[result.split.eq(name)].groupby("medical_specialty").text_group.nunique()
        if counts.min() < min_holdout_groups:
            raise ValueError(f"{name} has fewer than {min_holdout_groups} independent groups in a class: "
                             f"{counts.to_dict()}. Increase support threshold or explicitly revise evaluation design.")
    train = result.loc[result.split.eq("train")]
    train_counts = train.groupby("medical_specialty").text_group.nunique()
    if train_counts.min() < 5:
        raise ValueError("Training partition has fewer than 5 independent texts for a class; "
                         "increase min_class_groups. Do not silently omit that class from CV.")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    expected = set(train.medical_specialty)
    for fit_indices, val_indices in cv.split(train, train.medical_specialty, train.text_group):
        if set(train.iloc[fit_indices].medical_specialty) != expected or set(train.iloc[val_indices].medical_specialty) != expected:
            raise ValueError("A training CV fold lacks a class; increase min_class_groups or review split design.")
    return result


def build_vectorizer(config: PipelineConfig, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """Input must be clean_text; preserve single tokens and alphanumeric compounds."""
    return TfidfVectorizer(
        ngram_range=ngram_range, min_df=config.min_df,
        max_features=config.max_features, lowercase=False,
        tokenizer=str.split, token_pattern=None, sublinear_tf=True,
    )


def build_model_pipeline(estimator, config: PipelineConfig,
                         ngram_range: tuple = (1, 2)) -> Pipeline:
    """Use on clean_text in CV so vocabulary and IDF are learned inside each fold."""
    return Pipeline([("tfidf", build_vectorizer(config, ngram_range)), ("model", estimator)])
