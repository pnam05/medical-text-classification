"""Save descriptive tables and publication-friendly PNGs without a GUI."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

from src.config import ROOT

# Keep font/config caches in the project, including restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

from src.data_preprocessing import NEGATIONS, TOKEN_RE

sns.set_theme(style="whitegrid", palette="Set2", font="DejaVu Sans")

FIGURE_CAPTIONS = {
    "01_raw_class_distribution": "Phân bố toàn bộ nhãn trước xử lý; các lớp đủ điều kiện đánh giá được tô màu.",
    "02_missing_values": "Số ô thiếu hoặc chỉ chứa khoảng trắng theo cột gốc.",
    "03_raw_text_lengths": "Phân bố số từ và số ký tự của văn bản gốc có nội dung.",
    "04_raw_length_by_class": "Độ dài văn bản gốc theo các nhãn được chọn; không ẩn các ngoại lệ.",
    "05_cleaning_flow": "Số bản ghi còn lại sau từng bước lọc và khử trùng.",
    "06_class_distribution_before_after": "Các lớp đủ điều kiện trước và sau xử lý. Không cân bằng bằng cách bỏ lớp lớn.",
    "07_paired_lengths": "So sánh độ dài trên chính các bản ghi được giữ lại, tránh thay đổi thành phần mẫu.",
    "08_clean_length_by_class": "Phân bố độ dài sau xử lý theo từng nhãn.",
    "09_frequent_words": "Từ phổ biến trước và sau làm sạch trên cùng tập bản ghi; có tính cả từ dừng ở phía trước.",
    "10_frequent_bigrams": "Các bigram phổ biến sau làm sạch; từ dừng bị loại có thể làm hai từ trở thành liền kề.",
    "11_negation_retention": "Số văn bản có từ phủ định trước và sau xử lý; trước được mở rộng dạng viết tắt.",
    "12_conflicting_labels": "Xung đột nhãn NGUỒN trước giải quyết phạm vi, giới hạn hiển thị ở các lớp benchmark; không phải xung đột còn lại sau xử lý. Đường chéo bằng 0.",
    "13_split_distribution": "Số mẫu theo nhãn trong train/validation/test, chia theo nhóm trùng văn bản.",
    "14_train_tfidf_terms": "Đặc trưng có TF-IDF trung bình cao theo lớp trên train; đây không phải feature importance của mô hình.",
    "15_train_svd_projection": "Chiếu TF-IDF train xuống 2 chiều bằng TruncatedSVD; chỉ để khám phá, không đo chất lượng phân loại.",
    "16_specialty_resolution": "Phân loại các nhóm văn bản gốc theo số nhãn mục tiêu trước xét ngưỡng support; đơn vị là nhóm văn bản, không phải dòng CSV.",
    "17_specialty_support": "Số nhóm văn bản sạch độc lập của mọi chuyên khoa mục tiêu sau giải quyết nhãn; lớp chưa đủ support vẫn được lưu riêng.",
    "18_ambiguity_before_after": "Nhóm đa nhãn nguồn, nhóm còn nhiều nhãn mục tiêu và nhóm đa nhãn trong benchmark. Các cột thuộc các phạm vi khác nhau, không phải metric mô hình.",
}


def save_figure(fig, folder: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_specialty_eda(decisions, support, processed, output: Path) -> None:
    figures = output / "figures"
    counts = decisions.decision.value_counts().reindex(
        ["no_target_label", "ambiguous_target_labels", "single_target_label"], fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.barh(["Outside target scope", "Multiple target labels: review", "One target label"],
                   counts.values, color=["#9ca3af", "#db9565", "#328a9e"])
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Unique normalized raw documents", title="Scope resolution before support filtering")
    ax.margins(x=.15)
    save_figure(fig, figures, "16_specialty_resolution")

    ordered = support.sort_values("independent_groups")
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(ordered.medical_specialty, ordered.independent_groups,
                  color=["#328a9e" if x else "#9ca3af" for x in ordered.selected])
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Independent clean-text groups", title="Resolved specialty support (teal = benchmark)")
    ax.margins(x=.12)
    save_figure(fig, figures, "17_specialty_support")

    import json
    values = [int(decisions.original_labels.map(lambda s: len(json.loads(s)) > 1).sum()),
              int(decisions.target_label_count.gt(1).sum()),
              int(processed.groupby("text_group").medical_specialty.nunique().gt(1).sum())]
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.barh(["All raw groups: multiple source labels", "All raw groups: multiple target labels",
                   "Final benchmark: multiple target labels"], values, color=["#9ca3af", "#db9565", "#328a9e"])
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Conflicting groups", title="Label ambiguity at different stages")
    ax.margins(x=.15)
    save_figure(fig, figures, "18_ambiguity_before_after")


def frequency_table(texts, n: int = 1, already_clean: bool = False) -> pd.DataFrame:
    counts = Counter()
    for text in texts:
        tokens = str(text).split() if already_clean else TOKEN_RE.findall(str(text).lower())
        counts.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return pd.DataFrame(counts.most_common(), columns=["term", "count"])


def horizontal_terms(ax, table: pd.DataFrame, title: str, value: str = "count", top: int = 20):
    subset = table.head(top).iloc[::-1]
    ax.barh(subset["term"], subset[value], color="#328a9e")
    ax.set(title=title, xlabel="Occurrences" if value == "count" else "Mean TF-IDF")


def generate_eda(raw, selected_before, processed, stages, conflicts,
                 vectorizer, x_train, output: Path, seed: int) -> dict:
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    selected = selected_before["medical_specialty"].value_counts().index.tolist()
    raw_counts = raw["medical_specialty"].fillna("<missing>").replace("", "<missing>").value_counts()
    raw_counts.rename_axis("medical_specialty").to_csv(tables / "raw_class_counts.csv", header=["rows"])
    fig, ax = plt.subplots(figsize=(12, max(7, len(raw_counts) * .27)))
    colors = ["#328a9e" if label in selected else "#c5cbd0" for label in raw_counts.index[::-1]]
    bars = ax.barh(raw_counts.index[::-1], raw_counts.values[::-1], color=colors)
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set(xlabel="Records", title=f"Raw label distribution ({len(raw)} records, {len(raw_counts)} labels)")
    ax.margins(x=.12)
    save_figure(fig, figures, "01_raw_class_distribution")

    original_cols = [col for col in raw if col != "source_row"]
    missing = pd.DataFrame({
        "missing_rows": [int(raw[c].isna().sum() + raw[c].dropna().astype(str).str.strip().eq("").sum())
                         for c in original_cols],
    }, index=pd.Index(original_cols, name="column"))
    missing["missing_percent"] = 100 * missing["missing_rows"] / len(raw)
    missing.to_csv(tables / "missing_values.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(missing.index, missing["missing_rows"], color="#db9565")
    ax.bar_label(bars, labels=[f"{n} ({p:.1f}%)" for n, p in missing.itertuples(index=False)], padding=3)
    ax.set(xlabel="Missing / blank cells", title="Data completeness")
    ax.margins(x=.25)
    save_figure(fig, figures, "02_missing_values")

    nonempty = raw.loc[raw["transcription"].fillna("").str.strip().ne(""), "transcription"]
    raw_lengths = pd.DataFrame({"words": nonempty.str.split().str.len(), "characters": nonempty.str.len()})
    raw_lengths.describe(percentiles=[.01, .25, .5, .75, .95, .99]).to_csv(tables / "raw_length_statistics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, column in zip(axes, raw_lengths):
        sns.histplot(raw_lengths[column], bins=60, ax=ax, color="#328a9e")
        ax.axvline(raw_lengths[column].median(), color="#c25644", linestyle="--", label="Median")
        ax.set(title=f"Raw text: {column}", xlabel=column.title())
        ax.legend()
    save_figure(fig, figures, "03_raw_text_lengths")

    before = selected_before.assign(word_count=selected_before["transcription"].str.split().str.len())
    fig, ax = plt.subplots(figsize=(12, max(4, .6 * len(selected))))
    sns.boxplot(data=before, y="medical_specialty", x="word_count", order=selected, ax=ax, color="#89c7cb", fliersize=2)
    ax.set(title="Raw length by selected label", ylabel="", xlabel="Whitespace-separated words")
    save_figure(fig, figures, "04_raw_length_by_class")

    stages.to_csv(tables / "cleaning_flow.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(stages["stage"][::-1], stages["rows"][::-1], color="#328a9e")
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Remaining records", title="Preprocessing audit")
    ax.margins(x=.15)
    save_figure(fig, figures, "05_cleaning_flow")

    class_comparison = pd.DataFrame({
        "before": selected_before["medical_specialty"].value_counts(),
        "after": processed["medical_specialty"].value_counts(),
    }).reindex(selected).fillna(0).astype(int)
    class_comparison.to_csv(tables / "class_counts_before_after.csv")
    fig, ax = plt.subplots(figsize=(12, max(5, .5 * len(selected))))
    class_comparison.iloc[::-1].plot.barh(ax=ax, color=["#c5cbd0", "#328a9e"])
    ax.set(title="Selected labels before / after preprocessing", ylabel="", xlabel="Records")
    save_figure(fig, figures, "06_class_distribution_before_after")

    length_columns = ["raw_word_count", "clean_word_count", "raw_char_count", "clean_char_count"]
    processed[length_columns].describe(percentiles=[.01, .25, .5, .75, .95, .99]).to_csv(tables / "paired_length_statistics.csv")
    processed.groupby("medical_specialty")[length_columns].agg(["count", "mean", "median", "min", "max"]).to_csv(tables / "lengths_by_class.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, processed["raw_word_count"].max() + 1, 50)
    axes[0].hist(processed["raw_word_count"], bins=bins, alpha=.5, label="Before")
    axes[0].hist(processed["clean_word_count"], bins=bins, alpha=.6, label="After")
    axes[0].set(title="Same records: word count", xlabel="Words", ylabel="Records")
    axes[0].legend()
    axes[1].scatter(processed["raw_word_count"], processed["clean_word_count"], alpha=.25, s=12)
    limit = int(processed["raw_word_count"].max())
    axes[1].plot([0, limit], [0, limit], "--", color="gray", label="Equal length")
    axes[1].set(title="Paired lengths", xlabel="Before", ylabel="After")
    axes[1].legend()
    save_figure(fig, figures, "07_paired_lengths")

    fig, ax = plt.subplots(figsize=(12, max(4, .6 * len(selected))))
    sns.boxplot(data=processed, y="medical_specialty", x="clean_word_count", order=selected, ax=ax, color="#89c7cb", fliersize=2)
    ax.set(title="Cleaned length by label", ylabel="", xlabel="Clean tokens")
    save_figure(fig, figures, "08_clean_length_by_class")

    raw_freq = frequency_table(processed["transcription"])
    clean_freq = frequency_table(processed["clean_text"], already_clean=True)
    bigrams = frequency_table(processed["clean_text"], n=2, already_clean=True)
    for name, table in [("raw_word_frequencies", raw_freq), ("clean_word_frequencies", clean_freq), ("clean_bigram_frequencies", bigrams)]:
        table.to_csv(tables / f"{name}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    horizontal_terms(axes[0], raw_freq, "Before: same retained records")
    horizontal_terms(axes[1], clean_freq, "After: same retained records")
    save_figure(fig, figures, "09_frequent_words")
    fig, ax = plt.subplots(figsize=(11, 8))
    horizontal_terms(ax, bigrams, "Most frequent cleaned bigrams")
    save_figure(fig, figures, "10_frequent_bigrams")

    # Match the cleaner's negation expansion without removing stopwords or lemmatizing.
    import re
    def negation_tokens(text):
        text = str(text).lower().replace("\u2019", "'")
        text = re.sub(r"\b(?:cannot|can't)\b", "can not", text)
        text = re.sub(r"\bwon't\b", "will not", text)
        return set(TOKEN_RE.findall(re.sub(r"n't\b", " not", text)))
    before_sets = [negation_tokens(t) for t in processed["transcription"]]
    after_sets = [set(t.split()) for t in processed["clean_text"]]
    negations = pd.DataFrame({
        "term": sorted(NEGATIONS),
        "before_documents": [sum(t in s for s in before_sets) for t in sorted(NEGATIONS)],
        "after_documents": [sum(t in s for s in after_sets) for t in sorted(NEGATIONS)],
    }).set_index("term")
    negations.to_csv(tables / "negation_retention.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    negations.plot.barh(ax=ax, color=["#c5cbd0", "#328a9e"])
    ax.set(title="Negation document frequency on paired records", xlabel="Documents", ylabel="")
    save_figure(fig, figures, "11_negation_retention")

    overlap = pd.DataFrame(0, index=selected, columns=selected)
    for _, group in conflicts.loc[conflicts.medical_specialty.isin(selected)].groupby("text_group"):
        labels = group["medical_specialty"].unique()
        for first in labels:
            for second in labels:
                if first != second:
                    overlap.loc[first, second] += 1
    overlap.to_csv(tables / "conflicting_label_overlap.csv")
    fig, ax = plt.subplots(figsize=(max(11, .55 * len(selected)), max(9, .5 * len(selected))))
    sns.heatmap(overlap, annot=True, fmt="d", cmap="YlOrRd", ax=ax, annot_kws={"size": 6 if len(selected) > 12 else 10})
    ax.set(title="Conflicting text groups across labels (before conflict policy)")
    save_figure(fig, figures, "12_conflicting_labels")

    split_counts = pd.crosstab(processed["medical_specialty"], processed["split"]).reindex(
        index=selected, columns=["train", "validation", "test"], fill_value=0)
    split_counts.to_csv(tables / "split_class_counts.csv")
    (split_counts / split_counts.sum(axis=0) * 100).to_csv(tables / "split_class_percentages.csv")
    fig, ax = plt.subplots(figsize=(12, max(6, .55 * len(selected))))
    split_counts.plot.barh(ax=ax)
    ax.set(title="Grouped split: approximate 60 / 20 / 20", xlabel="Records", ylabel="")
    save_figure(fig, figures, "13_split_distribution")

    train = processed.loc[processed["split"].eq("train")]
    feature_names = vectorizer.get_feature_names_out()
    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, rows * 4), squeeze=False)
    term_tables = []
    for ax, label in zip(axes.flat, selected):
        mask = train["medical_specialty"].eq(label).to_numpy()
        mean = np.asarray(x_train[mask].mean(axis=0)).ravel()
        top = np.argsort(mean)[-12:][::-1]
        table = pd.DataFrame({"term": feature_names[top], "mean_tfidf": mean[top]})
        horizontal_terms(ax, table, label, value="mean_tfidf", top=12)
        term_tables.append(table.assign(medical_specialty=label))
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    pd.concat(term_tables, ignore_index=True).to_csv(tables / "train_top_tfidf_by_class.csv", index=False)
    save_figure(fig, figures, "14_train_tfidf_terms")

    projection_info = {"status": "skipped: fewer than 2 features or 3 train rows"}
    if min(x_train.shape) >= 3:
        svd = TruncatedSVD(n_components=2, random_state=seed)
        coordinates = svd.fit_transform(x_train)
        projection = pd.DataFrame(coordinates, columns=["component_1", "component_2"])
        projection["medical_specialty"] = train["medical_specialty"].to_numpy()
        projection["source_row"] = train["source_row"].to_numpy()
        projection.to_csv(tables / "train_svd_coordinates.csv", index=False)
        explained = float(svd.explained_variance_ratio_.sum())
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(data=projection, x="component_1", y="component_2", hue="medical_specialty",
                        hue_order=selected, palette=sns.color_palette("husl", len(selected)), alpha=.5, s=22, ax=ax)
        ax.set(title=f"Train TF-IDF: 2D SVD ({explained:.1%} explained variance)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        save_figure(fig, figures, "15_train_svd_projection")
        projection_info = {"status": "ok", "explained_variance_ratio_sum": explained}
    return {
        "raw_vocabulary_retained_records": int(len(raw_freq)),
        "clean_vocabulary_retained_records": int(len(clean_freq)),
        "svd": projection_info,
        "missing_values": missing["missing_rows"].to_dict(),
    }
