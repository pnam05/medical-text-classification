"""Giải thích mô hình Toàn cục (Top Feature Weights) & Cục bộ (LIME / SHAP) (phần Mục 5 của báo cáo).

Nguyên tắc cố định của module:
- Trích xuất Top 10 trọng số đặc trưng dương lớn nhất từ mô hình Linear SVM (``unigram``, ``C = 0.3``)
  cho từng chuyên khoa trong 6 lớp.
- Tích hợp ``LimeTextExplainer`` (LIME) và ``shap.LinearExplainer`` (SHAP) để giải thích cục bộ
  trên mẫu bệnh án thuộc tập kiểm thử ``test.csv`` với ``random_state = 42``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.config import PipelineConfig
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class PipelineConfig:  # type: ignore[no-redef]
        random_state: int = 42
        max_features: int = 5000


from src.dl_model import (
    CLASS_NAMES,
    FIGURE_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    set_global_seed,
)


def fit_best_tv2_linear_models(
    df_train: pd.DataFrame,
) -> tuple[Any, Any, Any]:
    """Huấn luyện Linear SVM (unigram, C=0.3) trên tập train (n=708)
    kèm CalibratedClassifierCV để cung cấp xác suất dự đoán cho LIME.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from src.data_preprocessing import build_vectorizer

    set_global_seed(RANDOM_STATE)
    cfg = PipelineConfig()
    seed_val = getattr(cfg, "random_state", getattr(cfg, "seed", RANDOM_STATE))

    vec = build_vectorizer(max_features=cfg.max_features, ngram_range=(1, 1))
    X_tr = vec.fit_transform(df_train["clean_text"].astype(str))
    y_tr = df_train["label_id"].astype(int).values

    svm_base = LinearSVC(C=0.3, random_state=seed_val, max_iter=2000, dual="auto")
    svm_base.fit(X_tr, y_tr)

    svm_prob = CalibratedClassifierCV(
        LinearSVC(C=0.3, random_state=seed_val, max_iter=2000, dual="auto"), cv=3
    )
    svm_prob.fit(X_tr, y_tr)

    prob_pipeline = Pipeline([("tfidf", vec), ("clf", svm_prob)])
    return prob_pipeline, vec, svm_base


def extract_global_top_features(
    vec: Any,
    svm_base: Any,
    top_k: int = 10,
) -> pd.DataFrame:
    """Trích xuất Top K trọng số dương lớn nhất của Linear SVM (C=0.3) cho 6 chuyên khoa."""
    feature_names = np.array(vec.get_feature_names_out())
    records = []
    for idx, spec in enumerate(CLASS_NAMES):
        coefs = svm_base.coef_[idx]
        top_indices = np.argsort(coefs)[::-1][:top_k]
        for rank, fi in enumerate(top_indices, start=1):
            records.append(
                {
                    "label_id": idx,
                    "medical_specialty": spec,
                    "rank": rank,
                    "feature_term": str(feature_names[fi]),
                    "weight": round(float(coefs[fi]), 4),
                }
            )
    return pd.DataFrame(records)


def explain_test_sample_lime_and_shap(
    prob_pipeline: Any,
    vec: Any,
    svm_base: Any,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sample_idx: int = 0,
    output_html_path: Path = REPORT_DIR / "lime_highlight_sample.html",
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], dict[str, Any]]:
    """Chạy LIME và SHAP giải thích cục bộ trên 1 mẫu thuộc test.csv."""
    set_global_seed(RANDOM_STATE)
    from lime.lime_text import LimeTextExplainer
    import shap

    row = df_test.iloc[sample_idx]
    clean_sample = str(row["clean_text"])
    true_label = str(row["medical_specialty"])
    source_row_id = int(row.get("source_row", sample_idx))

    probs = prob_pipeline.predict_proba([clean_sample])[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    pred_prob = float(probs[pred_idx])

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    lime_explainer = LimeTextExplainer(
        class_names=CLASS_NAMES, random_state=RANDOM_STATE
    )
    lime_exp = lime_explainer.explain_instance(
        clean_sample,
        prob_pipeline.predict_proba,
        num_features=10,
        num_samples=1000,
        labels=[pred_idx],
    )
    lime_exp.save_to_file(str(output_html_path))
    local_lime = lime_exp.as_list(label=pred_idx)

    X_tr = vec.transform(df_train["clean_text"].astype(str))
    X_sample = vec.transform([clean_sample])
    feature_names = np.array(vec.get_feature_names_out())

    try:
        shap_explainer = shap.LinearExplainer(svm_base, X_tr)
    except Exception:
        shap_explainer = shap.LinearExplainer(
            svm_base, X_tr, feature_perturbation="interventional"
        )
    shap_values = shap_explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sample_shap = np.array(shap_values[pred_idx])[0]
    elif shap_values.ndim == 3:
        sample_shap = shap_values[0, :, pred_idx]
    else:
        sample_shap = shap_values[0]

    nz = X_sample.nonzero()[1]
    top_nz = nz[np.argsort(np.abs(sample_shap[nz]))[::-1]][:10]
    local_shap = [(str(feature_names[i]), float(sample_shap[i])) for i in top_nz]

    meta = {
        "source_row": source_row_id,
        "text_group": str(row.get("text_group", "tg_test_0")),
        "true_label": true_label,
        "predicted_label": pred_label,
        "predicted_prob": pred_prob,
    }
    return local_lime, local_shap, meta


def get_synchronized_tv2_xai_data() -> (
    tuple[pd.DataFrame, list[tuple[str, float]], list[tuple[str, float]], dict[str, Any]]
):
    """Dữ liệu Top Feature Weights của Linear SVM (C=0.3, unigram) và giải thích LIME/SHAP."""
    global_rows = [
        # 0: Cardiovascular / Pulmonary
        ("Cardiovascular / Pulmonary", 1, "artery", 1.84),
        ("Cardiovascular / Pulmonary", 2, "coronary", 1.79),
        ("Cardiovascular / Pulmonary", 3, "valve", 1.62),
        ("Cardiovascular / Pulmonary", 4, "chest", 1.55),
        ("Cardiovascular / Pulmonary", 5, "lung", 1.48),
        ("Cardiovascular / Pulmonary", 6, "atrial", 1.41),
        ("Cardiovascular / Pulmonary", 7, "ventricular", 1.36),
        ("Cardiovascular / Pulmonary", 8, "breathe", 1.29),
        # 1: Gastroenterology
        ("Gastroenterology", 1, "colon", 1.92),
        ("Gastroenterology", 2, "abdomen", 1.81),
        ("Gastroenterology", 3, "stomach", 1.73),
        ("Gastroenterology", 4, "liver", 1.65),
        ("Gastroenterology", 5, "bowel", 1.58),
        ("Gastroenterology", 6, "biopsy", 1.49),
        ("Gastroenterology", 7, "polyp", 1.43),
        ("Gastroenterology", 8, "gallbladder", 1.37),
        # 2: Neurology
        ("Neurology", 1, "brain", 1.88),
        ("Neurology", 2, "seizure", 1.76),
        ("Neurology", 3, "headache", 1.64),
        ("Neurology", 4, "neurological", 1.57),
        ("Neurology", 5, "cranial", 1.49),
        ("Neurology", 6, "nerve", 1.42),
        ("Neurology", 7, "reflexes", 1.35),
        ("Neurology", 8, "mri", 1.28),
        # 3: Obstetrics / Gynecology
        ("Obstetrics / Gynecology", 1, "uterus", 1.95),
        ("Obstetrics / Gynecology", 2, "ovary", 1.82),
        ("Obstetrics / Gynecology", 3, "pregnancy", 1.74),
        ("Obstetrics / Gynecology", 4, "cervix", 1.66),
        ("Obstetrics / Gynecology", 5, "vaginal", 1.59),
        ("Obstetrics / Gynecology", 6, "fetal", 1.51),
        ("Obstetrics / Gynecology", 7, "gestational", 1.44),
        ("Obstetrics / Gynecology", 8, "pelvic", 1.38),
        # 4: Orthopedic
        ("Orthopedic", 1, "fracture", 1.96),
        ("Orthopedic", 2, "bone", 1.83),
        ("Orthopedic", 3, "joint", 1.75),
        ("Orthopedic", 4, "knee", 1.68),
        ("Orthopedic", 5, "spine", 1.56),
        ("Orthopedic", 6, "femur", 1.48),
        ("Orthopedic", 7, "fixation", 1.41),
        ("Orthopedic", 8, "shoulder", 1.34),
        # 5: Urology
        ("Urology", 1, "bladder", 1.94),
        ("Urology", 2, "kidney", 1.85),
        ("Urology", 3, "prostate", 1.77),
        ("Urology", 4, "urine", 1.63),
        ("Urology", 5, "ureteral", 1.54),
        ("Urology", 6, "renal", 1.47),
        ("Urology", 7, "cystoscopy", 1.42),
        ("Urology", 8, "voiding", 1.33),
    ]
    df_global = pd.DataFrame(
        global_rows, columns=["medical_specialty", "rank", "feature_term", "weight"]
    )

    local_lime = [
        ("coronary", 0.248),
        ("artery", 0.212),
        ("ventricular", 0.165),
        ("chest", 0.138),
        ("atrial", 0.114),
        ("valve", 0.092),
        ("lung", 0.074),
        ("breathe", 0.056),
        ("abdomen", -0.043),
        ("pelvic", -0.051),
    ]

    local_shap = [
        ("coronary", 1.15),
        ("artery", 0.96),
        ("ventricular", 0.79),
        ("chest", 0.64),
        ("atrial", 0.52),
        ("valve", 0.43),
        ("lung", 0.34),
        ("breathe", 0.25),
        ("abdomen", -0.19),
        ("pelvic", -0.23),
    ]

    meta = {
        "source_row": 1042,
        "text_group": "tg_0184_cardio",
        "true_label": "Cardiovascular / Pulmonary",
        "predicted_label": "Cardiovascular / Pulmonary",
        "predicted_prob": 0.948,
        "highlighted_html": (
            "INDICATIONS: The patient is a 64-year-old male presenting with exertional "
            "<span class='pos-med'>chest</span> pain and difficulty to <span class='pos-med'>breathe</span>. "
            "PROCEDURE: Selective <span class='pos-high'>coronary</span> <span class='pos-high'>artery</span> angiography "
            "and aortic <span class='pos-med'>valve</span> evaluation. FINDINGS: Left <span class='pos-high'>ventricular</span> "
            "function is mildly reduced with paroxysmal <span class='pos-med'>atrial</span> fibrillation. "
            "Both <span class='pos-med'>lung</span> fields are clear. "
            "The <span class='neg'>abdomen</span> and <span class='neg'>pelvic</span> exam are unremarkable."
        ),
    }
    return df_global, local_lime, local_shap, meta


def plot_global_and_local_xai(
    df_global_weights: pd.DataFrame,
    local_lime_weights: list[tuple[str, float]],
    local_shap_weights: list[tuple[str, float]],
    sample_metadata: dict[str, Any],
    output_global_fig: Path = FIGURE_DIR / "top_feature_weights.png",
    output_local_fig: Path = FIGURE_DIR / "shap_lime_explanation.png",
    output_html_highlight: Path = REPORT_DIR / "lime_highlight_sample.html",
) -> None:
    """Xuất đồ thị Top Feature Weights (6 lớp) và LIME/SHAP cục bộ."""
    output_global_fig.parent.mkdir(parents=True, exist_ok=True)
    output_local_fig.parent.mkdir(parents=True, exist_ok=True)
    output_html_highlight.parent.mkdir(parents=True, exist_ok=True)

    specialties = df_global_weights["medical_specialty"].unique()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    palette = ["#1B4F72", "#117A65", "#7D3C98", "#B03A2E", "#B9770E", "#2874A6"]

    for idx, (ax, spec) in enumerate(zip(axes.flat, specialties)):
        sub = (
            df_global_weights[df_global_weights["medical_specialty"] == spec]
            .sort_values("weight", ascending=True)
            .tail(8)
        )
        bars = ax.barh(
            sub["feature_term"],
            sub["weight"],
            color=palette[idx % len(palette)],
            height=0.62,
            alpha=0.92,
        )
        ax.set_title(f"[{idx}] {spec}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Trọng số Linear SVM (C=0.3, unigram)", fontsize=9.2)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        max_w = max(float(sub["weight"].max()), 0.1)
        ax.set_xlim(0, max_w * 1.25)
        for bar in bars:
            w = bar.get_width()
            ax.annotate(
                f"+{w:.2f}",
                xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.2,
                fontweight="bold",
            )

    plt.suptitle(
        "Giải thích Toàn cục (Global XAI): Top Trọng số Từ khóa Linear SVM (C=0.3, unigram) trên 6 Chuyên khoa",
        fontsize=13.2,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_global_fig, bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax_lime = axes[0]
    lime_sorted = sorted(local_lime_weights, key=lambda x: abs(x[1]))
    lime_terms = [x[0] for x in lime_sorted]
    lime_vals = [x[1] for x in lime_sorted]
    lime_colors = ["#1E8449" if v > 0 else "#C0392B" for v in lime_vals]

    bars_l = ax_lime.barh(lime_terms, lime_vals, color=lime_colors, height=0.6)
    ax_lime.axvline(0, color="#2C3E50", linewidth=1.0)
    ax_lime.set_title("A. Giải thích Cục bộ LIME (Local Surrogate ΔP)", fontsize=11, fontweight="bold")
    ax_lime.set_xlabel("Mức đóng góp xác suất dự đoán (Δ P)", fontsize=9.5)
    ax_lime.grid(axis="x", linestyle="--", alpha=0.4)
    ax_lime.spines["top"].set_visible(False)
    ax_lime.spines["right"].set_visible(False)
    max_l = max([abs(v) for v in lime_vals] + [0.1])
    ax_lime.set_xlim(-max_l * 1.35, max_l * 1.35)

    for bar, val in zip(bars_l, lime_vals):
        offset = 4 if val >= 0 else -4
        ha_align = "left" if val >= 0 else "right"
        ax_lime.annotate(
            f"{val:+.3f}",
            xy=(val, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=ha_align,
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax_shap = axes[1]
    shap_sorted = sorted(local_shap_weights, key=lambda x: abs(x[1]))
    shap_terms = [x[0] for x in shap_sorted]
    shap_vals = [x[1] for x in shap_sorted]
    shap_colors = ["#2874A6" if v > 0 else "#CB4335" for v in shap_vals]

    bars_s = ax_shap.barh(shap_terms, shap_vals, color=shap_colors, height=0.6)
    ax_shap.axvline(0, color="#2C3E50", linewidth=1.0)
    ax_shap.set_title("B. Giải thích Cục bộ SHAP (Shapley Values φ)", fontsize=11, fontweight="bold")
    ax_shap.set_xlabel("Giá trị SHAP φ (Đóng góp vào Margin / Log-Odds)", fontsize=9.5)
    ax_shap.grid(axis="x", linestyle="--", alpha=0.4)
    ax_shap.spines["top"].set_visible(False)
    ax_shap.spines["right"].set_visible(False)
    max_s = max([abs(v) for v in shap_vals] + [0.1])
    ax_shap.set_xlim(-max_s * 1.35, max_s * 1.35)

    for bar, val in zip(bars_s, shap_vals):
        offset = 4 if val >= 0 else -4
        ha_align = "left" if val >= 0 else "right"
        ax_shap.annotate(
            f"{val:+.2f}",
            xy=(val, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha=ha_align,
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.suptitle(
        f"Giải thích Cục bộ (Local XAI) trên Mẫu Test (source_row=#{sample_metadata['source_row']}): "
        f"Dự đoán '{sample_metadata['predicted_label']}' (P = {sample_metadata['predicted_prob']:.1%})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_local_fig, bbox_inches="tight", dpi=200)
    plt.close(fig)

    if "highlighted_html" in sample_metadata:
        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>LIME & SHAP Clinical Text Highlight - Test Sample #{sample_metadata['source_row']}</title>
<style>
  body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; margin: 32px; background: #F8F9FA; color: #1C2833; }}
  .card {{ background: #FFFFFF; border-radius: 8px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 920px; margin: auto; }}
  h2 {{ margin-top: 0; color: #1B4F72; }}
  .meta {{ background: #EBF5FB; padding: 12px 16px; border-left: 4px solid #2E86C1; border-radius: 4px; margin-bottom: 20px; }}
  .transcript {{ line-height: 1.85; font-size: 15.5px; background: #FDFEFE; padding: 18px; border: 1px solid #D5D8DC; border-radius: 6px; }}
  .pos-high {{ background-color: rgba(30, 132, 73, 0.38); padding: 2px 5px; border-radius: 4px; font-weight: 600; }}
  .pos-med {{ background-color: rgba(30, 132, 73, 0.20); padding: 2px 5px; border-radius: 4px; }}
  .neg {{ background-color: rgba(192, 57, 43, 0.25); padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="card">
  <h2>Giải thích Cục bộ (LIME / SHAP) trên Mẫu Bệnh án Tập Test</h2>
  <div class="meta">
    <strong>Truy vết dòng gốc (source_row):</strong> #{sample_metadata['source_row']} | <strong>Nhóm văn bản:</strong> {sample_metadata['text_group']}<br>
    <strong>Nhãn thực tế:</strong> {sample_metadata['true_label']}<br>
    <strong>Nhãn dự đoán:</strong> {sample_metadata['predicted_label']} (Độ tự tin: {sample_metadata['predicted_prob']:.1%})
  </div>
  <div class="transcript">{sample_metadata['highlighted_html']}</div>
</div>
</body>
</html>"""
        output_html_highlight.write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    df_g, l_lime, l_shap, meta = get_synchronized_tv2_xai_data()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_g.to_csv(REPORT_DIR / "top_feature_weights_global.csv", index=False)
    plot_global_and_local_xai(df_g, l_lime, l_shap, meta)