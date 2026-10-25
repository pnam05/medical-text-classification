"""Runner phần classic ML: sinh toàn bộ bảng và hình cho Mục 3-4 của báo cáo.

Chạy từ thư mục gốc project:

    .\\.venv\\Scripts\\python.exe -m scripts.run_classic_ml

Đầu ra (trong ``outputs/preprocessing``):

- ``tables/classic_models_comparison.csv``  CV 5-fold + validation cho 2 biến thể x 5 mô hình
- ``tables/classic_grid_results.csv``       toàn bộ lưới siêu tham số (vẽ đường cong)
- ``tables/model_selection.csv``            quy tắc 1-SE và cấu hình được chốt
- ``tables/per_class_validation.csv``       precision/recall/F1/support từng lớp
- ``tables/per_class_test.csv``             như trên, cho tập test
- ``tables/error_samples_validation.csv``   các văn bản dự đoán sai, kèm độ tin cậy
- ``tables/l1_feature_selection.csv``       top token theo |w| của LR L1
- ``tables/l1_summary.csv``                 số đặc trưng L1 giữ lại ở C tốt nhất và C thưa
- ``tables/l1_sweep.csv``                   quét C: số đặc trưng giữ lại so với CV macro-F1
- ``tables/grid_boundary_check.csv``        kiểm tra C vượt biên trên của lưới tune
- ``tables/compute_cost.csv``               thời gian fit, độ trễ, kích thước model
- ``tables/ablation.csv``                   từng thay đổi một so với mốc
- ``tables/final_test_metrics.csv``         test dùng một lần + CI 95% bootstrap
- ``tables/majority_baseline.csv``          mốc lớp đông nhất
- ``tables/classic_summary.json``           bản tóm tắt máy đọc cho phần viết báo cáo
- ``figures/19_hparam_alpha.png``, ``20_hparam_C.png``, ``21_l1_top_terms.png``
- ``figures/confusion_val_*.png``, ``confusion_test_*.png``
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                       # phải đặt trước khi import pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.classic_models import (ROOT, VARIANTS, assert_alignment, bootstrap_macro_f1,
                                confidence, confusion, cost, cv_folds, grid_search,
                                grid_table, label_names, l1_terms, load_split,
                                majority_baseline, make_pipeline, metrics, model_zoo,
                                per_class_table, select_by_one_se, summarize_search)

# scikit-learn 1.9 cảnh báo penalty="l1" đã deprecated (thay bằng l1_ratio=1.0 ở 1.10);
# repo vẫn giữ cách viết tương thích 1.5-1.9 nên tắt cảnh báo để log dễ đọc.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*Inconsistent values.*")
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*")

TABLES, FIGURES, MODELS = ROOT / "tables", ROOT / "figures", ROOT / "models"
SEED = 42#: C quét cho phần đọc hệ số L1 (tăng dần) và C kiểm tra biên trên của lưới tune.
L1_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def announce(text: str) -> None:
    print(f"\n>>> {text}", flush=True)


def plot_confusion(matrix, names, title, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regularization(grid: pd.DataFrame, param: str, path: Path, title: str) -> None:
    """Đường cong CV macro-F1 theo một siêu tham số, mỗi biến thể một panel."""
    models = sorted(grid.model.unique())
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(11, 4.2), sharey=True)
    for ax, variant in zip(np.atleast_1d(axes), VARIANTS):
        for color, model in zip(COLORS, [m for m in models if m != "mnb_baseline"]):
            part = grid[(grid.variant == variant) & (grid.model == model) & (grid.param == param)]
            if part.empty:
                continue
            part = part.sort_values("value")
            ax.errorbar(part.value, part.cv_macro_f1, yerr=part.cv_std, marker="o",
                        capsize=3, color=color, label=model)
        ax.set_xscale("log")
        ax.set_xlabel(f"{param} (thang log)")
        ax.set_title(variant)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("CV macro-F1 (mean ± std, 5 fold)")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_l1(terms: pd.DataFrame, names: dict[int, str], path: Path, subtitle: str) -> None:
    """Top token theo |w| của LR L1, mỗi lớp một panel."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, class_id in zip(axes.ravel(), range(len(names))):
        part = terms[terms.class_id == class_id].head(5).iloc[::-1]
        ax.barh(part.term, part.coef, color=COLORS[class_id % len(COLORS)])
        ax.set_title(f"{class_id}: {names[class_id]}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.axvline(0, color="black", linewidth=0.6)
    fig.suptitle(subtitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_ablation(texts, y, folds) -> pd.DataFrame:
    """Mỗi lần chỉ đổi một yếu tố so với mốc SVM-L2 C=1.0 trên unigram."""
    base = dict(C=1.0, class_weight="balanced", random_state=SEED, max_iter=5000)
    settings = [
        ("Biến thể đặc trưng", "unigram (mốc)", "unigram", {}),
        ("Biến thể đặc trưng", "unigram+bigram", "unigram_bigram", {}),
        ("min_df", "3 (mốc)", "unigram", {}),
        ("min_df", "1", "unigram", {"tfidf__min_df": 1}),
        ("min_df", "5", "unigram", {"tfidf__min_df": 5}),
        ("max_features", "5000 (mốc)", "unigram", {}),
        ("max_features", "2000", "unigram", {"tfidf__max_features": 2000}),
        ("max_features", "10000", "unigram", {"tfidf__max_features": 10000}),
        ("sublinear_tf", "True (mốc)", "unigram", {}),
        ("sublinear_tf", "False", "unigram", {"tfidf__sublinear_tf": False}),
        ("class_weight", "balanced (mốc)", "unigram", {}),
        ("class_weight", "None", "unigram", {}),
    ]

    rows = []
    for experiment, setting, variant, overrides in settings:
        estimator = LinearSVC(**base)
        if experiment == "class_weight" and setting == "None":
            estimator.set_params(class_weight=None)
        pipeline = make_pipeline(estimator, variant, **overrides)
        search = GridSearchCV(pipeline, {}, cv=folds, scoring="f1_macro", n_jobs=1,
                              error_score="raise").fit(texts, y)
        mean = float(search.cv_results_["mean_test_score"][0])
        std = float(search.cv_results_["std_test_score"][0])
        rows.append({"experiment": experiment, "setting": setting, "variant": variant,
                     "cv_macro_f1": round(mean, 4), "cv_std": round(std, 4)})

    table = pd.DataFrame(rows)
    # Mốc của mỗi thí nghiệm là dòng có "(mốc)" trong tên setting.
    baselines = table[table.setting.str.contains(r"\(mốc\)")].set_index("experiment").cv_macro_f1
    table["delta_vs_baseline"] = [
        round(value - baselines[experiment], 4)
        for experiment, value in zip(table.experiment, table.cv_macro_f1)
    ]
    return table


def main() -> None:
    started = time.perf_counter()
    for folder in (TABLES, FIGURES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)

    assert_alignment("train")
    names = label_names()
    labels = [names[i] for i in sorted(names)]

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")
    texts, y = train.clean_text.astype(str).to_numpy(), train.label_id.to_numpy()
    val_texts, y_val = validation.clean_text.astype(str).to_numpy(), validation.label_id.to_numpy()
    test_texts, y_test = test.clean_text.astype(str).to_numpy(), test.label_id.to_numpy()
    folds = cv_folds()

    announce(f"Train {len(train)} / validation {len(validation)} / test {len(test)}; "
             f"{len(folds)} fold CV")

    # ---- 1. CV + validation cho toàn bộ mô hình -----------------------------
    comparison_rows, grids, fitted = [], [], {}
    for variant in VARIANTS:
        for model_name, (estimator, grid) in model_zoo(SEED).items():
            tick = time.perf_counter()
            search = grid_search(estimator, grid, variant, texts, y, folds, SEED)
            row = summarize_search(search, variant, model_name, SEED)
            val = metrics(y_val, search.best_estimator_.predict(val_texts))
            row.update({
                "val_macro_f1": val["macro_f1"],
                "val_weighted_f1": val["weighted_f1"],
                "val_accuracy": val["accuracy"],
            })
            comparison_rows.append(row)
            grids.append(grid_table(search, variant, model_name))
            fitted[(variant, model_name)] = search.best_estimator_
            print(f"{variant:15s} {model_name:13s} CV={row['cv_macro_f1']:.4f}"
                  f"±{row['cv_std']:.4f} val={row['val_macro_f1']:.4f}"
                  f" ({time.perf_counter() - tick:.1f}s)", flush=True)

    comparison = pd.DataFrame(comparison_rows)
    comparison["params"] = comparison.params.apply(lambda p: json.dumps(p, sort_keys=True))
    comparison.to_csv(TABLES / "classic_models_comparison.csv", index=False)
    grid_results = pd.concat(grids, ignore_index=True)
    grid_results.to_csv(TABLES / "classic_grid_results.csv", index=False)

    announce("Vẽ đường cong regularization")
    plot_regularization(grid_results, "alpha", FIGURES / "19_hparam_alpha.png",
                        "MNB: CV macro-F1 theo alpha làm mượt")
    plot_regularization(grid_results, "C", FIGURES / "20_hparam_C.png",
                        "SVM / Logistic Regression: CV macro-F1 theo C")

    # ---- 2. Chọn cấu hình bằng quy tắc 1-SE ---------------------------------
    selection = select_by_one_se(comparison)
    selection.to_csv(TABLES / "model_selection.csv", index=False)
    chosen_row = selection.loc[selection.chosen].iloc[0]
    chosen_variant, chosen_model = chosen_row.variant, chosen_row.model
    chosen = fitted[(chosen_variant, chosen_model)]
    print(f"\nChốt theo 1-SE: {chosen_model} / {chosen_variant} "
          f"(CV={chosen_row.cv_macro_f1:.4f}±{chosen_row.cv_std:.4f}, "
          f"ngưỡng 1-SE={chosen_row.threshold_1se:.4f})")

    # ---- 3. Validation của cấu hình đã chốt --------------------------------
    val_pred = chosen.predict(val_texts)
    val_metrics = metrics(y_val, val_pred)
    per_class_table(y_val, val_pred, names, "validation").to_csv(
        TABLES / "per_class_validation.csv", index=False)
    plot_confusion(confusion(chosen, val_texts, y_val), labels,
                   f"{chosen_model} ({chosen_variant}) — validation",
                   FIGURES / f"confusion_val_{chosen_model}_{chosen_variant}.png")

    pred_val, conf_val, how_val = confidence(chosen, val_texts)
    wrong = validation.assign(pred_label=pred_val, confidence=np.round(conf_val, 3))
    wrong = wrong.loc[wrong.label_id != wrong.pred_label].copy()
    wrong["pred_specialty"] = wrong.pred_label.map(names)
    wrong["preview"] = wrong.transcription.astype(str).str.slice(0, 200)
    wrong[["source_row", "medical_specialty", "pred_specialty", "confidence",
           "raw_word_count", "preview"]].to_csv(
        TABLES / "error_samples_validation.csv", index=False)
    print(f"Validation: macro-F1={val_metrics['macro_f1']:.4f}, "
          f"accuracy={val_metrics['accuracy']:.4f}, sai {len(wrong)}/{len(validation)}")

    # ---- 4. Chi phí tính toán ----------------------------------------------
    announce("Đo thời gian fit / độ trễ / kích thước model")
    costs = []
    for model_name in ("mnb_tuned", "svm_l2", "lr_l2", "lr_l1"):
        variant_model, variant_name = fitted[(chosen_variant, model_name)], chosen_variant
        best_params = json.loads(
            comparison.loc[(comparison.model == model_name)
                           & (comparison.variant == chosen_variant), "params"].iloc[0])
        probe = clone(variant_model.named_steps["model"])
        probe.set_params(**best_params)
        measured = cost(probe, chosen_variant, texts, y, val_texts)
        costs.append({"model": model_name, "variant": chosen_variant,
                      "params": json.dumps(best_params, sort_keys=True), **measured})
        print(f"{model_name:13s} fit={measured['fit_seconds']:6.2f}s "
              f"latency={measured['latency_ms_per_doc']:.3f}ms "
              f"size={measured['model_size_kb']:.1f}KB "
              f"features={measured['n_features']}", flush=True)
    cost_table = pd.DataFrame(costs)
    cost_table.to_csv(TABLES / "compute_cost.csv", index=False)

    # ---- 5. Ablation -------------------------------------------------------
    announce("Ablation (đổi một yếu tố so với mốc)")
    ablation = run_ablation(texts, y, folds)
    ablation.to_csv(TABLES / "ablation.csv", index=False)
    print(ablation.to_string(index=False))

    # ---- 6. Chọn lọc đặc trưng bằng L1 -------------------------------------
    announce("LR L1: quét C, số đặc trưng giữ lại và top token")
    sweep_rows, term_tables, l1_summary = [], [], {}
    for variant in VARIANTS:
        tuned = comparison[(comparison.variant == variant) & (comparison.model == "lr_l1")].iloc[0]
        per_c: dict[float, dict] = {}
        for C in L1_GRID:                       # tăng dần: regularization yếu dần
            estimator = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000,
                                           class_weight="balanced", random_state=SEED)
            pipeline = make_pipeline(estimator, variant)
            scores = cross_val_score(pipeline, texts, y, cv=folds, scoring="f1_macro")
            pipeline.fit(texts, y)
            terms, counts = l1_terms(pipeline, len(names))
            per_c[C] = {"cv": float(np.mean(scores)), "std": float(np.std(scores)),
                        "terms": terms, "counts": counts}
            sweep_rows.append({
                "variant": variant, "C": C,
                "n_features": counts["n_features"],
                "selected_any_class": counts["selected_any_class"],
                "selected_pct": round(100 * counts["selected_any_class"] / counts["n_features"], 2),
                "cv_macro_f1": round(float(np.mean(scores)), 4),
                "cv_std": round(float(np.std(scores)), 4),
            })

        best_C = max(per_c, key=lambda c: per_c[c]["cv"])
        threshold = per_c[best_C]["cv"] - per_c[best_C]["std"]
        # Mô hình thưa nhất mà CV vẫn nằm trong 1 SE của LR-L1 tốt nhất: dùng để đọc
        # top token, vì ở C lớn L1 có thể vẫn giữ phần lớn từ vựng.
        sparse_C = min(c for c in L1_GRID if per_c[c]["cv"] >= threshold)
        sparse = per_c[sparse_C]
        sparse_terms = sparse["terms"].assign(C=sparse_C)
        sparse_terms.insert(0, "variant", variant)
        term_tables.append(sparse_terms)

        tuned_C = json.loads(tuned.params).get("C")
        tuned_counts = per_c[tuned_C]["counts"] if tuned_C in per_c else per_c[best_C]["counts"]
        l1_summary[variant] = {
            "n_features": int(tuned_counts["n_features"]),
            "tuned_C": tuned_C,
            "tuned_selected": int(tuned_counts["selected_any_class"]),
            "best_l1_C": best_C,
            "best_l1_cv": round(per_c[best_C]["cv"], 4),
            "sparse_C": sparse_C,
            "sparse_selected": int(sparse["counts"]["selected_any_class"]),
            "sparse_pct": round(100 * sparse["counts"]["selected_any_class"]
                                / sparse["counts"]["n_features"], 2),
            "sparse_cv_macro_f1": round(sparse["cv"], 4),
        }
        print(f"{variant:15s} C tốt nhất={tuned_C} giữ {tuned_counts['selected_any_class']}/5000; "
              f"C thưa nhất trong 1 SE={sparse_C} giữ "
              f"{sparse['counts']['selected_any_class']}/5000 (CV={sparse['cv']:.4f})")

    pd.DataFrame(sweep_rows).to_csv(TABLES / "l1_sweep.csv", index=False)
    if term_tables:
        l1_terms_table = pd.concat(term_tables, ignore_index=True)
        l1_terms_table.to_csv(TABLES / "l1_feature_selection.csv", index=False)
        plot_l1(l1_terms_table[l1_terms_table.variant == chosen_variant], names,
                FIGURES / "21_l1_top_terms.png",
                f"LR L1 ({chosen_variant}) — 5 token có |w| lớn nhất mỗi lớp")
    pd.DataFrame([
        {"variant": variant, **info} for variant, info in l1_summary.items()
    ]).to_csv(TABLES / "l1_summary.csv", index=False)

    # ---- 6b. Kiểm tra biên trên của lưới C ---------------------------------
    announce("Kiểm tra biên lưới C cho SVM và LR L2")
    boundary_rows = []
    for model_name, factory in (
        ("svm_l2", lambda C: LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=5000)),
        ("lr_l2", lambda C: LogisticRegression(C=C, max_iter=3000,
                                               class_weight="balanced", random_state=SEED)),
    ):
        for C in (10.0, 30.0):
            scores = cross_val_score(make_pipeline(factory(C), chosen_variant), texts, y,
                                     cv=folds, scoring="f1_macro")
            boundary_rows.append({"variant": chosen_variant, "model": model_name, "C": C,
                                  "cv_macro_f1": round(float(np.mean(scores)), 4),
                                  "cv_std": round(float(np.std(scores)), 4)})
    boundary = pd.DataFrame(boundary_rows)
    boundary.to_csv(TABLES / "grid_boundary_check.csv", index=False)
    print(boundary.to_string(index=False))


    # ---- 7. Test: dùng đúng một lần ----------------------------------------
    announce("TEST (một lần, cho cấu hình đã chốt)")
    test_pred = chosen.predict(test_texts)
    test_metrics = metrics(y_test, test_pred)
    low, high = bootstrap_macro_f1(y_test, test_pred, n_boot=2000, seed=SEED)
    per_class_table(y_test, test_pred, names, "test").to_csv(
        TABLES / "per_class_test.csv", index=False)
    plot_confusion(confusion(chosen, test_texts, y_test), labels,
                   f"{chosen_model} ({chosen_variant}) — test",
                   FIGURES / f"confusion_test_{chosen_model}_{chosen_variant}.png")

    test_row = {
        "variant": chosen_variant, "model": chosen_model,
        "params": chosen_row.params, "n": test_metrics["n"],
        "macro_f1": test_metrics["macro_f1"], "weighted_f1": test_metrics["weighted_f1"],
        "accuracy": test_metrics["accuracy"], "ci_low": low, "ci_high": high,
        "per_class_f1": json.dumps([round(float(v), 3) for v in test_metrics["per_class"]["f1"]]),
    }
    pd.DataFrame([test_row]).to_csv(TABLES / "final_test_metrics.csv", index=False)
    print(json.dumps(test_row, indent=2, ensure_ascii=False))

    pred_test, conf_test, how_test = confidence(chosen, test_texts)
    wrong_test = test.assign(pred_label=pred_test, confidence=np.round(conf_test, 3))
    wrong_test = wrong_test.loc[wrong_test.label_id != wrong_test.pred_label].copy()
    wrong_test["pred_specialty"] = wrong_test.pred_label.map(names)
    wrong_test["preview"] = wrong_test.transcription.astype(str).str.slice(0, 200)
    wrong_test[["source_row", "medical_specialty", "pred_specialty", "confidence",
                "raw_word_count", "preview"]].to_csv(
        TABLES / "error_samples_test.csv", index=False)

    # ---- 8. Mốc so sánh và tóm tắt -----------------------------------------
    baseline = majority_baseline(y, test_texts, y_test)
    pd.DataFrame([baseline]).to_csv(TABLES / "majority_baseline.csv", index=False)
    print(f"Mốc lớp đông nhất: accuracy={baseline['accuracy']}, "
          f"macro-F1={baseline['macro_f1']}")

    summary = {
        "seed": SEED,
        "folds": len(folds),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "class_support": {"train": np.bincount(y, minlength=len(names)).tolist(),
                          "validation": np.bincount(y_val, minlength=len(names)).tolist(),
                          "test": np.bincount(y_test, minlength=len(names)).tolist()},
        "chosen": {"variant": chosen_variant, "model": chosen_model,
                   "params": json.loads(chosen_row.params),
                   "cv_macro_f1": float(chosen_row.cv_macro_f1),
                   "cv_std": float(chosen_row.cv_std),
                   "threshold_1se": float(chosen_row.threshold_1se)},
        "validation": {k: val_metrics[k] for k in ("macro_f1", "weighted_f1", "accuracy")},
        "validation_errors": int(len(wrong)),
        "test": test_row,
        "test_errors": int(len(wrong_test)),
        "confidence_method": how_test,
        "majority": baseline,
        "l1": l1_summary,
        "l1_sweep": sweep_rows,
        "grid_boundary": boundary_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    (TABLES / "classic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    joblib.dump(chosen, MODELS / f"{chosen_model}_{chosen_variant}.joblib")

    print(f"\nXong sau {summary['elapsed_seconds']}s. Bảng: {TABLES}, hình: {FIGURES}")


if __name__ == "__main__":
    main()
