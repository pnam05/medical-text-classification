"""Học máy cổ điển cho phân loại chuyên khoa bệnh án (phần Mục 3-4 của báo cáo).

Nguyên tắc cố định của module:

- Mọi ước lượng cross-validation dùng ``build_model_pipeline`` trên ``clean_text``,
  tức TF-IDF được **fit lại bên trong từng fold** (yêu cầu ở docs/preprocessing.md),
  không dùng ma trận TF-IDF đã fit trên toàn train.
- Chỉ số chính là macro-F1 (``scoring="f1_macro"``); accuracy và weighted-F1 chỉ là
  chỉ số phụ.
- Fold lấy từ ``data/train_cv_folds.csv`` do pipeline sinh, nên mọi mô hình được
  chấm trên đúng cùng các fold và cùng nhóm văn bản (``text_group``).
- Không đọc, không sửa validation/test trong lúc tune.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import PipelineConfig
from src.data_preprocessing import build_vectorizer

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "preprocessing"
SPLITS = ("train", "validation", "test")
#: Tên biến thể -> ngram_range. Trùng với hai bộ đặc trưng đã bàn giao.
VARIANTS = {"unigram": (1, 1), "unigram_bigram": (1, 2)}
#: Thứ tự phức tạp tăng dần, dùng để phá hoà khi hai cấu hình tương đương.
COMPLEXITY = {"mnb_baseline": 0, "mnb_tuned": 1, "svm_l2": 2, "lr_l2": 3, "lr_l1": 4}
N_CLASSES = 6

# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------


def load_split(split: str, root: Path = ROOT) -> pd.DataFrame:
    """Bảng CSV của một tập, đã kiểm tra cột cần thiết."""
    if split not in SPLITS:
        raise ValueError(f"split phải là {SPLITS}, nhận {split!r}")
    path = root / "data" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Chạy `python -m scripts.run_preprocessing` trước."
        )
    table = pd.read_csv(path)
    missing = {"source_row", "clean_text", "label_id", "medical_specialty"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột {sorted(missing)}")
    return table


def assert_alignment(split: str = "train", root: Path = ROOT) -> None:
    """Xác nhận row i của CSV khớp source_row/label_id đã bàn giao."""
    table = load_split(split, root)
    rows = np.load(root / "features" / f"source_rows_{split}.npy")
    y = np.load(root / "features" / f"y_{split}.npy")
    for name, value in (("source_rows", rows), ("y", y)):
        if len(value) != len(table):
            raise ValueError(f"Độ dài lệch ở {split}: {name}={len(value)}, csv={len(table)}")
    if not np.array_equal(table["source_row"].to_numpy(), rows):
        raise ValueError(f"source_row không khớp ở {split}")
    if not np.array_equal(table["label_id"].to_numpy(), y):
        raise ValueError(f"label_id không khớp y ở {split}")


def label_names(root: Path = ROOT) -> dict[int, str]:
    """{label_id: tên chuyên khoa} từ artifacts/label_mapping.json."""
    mapping = json.loads((root / "artifacts" / "label_mapping.json").read_text(encoding="utf-8"))
    return {int(v): k for k, v in mapping.items()}


def cv_folds(n_splits: int = 5, root: Path = ROOT) -> list[tuple[np.ndarray, np.ndarray]]:
    """Đọc fold chia sẵn thành list[(train_idx, held_out_idx)] theo vị trí dòng 0..707.

    Fold được ghép theo ``source_row`` chứ không theo thứ tự dòng của file fold, nên
    không phụ thuộc việc hai file có được sắp xếp giống nhau hay không.
    """
    path = root / "data" / "train_cv_folds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}. Chạy `python -m scripts.run_preprocessing` trước.")

    table = pd.read_csv(path)
    if "cv_fold" not in table.columns:
        raise ValueError(f"{path} thiếu cột 'cv_fold'; cột hiện có: {list(table.columns)}")

    train = load_split("train", root)
    fold_per_row = train["source_row"].map(table.set_index("source_row")["cv_fold"])
    if fold_per_row.isna().any():
        raise ValueError("Có dòng train không tìm thấy fold trong train_cv_folds.csv")
    folds = fold_per_row.to_numpy(dtype=int)

    found = sorted(int(v) for v in np.unique(folds))
    if found != list(range(n_splits)):
        raise ValueError(f"cv_fold phải gồm đúng 0..{n_splits - 1}, nhận {found}")

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        held_out = np.where(folds == k)[0]
        train_idx = np.where(folds != k)[0]
        if held_out.size == 0:
            raise ValueError(f"Fold {k} rỗng")
        pairs.append((train_idx, held_out))

    covered = np.concatenate([held for _, held in pairs])
    if covered.size != len(folds) or len(np.unique(covered)) != len(folds):
        raise ValueError("Các fold chồng lấn hoặc không phủ hết train rows")
    return pairs


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------


def make_pipeline(estimator, variant: str, **vectorizer_overrides) -> Pipeline:
    """Pipeline TF-IDF -> estimator; mặc định giống bộ đặc trưng đã bàn giao.

    ``vectorizer_overrides`` nhận cả tên trần (``min_df``) và tên có tiền tố
    pipeline (``tfidf__min_df``) để dùng thuận tiện ở phần ablation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant phải là {tuple(VARIANTS)}, nhận {variant!r}")
    clean = {key.removeprefix("tfidf__"): value for key, value in vectorizer_overrides.items()}
    vectorizer = build_vectorizer(PipelineConfig(), VARIANTS[variant])
    if clean:
        vectorizer.set_params(**clean)
    return Pipeline([("tfidf", vectorizer), ("model", estimator)])


def model_zoo(seed: int = 42) -> dict[str, tuple[object, dict]]:
    """name -> (estimator, grid). Grid rỗng = baseline, không tune."""
    return {
        "mnb_baseline": (MultinomialNB(), {}),
        "mnb_tuned": (MultinomialNB(), {"model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}),
        "svm_l2": (
            LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        ),
        "lr_l2": (
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        ),
        "lr_l1": (
            # penalty="l1" là cách viết tương thích scikit-learn 1.5-1.9; từ 1.10 tham
            # số này được thay bằng l1_ratio=1.0 với solver="saga".
            LogisticRegression(
                penalty="l1", solver="saga", max_iter=5000, class_weight="balanced",
                random_state=seed,
            ),
            {"model__C": [0.3, 1.0, 3.0, 10.0, 30.0]},
        ),
    }


def grid_search(estimator, grid: dict, variant: str, texts, y, folds, seed: int = 42):
    """GridSearchCV 5-fold trên train; refit trên toàn train bằng macro-F1.

    ``grid`` rỗng nghĩa là baseline không tune: vẫn đi qua GridSearchCV để mọi
    mô hình có cùng một đường đo (cùng fold, cùng chỉ số, cùng cách refit).
    """
    scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    search = GridSearchCV(
        make_pipeline(estimator, variant),
        grid or {},
        cv=folds,
        scoring=scoring,
        refit="f1_macro",
        return_train_score=True,
        n_jobs=1,          # sandbox chặn pipe của joblib -> chỉ chạy tuần tự
        error_score="raise",
    )
    search.fit(texts, y)
    return search


def summarize_search(search, variant, model_name, seed: int = 42) -> dict:
    """Một dòng bảng so sánh từ kết quả GridSearchCV (fold tốt nhất)."""
    i = search.best_index_
    res = search.cv_results_
    return {
        "variant": variant,
        "model": model_name,
        "params": {k.replace("model__", ""): (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in search.best_params_.items()},
        "cv_macro_f1": round(float(res["mean_test_f1_macro"][i]), 4),
        "cv_std": round(float(res["std_test_f1_macro"][i]), 4),
        "cv_accuracy": round(float(res["mean_test_accuracy"][i]), 4),
        "train_macro_f1": round(float(res["mean_train_f1_macro"][i]), 4),
    }


def grid_table(search, variant, model_name) -> pd.DataFrame:
    """Bảng dài toàn bộ lưới, dùng để vẽ đường cong regularization."""
    rows = []
    res = search.cv_results_
    for params, mean, std, train_mean in zip(
        res["params"], res["mean_test_f1_macro"], res["std_test_f1_macro"],
        res["mean_train_f1_macro"],
    ):
        clean = {k.replace("model__", ""): v for k, v in params.items()}
        rows.append({
            "variant": variant,
            "model": model_name,
            "param": next(iter(clean), "none"),
            "value": next(iter(clean.values()), None),
            "cv_macro_f1": round(float(mean), 4),
            "cv_std": round(float(std), 4),
            "train_macro_f1": round(float(train_mean), 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Đo lường
# ---------------------------------------------------------------------------


def metrics(y_true, y_pred, n_classes: int = N_CLASSES) -> dict:
    """Macro/weighted-F1, accuracy và precision/recall/F1/support từng lớp."""
    labels = np.arange(n_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                        average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels,
                                           average="weighted", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class": {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        },
    }


def per_class_table(y_true, y_pred, names: dict[int, str], split: str) -> pd.DataFrame:
    """Bảng metric từng lớp cho một tập, kèm support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(names)), zero_division=0
    )
    return pd.DataFrame({
        "split": split,
        "class_id": list(range(len(names))),
        "medical_specialty": [names[i] for i in range(len(names))],
        "support": support,
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "f1": np.round(f1, 3),
    })


def confidence(model, texts) -> tuple[np.ndarray, np.ndarray, str]:
    """(nhãn dự đoán, điểm tin cậy, tên cách tính) cho từng văn bản.

    Logistic Regression có ``predict_proba``. LinearSVC chỉ có margin, nên margin
    được đưa qua softmax để so sánh tương đối; đây **không** phải xác suất hiệu chỉnh.
    """
    pred = model.predict(texts)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        return pred, proba.max(axis=1), "predict_proba"
    margin = model.decision_function(texts)
    exp = np.exp(margin - margin.max(axis=1, keepdims=True))
    return pred, (exp / exp.sum(axis=1, keepdims=True)).max(axis=1), "softmax(margin)"


def bootstrap_macro_f1(y_true, y_pred, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Khoảng tin cậy 95% của macro-F1 bằng bootstrap trên các văn bản của tập đó."""
    rng = np.random.default_rng(seed)
    index = np.arange(len(y_true))
    values = []
    for _ in range(n_boot):
        sample = rng.choice(index, len(index), replace=True)
        values.append(f1_score(y_true[sample], y_pred[sample],
                               labels=np.arange(N_CLASSES), average="macro", zero_division=0))
    low, high = np.percentile(values, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def cost(estimator, variant: str, train_texts, y, eval_texts,
         fit_repeats: int = 3, predict_repeats: int = 20) -> dict:
    """Thời gian fit, độ trễ suy luận và kích thước model của một cấu hình.

    Đo trên CPU của máy chạy, không GPU. ``latency_ms_per_doc`` là trung bình trên
    ``predict_repeats`` lần dự đoán toàn bộ ``eval_texts``.
    """
    template = make_pipeline(estimator, variant)
    fit_seconds = []
    fitted = None
    for _ in range(fit_repeats):
        candidate = clone(template)
        start = time.perf_counter()
        candidate.fit(train_texts, y)
        fit_seconds.append(time.perf_counter() - start)
        fitted = candidate

    start = time.perf_counter()
    for _ in range(predict_repeats):
        fitted.predict(eval_texts)
    latency = (time.perf_counter() - start) / (predict_repeats * len(eval_texts)) * 1000

    buffer = io.BytesIO()
    joblib.dump(fitted, buffer)
    coef = getattr(fitted.named_steps["model"], "coef_", None)
    return {
        "fit_seconds": round(float(np.mean(fit_seconds)), 2),
        "fit_seconds_std": round(float(np.std(fit_seconds)), 2),
        "latency_ms_per_doc": round(float(latency), 3),
        "model_size_kb": round(len(buffer.getvalue()) / 1024, 1),
        "n_features": int(len(fitted.named_steps["tfidf"].get_feature_names_out())),
        "nonzero_coefs": int(np.count_nonzero(coef)) if coef is not None else None,
    }


# ---------------------------------------------------------------------------
# Chọn cấu hình
# ---------------------------------------------------------------------------


def select_by_one_se(comparison: pd.DataFrame) -> pd.DataFrame:
    """Quy tắc 1-SE: trong 1 độ lệch chuẩn của mốc tốt nhất, chọn cấu hình ổn định nhất.

    Mốc là cấu hình có CV macro-F1 cao nhất; ngưỡng là ``cv - cv_std``. Ứng viên
    trong ngưỡng được xếp theo ``cv_std`` tăng dần, rồi CV giảm dần, rồi mức phức
    tạp tăng dần. Cách này tránh việc chọn một cấu hình nhích cao hơn chỉ vì may mắn
    ở một fold.
    """
    best = comparison.loc[comparison.cv_macro_f1.idxmax()]
    threshold = best.cv_macro_f1 - best.cv_std

    table = comparison.copy()
    table["within_1se"] = table.cv_macro_f1 >= threshold
    table["complexity"] = table.model.map(COMPLEXITY)
    ranked = table.loc[table.within_1se].sort_values(
        by=["cv_std", "cv_macro_f1", "complexity"], ascending=[True, False, True]
    )
    chosen_key = (ranked.iloc[0].model, ranked.iloc[0].variant)
    table["chosen"] = [(m, v) == chosen_key for m, v in zip(table.model, table.variant)]
    table["threshold_1se"] = round(float(threshold), 4)
    return table


def l1_terms(pipeline, n_classes: int = N_CLASSES, top_k: int = 8) -> tuple[pd.DataFrame, dict]:
    """Đặc trưng mà LR L1 giữ lại và top token theo |w| cho từng lớp."""
    vectorizer = pipeline.named_steps["tfidf"]
    coef = pipeline.named_steps["model"].coef_
    terms = vectorizer.get_feature_names_out()
    if coef.shape[0] != n_classes:
        raise ValueError(f"coef có {coef.shape[0]} hàng, cần {n_classes}")

    rows = []
    for class_id in range(n_classes):
        order = np.argsort(-np.abs(coef[class_id]))[:top_k]
        for rank, idx in enumerate(order, start=1):
            rows.append({
                "class_id": class_id, "rank": rank, "term": terms[idx],
                "coef": round(float(coef[class_id, idx]), 5),
                "kept_for_class": float(coef[class_id, idx]) != 0.0,
            })
    counts = {
        "n_features": int(len(terms)),
        "selected_any_class": int(np.count_nonzero(np.abs(coef).sum(axis=0))),
        "selected_per_class": [int(np.count_nonzero(coef[c])) for c in range(n_classes)],
    }
    return pd.DataFrame(rows), counts


def majority_baseline(y_train, texts_eval, y_eval) -> dict:
    """Mốc ngẫu nhiên-theo-lớp-đông: accuracy và macro-F1 của DummyClassifier."""
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    pred = dummy.predict(np.zeros((len(texts_eval), 1)))
    return {
        "strategy": "most_frequent",
        "majority_class": int(np.bincount(y_train).argmax()),
        "n": int(len(y_eval)),
        "accuracy": round(float(accuracy_score(y_eval, pred)), 4),
        "macro_f1": round(float(f1_score(y_eval, pred, labels=np.arange(N_CLASSES),
                                         average="macro", zero_division=0)), 4),
    }


def confusion(model, texts, y, n_classes: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn n_classes x n_classes, kể cả khi một lớp không xuất hiện."""
    return confusion_matrix(y, model.predict(texts), labels=np.arange(n_classes))
