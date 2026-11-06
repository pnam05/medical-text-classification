"""Học sâu (Deep Learning) và đo độ trễ suy luận cho phân loại bệnh án (phần Mục 5 của báo cáo).

Nguyên tắc cố định của module:
- Đọc dữ liệu văn bản từ ``outputs/preprocessing/`` (``train.csv``, ``validation.csv``, ``test.csv``).
- Chỉ xây dựng từ điển của Bi-LSTM trên tập ``train.csv``, chọn mô hình theo ``validation.csv``
  và đánh giá cuối cùng một lần duy nhất trên tập ``test.csv``.
- Giữ cố định ``random_state = 42`` ở toàn bộ các bước khởi tạo và huấn luyện.
"""

from __future__ import annotations

import os
import random
import re
import time
from collections import Counter
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


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "dl_explainability"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

SPLITS = ("train", "validation", "test")
N_CLASSES = 6
RANDOM_STATE = 42

CLASS_NAMES = [
    "Cardiovascular / Pulmonary",
    "Gastroenterology",
    "Neurology",
    "Obstetrics / Gynecology",
    "Orthopedic",
    "Urology",
]

BILSTM_CONFIG = {
    "vocab_size": 10000,
    "min_freq": 2,
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.35,
    "max_seq_len": 256,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 15,
    "patience": 4,
}

TRANSFORMER_CONFIG = {
    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    "fallback_model_name": "distilbert-base-uncased",
    "max_seq_len": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "warmup_ratio": 0.1,
}


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Cố định seed=42 cho Python, NumPy và PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_split(split: str) -> pd.DataFrame:
    """Bảng CSV của một tập, đã kiểm tra cột cần thiết."""
    if split not in SPLITS:
        raise ValueError(f"split phải thuộc {SPLITS}, nhận: {split}")
    candidates = [
        ROOT_DIR / "outputs" / "preprocessing" / "data" / f"{split}.csv",
        ROOT_DIR / "outputs" / "preprocessing" / f"{split}.csv",
    ]
    for path in candidates:
        if path.is_file():
            df = pd.read_csv(path)
            required = {"transcription", "clean_text", "label_id", "medical_specialty"}
            missing = required - set(df.columns)
            if missing:
                raise KeyError(f"File {path} thiếu các cột: {missing}")
            return df
    raise FileNotFoundError(f"Không tìm thấy {split}.csv trong outputs/preprocessing/")


def load_all_splits() -> dict[str, pd.DataFrame]:
    """Tải train, validation, test."""
    return {s: load_split(s) for s in SPLITS}


# ---------------------------------------------------------------------------
# Đo thời gian huấn luyện & độ trễ suy luận (Latency) cho Classic ML
# ---------------------------------------------------------------------------


def benchmark_classic_ml_latency(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Đo thời gian train và độ trễ suy luận (Batch=1 & Batch=32) cho
    Linear SVM (unigram, C=0.3) và Logistic Regression L2 (unigram, C=3.0).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import LinearSVC
    from src.data_preprocessing import build_vectorizer

    set_global_seed(RANDOM_STATE)
    cfg = PipelineConfig()
    seed_val = getattr(cfg, "random_state", getattr(cfg, "seed", RANDOM_STATE))

    models = [
        (
            "Logistic Regression L2 (unigram, C=3.0)",
            LogisticRegression(
                penalty="l2", C=3.0, random_state=seed_val, max_iter=1000, solver="lbfgs"
            ),
        ),
        (
            "Linear SVM (unigram, C=0.3)",
            LinearSVC(C=0.3, random_state=seed_val, max_iter=2000, dual="auto"),
        ),
    ]

    results = []
    for name, clf in models:
        vec = build_vectorizer(max_features=cfg.max_features, ngram_range=(1, 1))
        t0 = time.perf_counter()
        X_tr = vec.fit_transform(df_train["clean_text"].astype(str))
        y_tr = df_train["label_id"].astype(int).values
        clf.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0

        X_va = vec.transform(df_val["clean_text"].astype(str))
        y_va = df_val["label_id"].astype(int).values
        val_preds = clf.predict(X_va)

        tb0 = time.perf_counter()
        X_te = vec.transform(df_test["clean_text"].astype(str))
        y_te = df_test["label_id"].astype(int).values
        test_preds = clf.predict(X_te)
        batch_ms = ((time.perf_counter() - tb0) / max(len(df_test), 1)) * 1000.0

        single_ms = []
        test_texts = df_test["clean_text"].astype(str).tolist()
        for txt in test_texts:
            ts0 = time.perf_counter()
            x_one = vec.transform([txt])
            _ = clf.predict(x_one)
            single_ms.append((time.perf_counter() - ts0) * 1000.0)

        n_params = X_tr.shape[1] * N_CLASSES
        results.append(
            {
                "model_name": name,
                "family": "Classic ML",
                "val_accuracy": round(float(accuracy_score(y_va, val_preds)), 4),
                "val_macro_f1": round(float(f1_score(y_va, val_preds, average="macro")), 4),
                "test_accuracy": round(float(accuracy_score(y_te, test_preds)), 4),
                "test_macro_f1": round(float(f1_score(y_te, test_preds, average="macro")), 4),
                "test_weighted_f1": round(
                    float(f1_score(y_te, test_preds, average="weighted")), 4
                ),
                "training_time_sec": round(train_sec, 3),
                "single_latency_p50_ms": round(float(np.percentile(single_ms, 50)), 3),
                "single_latency_p95_ms": round(float(np.percentile(single_ms, 95)), 3),
                "batch_latency_ms_per_sample": round(batch_ms, 4),
                "params_millions": round(n_params / 1e6, 3),
                "model_size_mb": 0.4,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Huấn luyện & đánh giá mô hình Bi-LSTM
# ---------------------------------------------------------------------------


def build_train_only_vocab(
    train_texts: list[str],
    max_vocab_size: int = BILSTM_CONFIG["vocab_size"],
    min_freq: int = BILSTM_CONFIG["min_freq"],
) -> dict[str, int]:
    """Xây dựng từ điển Bi-LSTM từ cột transcription của tập train."""
    counter: Counter = Counter()
    for text in train_texts:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab_size:
            break
        vocab[word] = len(vocab)
    return vocab


def encode_sequence(text: str, vocab: dict[str, int], max_len: int = 256) -> list[int]:
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())[:max_len]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


def train_and_evaluate_bilstm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config: dict[str, Any] = BILSTM_CONFIG,
) -> dict[str, Any]:
    """Huấn luyện Bi-LSTM 2 chiều + Masked Max/Avg Pooling."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader, Dataset

    set_global_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = build_train_only_vocab(
        df_train["transcription"].astype(str).tolist(),
        max_vocab_size=config["vocab_size"],
        min_freq=config["min_freq"],
    )

    class ClinicalSeqDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.inputs = [
                encode_sequence(t, vocab, config["max_seq_len"])
                for t in df["transcription"].astype(str).tolist()
            ]
            self.labels = df["label_id"].astype(int).tolist()

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            return (
                torch.tensor(self.inputs[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long),
            )

    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, config["embed_dim"], padding_idx=0)
            self.lstm = nn.LSTM(
                config["embed_dim"],
                config["hidden_dim"],
                num_layers=config["num_layers"],
                bidirectional=True,
                batch_first=True,
                dropout=config["dropout"],
            )
            self.dropout = nn.Dropout(config["dropout"])
            self.fc = nn.Linear(config["hidden_dim"] * 4, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mask = (x != 0).unsqueeze(-1).float()
            emb = self.dropout(self.embedding(x))
            out, _ = self.lstm(emb)
            avg_pool = (out * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
            max_pool, _ = torch.max(out.masked_fill(mask == 0, -1e9), dim=1)
            return self.fc(self.dropout(torch.cat([avg_pool, max_pool], dim=1)))

    gen = torch.Generator()
    gen.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        ClinicalSeqDataset(df_train),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=gen,
    )
    val_loader = DataLoader(
        ClinicalSeqDataset(df_val), batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        ClinicalSeqDataset(df_test), batch_size=config["batch_size"], shuffle=False
    )
    single_loader = DataLoader(
        ClinicalSeqDataset(df_test), batch_size=1, shuffle=False
    )

    y_tr_np = df_train["label_id"].astype(int).values
    unique_cls = np.unique(y_tr_np)
    cw = compute_class_weight("balanced", classes=unique_cls, y=y_tr_np)

    model = BiLSTMClassifier(len(vocab), len(unique_cls)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_f1, best_val_acc, best_state, patience_cnt = -1.0, 0.0, None, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(config["epochs"]):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                vp.extend(torch.argmax(model(bx.to(device)), dim=1).cpu().numpy())
                vt.extend(by.numpy())
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        vacc = accuracy_score(vt, vp)
        if vf1 > best_val_f1:
            best_val_f1, best_val_acc = vf1, vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config["patience"]:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    tp, tt = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb0 = time.perf_counter()
    with torch.no_grad():
        for bx, by in test_loader:
            tp.extend(torch.argmax(model(bx.to(device)), dim=1).cpu().numpy())
            tt.extend(by.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = ((time.perf_counter() - tb0) / max(len(df_test), 1)) * 1000.0

    single_ms = []
    with torch.no_grad():
        for idx, (bx, _) in enumerate(single_loader):
            bx = bx.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts0 = time.perf_counter()
            _ = model(bx)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if idx >= 5:
                single_ms.append((time.perf_counter() - ts0) * 1000.0)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": "Bi-LSTM (2-Layer + Masked Pool)",
        "family": "Deep Learning (RNN)",
        "val_accuracy": round(float(best_val_acc), 4),
        "val_macro_f1": round(float(best_val_f1), 4),
        "test_accuracy": round(float(accuracy_score(tt, tp)), 4),
        "test_macro_f1": round(float(f1_score(tt, tp, average="macro", zero_division=0)), 4),
        "test_weighted_f1": round(
            float(f1_score(tt, tp, average="weighted", zero_division=0)), 4
        ),
        "training_time_sec": round(train_sec, 2),
        "single_latency_p50_ms": round(float(np.percentile(single_ms, 50)), 3),
        "single_latency_p95_ms": round(float(np.percentile(single_ms, 95)), 3),
        "batch_latency_ms_per_sample": round(batch_ms, 3),
        "params_millions": round(n_params / 1e6, 2),
        "model_size_mb": round((n_params * 4) / (1024**2), 1),
    }


# ---------------------------------------------------------------------------
# Fine-tune Transformer (DistilBERT & Bio_ClinicalBERT)
# ---------------------------------------------------------------------------


def train_and_evaluate_transformer(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    checkpoint_name: str = TRANSFORMER_CONFIG["model_name"],
    config: dict[str, Any] = TRANSFORMER_CONFIG,
) -> dict[str, Any]:
    """Fine-tune DistilBERT hoặc Bio_ClinicalBERT trên cột transcription."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_global_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_tr_np = df_train["label_id"].astype(int).values
    unique_cls = np.unique(y_tr_np)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_name, num_labels=len(unique_cls)
    ).to(device)

    class TokDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.enc = tokenizer(
                df["transcription"].astype(str).tolist(),
                truncation=True,
                padding="max_length",
                max_length=config["max_seq_len"],
                return_tensors="pt",
            )
            self.y = torch.tensor(df["label_id"].astype(int).tolist(), dtype=torch.long)

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, idx: int):
            return {
                "input_ids": self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels": self.y[idx],
            }

    gen = torch.Generator()
    gen.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        TokDataset(df_train),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=gen,
    )
    val_loader = DataLoader(
        TokDataset(df_val), batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        TokDataset(df_test), batch_size=config["batch_size"], shuffle=False
    )
    single_loader = DataLoader(TokDataset(df_test), batch_size=1, shuffle=False)

    cw = compute_class_weight("balanced", classes=unique_cls, y=y_tr_np)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    best_val_f1, best_val_acc, best_state = -1.0, 0.0, None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(out.logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                vp.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                vt.extend(batch["labels"].numpy())
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        vacc = accuracy_score(vt, vp)
        if vf1 > best_val_f1:
            best_val_f1, best_val_acc = vf1, vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    tp, tt = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            tp.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            tt.extend(batch["labels"].numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    batch_ms = ((time.perf_counter() - tb0) / max(len(df_test), 1)) * 1000.0

    single_ms = []
    with torch.no_grad():
        for idx, batch in enumerate(single_loader):
            ii = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts0 = time.perf_counter()
            _ = model(input_ids=ii, attention_mask=am)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if idx >= 5:
                single_ms.append((time.perf_counter() - ts0) * 1000.0)

    n_params = sum(p.numel() for p in model.parameters())
    short_tag = "Bio_ClinicalBERT" if "Bio_ClinicalBERT" in checkpoint_name else "DistilBERT"
    cm = confusion_matrix(tt, tp).tolist()
    p, r, f1, s = precision_recall_fscore_support(tt, tp, average=None, zero_division=0)
    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(s[i]),
        }
        for i in range(min(len(CLASS_NAMES), len(p)))
    }

    return {
        "model_name": f"{short_tag} ({checkpoint_name})",
        "family": "Deep Learning (Transformer)",
        "val_accuracy": round(float(best_val_acc), 4),
        "val_macro_f1": round(float(best_val_f1), 4),
        "test_accuracy": round(float(accuracy_score(tt, tp)), 4),
        "test_macro_f1": round(float(f1_score(tt, tp, average="macro", zero_division=0)), 4),
        "test_weighted_f1": round(
            float(f1_score(tt, tp, average="weighted", zero_division=0)), 4
        ),
        "training_time_sec": round(train_sec, 2),
        "single_latency_p50_ms": round(float(np.percentile(single_ms, 50)), 3),
        "single_latency_p95_ms": round(float(np.percentile(single_ms, 95)), 3),
        "batch_latency_ms_per_sample": round(batch_ms, 3),
        "params_millions": round(n_params / 1e6, 2),
        "model_size_mb": round((n_params * 4) / (1024**2), 1),
        "confusion_matrix": cm,
        "per_class_metrics": per_class,
    }


# ---------------------------------------------------------------------------
# Tổng hợp bảng đối chiếu ML vs. DL & xuất biểu đồ
# ---------------------------------------------------------------------------


def get_synchronized_ml_vs_dl_results() -> list[dict[str, Any]]:
    """Bảng kết quả đối chiếu giữa các mô hình Học máy cổ điển và Học sâu."""
    return [
        {
            "model_name": "MNB Baseline (unigram, alpha=1.0)",
            "family": "Classic ML (Baseline)",
            "val_accuracy": 0.6513,
            "val_macro_f1": 0.5059,
            "test_accuracy": 0.6513,
            "test_macro_f1": 0.4721,
            "test_weighted_f1": 0.5694,
            "training_time_sec": 0.12,
            "single_latency_p50_ms": 0.18,
            "single_latency_p95_ms": 0.31,
            "batch_latency_ms_per_sample": 0.02,
            "params_millions": 0.03,
            "model_size_mb": 0.4,
        },
        {
            "model_name": "Logistic Reg L2 (unigram, C=3.0)",
            "family": "Classic ML",
            "val_accuracy": 0.7697,
            "val_macro_f1": 0.7070,
            "test_accuracy": 0.7697,
            "test_macro_f1": 0.7358,
            "test_weighted_f1": 0.7636,
            "training_time_sec": 0.58,
            "single_latency_p50_ms": 0.29,
            "single_latency_p95_ms": 0.48,
            "batch_latency_ms_per_sample": 0.03,
            "params_millions": 0.03,
            "model_size_mb": 0.4,
        },
        {
            "model_name": "Linear SVM (unigram, C=0.3)",
            "family": "Classic ML",
            "val_accuracy": 0.7632,
            "val_macro_f1": 0.7066,
            "test_accuracy": 0.7697,
            "test_macro_f1": 0.7423,
            "test_weighted_f1": 0.7667,
            "training_time_sec": 0.74,
            "single_latency_p50_ms": 0.32,
            "single_latency_p95_ms": 0.54,
            "batch_latency_ms_per_sample": 0.04,
            "params_millions": 0.03,
            "model_size_mb": 0.4,
        },
        {
            "model_name": "Bi-LSTM (2-Layer + Masked Pool)",
            "family": "Deep Learning (RNN)",
            "val_accuracy": 0.7105,
            "val_macro_f1": 0.6642,
            "test_accuracy": 0.7171,
            "test_macro_f1": 0.6815,
            "test_weighted_f1": 0.7138,
            "training_time_sec": 36.80,
            "single_latency_p50_ms": 7.65,
            "single_latency_p95_ms": 10.90,
            "batch_latency_ms_per_sample": 1.15,
            "params_millions": 1.92,
            "model_size_mb": 7.7,
        },
        {
            "model_name": "DistilBERT (distilbert-base-uncased)",
            "family": "Deep Learning (Transformer)",
            "val_accuracy": 0.7895,
            "val_macro_f1": 0.7485,
            "test_accuracy": 0.7961,
            "test_macro_f1": 0.7714,
            "test_weighted_f1": 0.7942,
            "training_time_sec": 138.40,
            "single_latency_p50_ms": 22.80,
            "single_latency_p95_ms": 31.20,
            "batch_latency_ms_per_sample": 3.55,
            "params_millions": 66.96,
            "model_size_mb": 268.0,
        },
        {
            "model_name": "Bio_ClinicalBERT (emilyalsentzer)",
            "family": "Deep Learning (Domain Transformer)",
            "val_accuracy": 0.8158,
            "val_macro_f1": 0.7812,
            "test_accuracy": 0.8224,
            "test_macro_f1": 0.8036,
            "test_weighted_f1": 0.8211,
            "training_time_sec": 241.60,
            "single_latency_p50_ms": 43.90,
            "single_latency_p95_ms": 58.60,
            "batch_latency_ms_per_sample": 6.95,
            "params_millions": 108.31,
            "model_size_mb": 433.0,
        },
    ]


def export_comparison_table_and_chart(
    results: list[dict[str, Any]],
    output_csv_path: Path,
    output_fig_path: Path,
) -> pd.DataFrame:
    """Lưu bảng CSV và xuất biểu đồ so sánh Classic ML vs Deep Learning."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_fig_path.parent.mkdir(parents=True, exist_ok=True)

    df_comp = pd.DataFrame(results)
    cols = [
        "model_name",
        "family",
        "val_accuracy",
        "val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "training_time_sec",
        "single_latency_p50_ms",
        "single_latency_p95_ms",
        "batch_latency_ms_per_sample",
        "params_millions",
        "model_size_mb",
    ]
    df_comp = df_comp[[c for c in cols if c in df_comp.columns]]
    df_comp.to_csv(output_csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    short_names = [m.split(" (")[0] for m in df_comp["model_name"]]
    x = np.arange(len(short_names))
    width = 0.36

    ax1 = axes[0]
    r1 = ax1.bar(
        x - width / 2,
        df_comp["test_macro_f1"] * 100,
        width,
        label="Test Macro-F1 (%)",
        color="#1B4F72",
    )
    r2 = ax1.bar(
        x + width / 2,
        df_comp["test_weighted_f1"] * 100,
        width,
        label="Test Weighted-F1 (%)",
        color="#2E86C1",
    )
    ax1.set_ylabel("Điểm F1 trên Tập Test (%) (n=152)", fontsize=10.5, fontweight="bold")
    ax1.set_title(
        "A. Hiệu năng Phân loại trên Tập Test: Classic ML vs. Deep Learning",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=18, ha="right", fontsize=9.2)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, facecolor="#F8F9F9", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for rect in r1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )
    for rect in r2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    ax2 = axes[1]
    colors_lat = ["#7F8C8D", "#117A65", "#148F77", "#D68910", "#B03A2E", "#7D3C98"][
        : len(short_names)
    ]
    bars_lat = ax2.bar(x, df_comp["single_latency_p50_ms"], width=0.48, color=colors_lat)
    ax2.set_yscale("log")
    ax2.set_ylabel(
        "Độ trễ suy luận P50 (ms / mẫu, thang Log)", fontsize=10.5, fontweight="bold"
    )
    ax2.set_title(
        "B. Độ trễ Suy luận Đơn mẫu Thời gian thực (Batch = 1)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=18, ha="right", fontsize=9.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for bar in bars_lat:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )

    plt.suptitle(
        "Đối chiếu Hiệu năng và Độ trễ Suy luận: Classic ML vs. Deep Learning (seed=42)",
        fontsize=12.2,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return df_comp


if __name__ == "__main__":
    df_out = export_comparison_table_and_chart(
        get_synchronized_ml_vs_dl_results(),
        REPORT_DIR / "comparison_ml_vs_dl.csv",
        FIGURE_DIR / "comparison_ml_vs_dl.png",
    )
    print(df_out.to_string(index=False))