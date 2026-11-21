"""Script thực thi chính cho Học sâu & Giải thích mô hình (Mục 5-6 báo cáo).

Đọc dữ liệu đã tiền xử lý từ ``outputs/preprocessing/``, chạy các thực nghiệm:
1. Đo thời gian huấn luyện và độ trễ suy luận (Latency) cho Bi-LSTM, DistilBERT, Bio_ClinicalBERT
   đối chiếu với các mô hình học máy cổ điển.
2. Trích xuất Top Feature Weights toàn cục từ Linear SVM (C=0.3).
3. Tích hợp LIME và SHAP giải thích cục bộ trên mẫu kiểm thử độc lập.
4. Ghi toàn bộ kết quả CSV, JSON, HTML và biểu đồ PNG vào ``outputs/dl_explainability/``.

Chạy:
    python -m scripts.run_dl_explainability
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.dl_model import (
    FIGURE_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    benchmark_classic_ml_latency,
    export_comparison_table_and_chart,
    get_synchronized_ml_vs_dl_results,
    load_all_splits,
    train_and_evaluate_bilstm,
    train_and_evaluate_transformer,
)
from src.explainability import (
    explain_test_sample_lime_and_shap,
    extract_global_top_features,
    fit_best_tv2_linear_models,
    get_synchronized_tv2_xai_data,
    plot_global_and_local_xai,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy pipeline Deep Learning & Explainability (Mục 5-6 báo cáo)"
    )
    parser.add_argument(
        "--mode",
        choices=["full_train", "fast_report"],
        default="fast_report",
        help="Chọn 'full_train' để huấn luyện lại toàn bộ hoặc 'fast_report' để xuất báo cáo & đồ thị.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" BẮT ĐẦU CHẠY THỰC NGHIỆM DEEP LEARNING & EXPLAINABILITY")
    print("=" * 80)

    if args.mode == "full_train":
        print("\n[Step 1/4] Tải dữ liệu văn bản đã làm sạch...")
        dfs = load_all_splits()
        train_df, val_df, test_df = dfs["train"], dfs["validation"], dfs["test"]
        print(
            f"  -> Train: {len(train_df)} mẫu | Val: {len(val_df)} mẫu | Test: {len(test_df)} mẫu"
        )

        print("\n[Step 2/4] Huấn luyện Bi-LSTM, DistilBERT, Bio_ClinicalBERT & đo Latency...")
        ml_res = benchmark_classic_ml_latency(train_df, val_df, test_df)
        bilstm_res = train_and_evaluate_bilstm(train_df, val_df, test_df)
        distil_res = train_and_evaluate_transformer(
            train_df, val_df, test_df, checkpoint_name="distilbert-base-uncased"
        )
        bioclin_res = train_and_evaluate_transformer(
            train_df, val_df, test_df, checkpoint_name="emilyalsentzer/Bio_ClinicalBERT"
        )
        all_results = ml_res + [bilstm_res, distil_res, bioclin_res]

        print("\n[Step 3/4] Trích xuất Top Feature Weights & Giải thích cục bộ LIME / SHAP...")
        prob_pipe, vec, svm_base = fit_best_tv2_linear_models(train_df)
        df_global = extract_global_top_features(vec, svm_base, top_k=10)
        local_lime, local_shap, sample_meta = explain_test_sample_lime_and_shap(
            prob_pipe, vec, svm_base, train_df, test_df, sample_idx=0
        )
    else:
        print("\n[Step 1/3] Tổng hợp bảng đối chiếu hiệu năng và độ trễ suy luận...")
        all_results = get_synchronized_ml_vs_dl_results()
        print("\n[Step 2/3] Tổng hợp kết quả Global Top Feature Weights & Local LIME/SHAP...")
        df_global, local_lime, local_shap, sample_meta = get_synchronized_tv2_xai_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    comp_csv = REPORT_DIR / "comparison_ml_vs_dl.csv"
    comp_fig = FIGURE_DIR / "comparison_ml_vs_dl.png"
    df_comp = export_comparison_table_and_chart(all_results, comp_csv, comp_fig)

    print("\n" + "-" * 75)
    print(" KẾT QUẢ ĐỐI CHIẾU HIỆU NĂNG & ĐỘ TRỄ SUY LUẬN: CLASSIC ML VS. DEEP LEARNING")
    print("-" * 75)
    print(df_comp.to_string(index=False))

    global_csv = REPORT_DIR / "top_feature_weights_global.csv"
    df_global.to_csv(global_csv, index=False)
    plot_global_and_local_xai(
        df_global,
        local_lime,
        local_shap,
        sample_meta,
        output_global_fig=FIGURE_DIR / "top_feature_weights.png",
        output_local_fig=FIGURE_DIR / "shap_lime_explanation.png",
        output_html_highlight=REPORT_DIR / "lime_highlight_sample.html",
    )

    summary_json = OUTPUT_DIR / "dl_explainability_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ml_vs_dl_comparison": all_results,
                "local_lime_explanation": local_lime,
                "local_shap_explanation": local_shap,
                "sample_metadata": {
                    k: v for k, v in sample_meta.items() if k != "highlighted_html"
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n  -> Đã lưu: {comp_csv}")
    print(f"  -> Đã lưu: {comp_fig}")
    print(f"  -> Đã lưu: {FIGURE_DIR / 'top_feature_weights.png'}")
    print(f"  -> Đã lưu: {FIGURE_DIR / 'shap_lime_explanation.png'}")
    print(f"  -> Đã lưu: {summary_json}")
    print("\n" + "=" * 80)
    print(" HOÀN THÀNH THỰC NGHIỆM DEEP LEARNING & EXPLAINABILITY THÀNH CÔNG!")
    print("=" * 80)


if __name__ == "__main__":
    main()