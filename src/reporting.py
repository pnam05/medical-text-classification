"""Vietnamese preprocessing report, Markdown and a portable HTML figure gallery."""

import base64
import html
from pathlib import Path

import pandas as pd

from src.eda import FIGURE_CAPTIONS


def markdown_table(frame: pd.DataFrame) -> str:
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = [safe(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(safe(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(output: Path, summary: dict, processed: pd.DataFrame,
                 stages: pd.DataFrame, removed: pd.DataFrame) -> None:
    raw_dup = summary["raw_duplicates_all_labels"]
    final_dup = summary["final_duplicates"]
    config = summary["config"]
    split_counts = pd.crosstab(processed["medical_specialty"], processed["split"]).reindex(
        columns=["train", "validation", "test"]).reset_index()
    selected = ", ".join(summary["selected_specialties"])
    mean_raw = processed["raw_word_count"].mean()
    mean_clean = processed["clean_word_count"].mean()
    policy = (
        "Giữ các nhãn mâu thuẫn và buộc mọi bản sao của cùng văn bản vào cùng một tập. "
        "Không suy đoán nhãn đúng và không tự chuyển bài toán thành đa nhãn."
        if config["conflict_policy"] == "group" else
        "Loại toàn bộ nhóm văn bản sạch có nhiều nhãn, không chọn nhãn tùy ý. "
        "Tập còn lại có thể nhỏ và thay đổi phân bố đáng kể."
    )
    paragraphs = [
        ("1. Giới thiệu bài toán", [
            "Bài toán phân loại đa lớp sử dụng nội dung ghi chép lâm sàng (transcription) để dự đoán "
            "nhãn medical_specialty. Mục tiêu của đồ án là chuẩn bị dữ liệu tái lập cho việc so sánh "
            "Machine Learning cổ điển, Deep Learning và khả năng diễn giải mô hình.",
            "Phần việc preprocessing gồm khảo sát dữ liệu, làm sạch, lựa chọn phạm vi lớp, chia tập, "
            "trích xuất TF-IDF và bàn giao dữ liệu dùng chung. Chỉ transcription được dùng làm đầu vào; "
            "description, keywords và sample_name không được dùng làm đặc trưng.",
        ]),
        ("2. Khảo sát dữ liệu ban đầu", [
            f"CSV thực tế có {summary['raw_rows']:,} bản ghi và {summary['raw_classes']} nhãn. "
            f"Số ô transcription thiếu/rỗng: {summary['eda']['missing_values']['transcription']}. "
            "Thiếu keywords không làm bản ghi bị loại vì đây không phải đầu vào của mô hình.",
            f"Trên các dòng có văn bản và nhãn hợp lệ, chuẩn hóa chữ hoa/thường và khoảng trắng cho thấy "
            f"{raw_dup['duplicate_excess_rows']:,} dòng vượt quá số văn bản duy nhất, "
            f"{raw_dup['conflicting_label_groups']:,} nhóm có nhiều nhãn. "
            "Nếu chia ngẫu nhiên theo dòng, cùng bệnh án có thể xuất hiện ở cả train và test.",
            f"Giữ các lớp có ít nhất {config['min_class_groups']} nhóm văn bản độc lập sau xử lý "
            f"và khử trùng; giới hạn Top K bổ sung: {config['top_k']}. Các lớp được chọn: {selected}. "
            f"Ngưỡng support là lựa chọn thiết kế; mỗi holdout cần ít nhất {config['min_holdout_groups']} nhóm/lớp, "
            "không phải ngưỡng tối ưu đã được chứng minh. Kiểm tra mọi lớp có mặt trong từng tập và "
            "từng fold CV train; nếu không, dừng với lỗi rõ ràng. Một số nhãn biểu thị loại tài liệu "
            "(ví dụ Consult - History and Phy.), không hoàn toàn là chuyên khoa; báo cáo giữ đúng ontology nguồn.",
        ]),
        ("3. Quy trình tiền xử lý", [
            "Loại văn bản/nhãn thiếu hoặc trắng → chuẩn hóa Unicode/HTML và chữ thường → mở rộng phủ định "
            "viết tắt → tokenize → lưu toàn bộ dữ liệu hợp lệ → kiểm tra nhãn mâu thuẫn → "
            "khử trùng (văn bản gốc chuẩn hóa, nhãn) → xét số nhóm độc lập của từng lớp. "
            "Không loại bản gốc khác nhau chỉ vì cleaner tạo ra cùng một chuỗi.",
            "Bảo toàn no, not, without, negative, denies và các từ phủ định trong NEGATIONS. "
            "Chặn trường hợp WordNet tạo phủ định mới, chẳng hạn nos thành no. "
            f"Profile lần chạy này: {config['cleaning_profile']}. Conservative giữ stopwords và dạng từ gốc, "
            "không dùng WordNet. Legacy là tùy chọn đối chứng: bỏ stopwords và lemmatization danh từ, "
            "có thể mất nghĩa và cần kiểm chứng bằng CV trên train. Giữ các mã b12, t4, spo2, "
            "số thập phân, tỉ số huyết áp, phần trăm và dấu so sánh sát số. "
            + ("Lần chạy này giữ token số." if config['keep_numbers'] else "Lần chạy này loại token thuần số."),
            policy,
            "text_group là SHA-256 của clean_text; raw_group là SHA-256 của văn bản chuẩn hóa bảo thủ. "
            "source_row là chỉ số bản ghi bắt đầu từ 0 trong CSV, không phải số dòng vật lý. "
            "Mọi dòng bị loại có lý do trong tables/removed_rows.csv; các nhãn mâu thuẫn nằm trong tables/conflicting_labels.csv.",
            f"data/cleaned_all.csv giữ {summary['catalog_rows']} dòng có nội dung dùng được của mọi lớp. "
            "data/rare_or_excluded_classes.csv giữ các lớp ngoài benchmark. Các file này KHÔNG phải tập train: "
            "cột benchmark_split giúp phát hiện văn bản chung với holdout. Không gộp chúng vào train. "
            "class_eligibility.csv giải thích quyết định theo lớp; threshold_coverage.csv cho thấy đánh đổi theo ngưỡng.",
        ]),
        ("4. Kết quả sau xử lý", [
            f"Giữ lại {len(processed):,} bản ghi thuộc {processed['medical_specialty'].nunique()} nhãn, "
            f"với {final_dup['unique_text_groups']:,} nhóm văn bản sạch duy nhất. "
            f"Còn {final_dup['conflicting_label_groups']:,} nhóm nhiều nhãn theo chính sách đã chọn.",
            f"Trên cùng các bản ghi giữ lại, độ dài trung bình là {mean_raw:.1f} từ tách theo khoảng trắng "
            f"trước xử lý và {mean_clean:.1f} token sau xử lý. Hai cách đếm token khác nhau nên đây là mô tả tác động của pipeline, "
            "không phải thước đo giữ nguyên ý nghĩa. Phân bố trước–sau toàn lớp và so sánh ghép cặp được vẽ riêng.",
            f"Tỷ lệ lớp lớn nhất/nhỏ nhất sau xử lý là {summary['class_imbalance_ratio']:.2f}. "
            "Không undersample lớp lớn, không SMOTE trên TF-IDF, không thay đổi phân bố validation/test. "
            "train_class_weights.json được tính riêng từ train để tùy chọn dùng trong mô hình hỗ trợ; "
            "không tự áp trọng số cho mọi mô hình. Báo cáo Macro-F1, Weighted-F1 và support từng lớp. "
            "Lớp chỉ có vài mẫu test sẽ có ước lượng F1 không ổn định.",
        ]),
        ("5. Chia tập và biểu diễn đặc trưng", [
            f"StratifiedGroupKFold với 5 fold, shuffle=True, random_state={config['random_state']}; "
            "fold 0 làm test, fold 1 làm validation, 3 fold còn lại làm train. Tỷ lệ xấp xỉ 60/20/20 "
            "và có thể lệch do phải giữ nguyên nhóm. Kiểm tra cả raw_group và text_group đều không giao nhau giữa các tập.",
            "train_cv_folds.csv lưu fold của từng dòng train; train_cv_class_counts.csv kiểm tra support từng lớp. "
            "Thay đổi cleaner có thể thay đổi nhóm và split: không so sánh trực tiếp điểm số giữa các lần chạy "
            "như một ablation nhân quả. Muốn so sánh cleaner phải cố định manifest split và nhóm an toàn cho cả hai.",
            f"Xuất hai bộ TF-IDF: unigram (1,1) và unigram+bigram (1,2), min_df={config['min_df']}, "
            f"max_features={config['max_features']}, sublinear_tf=True và chuẩn hóa L2. "
            "Mỗi vectorizer chỉ fit trên train rồi transform validation/test. Ma trận lưu dạng sparse NPZ; "
            "vectorizer và LabelEncoder lưu bằng joblib; tên đặc trưng và nhãn số được xuất riêng.",
            "Khi GridSearchCV, phải dùng build_model_pipeline trên clean_text và StratifiedGroupKFold với "
            "groups=text_group để fit TF-IDF lại bên trong mỗi fold. Không dùng ma trận TF-IDF đã fit trên toàn train "
            "để báo cáo cross-validation. Sau khi chốt cấu hình, có thể fit lại trên train+validation và đánh giá test đúng một lần.",
            "Đầu vào cho DL nằm trong transcription gốc cùng các split cố định; tokenizer/vocabulary học từ dữ liệu "
            "chỉ được fit trên train. Không bắt buộc dùng clean_text đã bỏ stopwords/số cho Transformer.",
        ]),
        ("6. Hạn chế và nội dung bàn giao", [
            "Các nhãn mâu thuẫn làm bài toán đơn nhãn không thể được giải thích chỉ bằng nội dung văn bản. "
            f"Benchmark có {summary['ambiguous_benchmark_rows']} dòng trong nhóm nhiều nhãn. Đây là benchmark "
            "đơn nhãn còn nhiễu nhãn, không phải gold standard lâm sàng. Không tự chọn nhãn đa số, "
            "không gộp các lớp hiếm thành Other và không tự chuyển thành đa nhãn. "
            "tables/label_review_queue.csv là danh sách cần xem xét lại ontology/annotation. "
            "Cần nhóm thống nhất chính sách nhãn trước khi diễn giải metric. Tùy chọn --conflict-policy drop "
            "hỗ trợ thử nghiệm đối chứng nhưng kết quả không trực tiếp so sánh được với tập giữ xung đột.",
            f"Giới hạn accuracy top-1 thực nghiệm của hàm dự đoán xác định chỉ nhận clean_text trên toàn "
            f"benchmark này là {summary['deterministic_clean_text_accuracy_ceiling']:.2%}: "
            "với mỗi nhóm đầu vào giống hệt, chỉ có thể dự đoán một nhãn, nên lấy số dòng nhãn phổ biến "
            "nhất của từng nhóm cộng lại rồi chia tổng dòng. Đây là chẩn đoán tính nhất quán nhãn, "
            "không phải điểm đã huấn luyện, không phải dự báo khả năng tổng quát, không phải giới hạn "
            "Macro-F1 hay giới hạn cho mô hình dùng đầu vào khác. Cần xem xét lại ontology hoặc thiết kế "
            "bài toán đa nhãn có kiểm chứng; không có phép làm sạch văn bản nào tự sửa được cấu trúc nhãn này.",
            "Chưa dò trùng gần bằng fuzzy matching và chưa có patient ID để bảo đảm cách ly theo bệnh nhân. "
            "Hash ngăn trùng theo quy tắc hiện tại, không chứng minh hết rò rỉ. Việc bỏ số, stopwords và "
            "lemmatization có thể làm mất ngữ cảnh; bảo toàn token phủ định không đảm bảo bảo toàn phạm vi phủ định. "
            "EDA trên toàn bộ tập chỉ mang tính mô tả; tránh dùng test để chọn siêu tham số.",
            "Biểu đồ từ khóa TF-IDF và SVD là phân tích mô tả, không phải giải thích dự đoán hay bằng chứng về "
            "độ chính xác. Phần này chưa huấn luyện mô hình, chưa có metric và chưa triển khai XAI của thành viên 2/3.",
        ]),
    ]
    if config["benchmark_policy"] == "specialty_single_label":
        resolution = summary["specialty_resolution"]
        paragraphs[1][1][-1] = (
            f"Phạm vi thử nghiệm gồm {len(summary['specialty_scope']['included_labels'])} nhãn trong "
            "configs/specialty_scope.json. Đây là quyết định phạm vi thực nghiệm, không phải danh mục chuyên khoa chuẩn. "
            "Nhãn ghép được giữ nguyên; có những chuyên khoa thật ngoài phạm vi lần này. "
            f"Sau giải quyết nhãn, lớp cần ít nhất {config['min_class_groups']} nhóm độc lập; "
            f"kiểm tra thực tế mỗi lớp có ít nhất {config['min_holdout_groups']} nhóm ở từng holdout. "
            "Đây là mức sàn thiết kế, không bảo đảm metric ổn định hay ngưỡng tối ưu. "
            f"Không giới hạn số lớp mặc định. Các lớp đủ điều kiện: {selected}."
        )
        paragraphs[2][1][0] = (
            "Loại bản ghi thiếu văn bản/nhãn → gom raw_group bằng Unicode, chữ thường và khoảng trắng → "
            "lấy hợp các nhãn mục tiêu trong từng nhóm trên TOÀN BỘ phạm vi đã khai báo. "
            "Không có nhãn mục tiêu: ngoài phạm vi. Có nhiều nhãn mục tiêu: tách để rà soát. "
            "Đúng một nhãn mục tiêu: chọn một dòng gốc mang chính nhãn đó (source_row nhỏ nhất). "
            "Sau đó kiểm tra va chạm cleaner, đếm support và chọn lớp. Việc bỏ lớp hiếm không được "
            "biến nhóm nhiều nhãn thành nhóm đơn nhãn. Không gán nhãn theo đa số hoặc thứ tự tên lớp."
        )
        paragraphs[2][1][2] = (
            "Ví dụ Surgery + Urology → một văn bản Urology; Urology + Nephrology → hàng đợi rà soát. "
            "Nếu các bản gốc khác nhau bị cleaner tạo cùng chuỗi nhưng khác nhãn đã giải quyết, "
            "tách toàn bộ va chạm này vào cleaning_label_collisions.csv, không coi là trùng bản gốc. "
            "specialty_group_decisions.csv lưu nhãn nguồn, nhãn mục tiêu, tất cả ID nguồn và quyết định từng nhóm. "
            "label_review_queue.csv chỉ chứa nhóm còn nhiều nhãn MỤC TIÊU; source_label_conflicts.csv giữ chẩn đoán nhãn nguồn."
        )
        paragraphs[3][1].insert(0,
            f"Có {resolution['raw_groups']:,} nhóm văn bản gốc hợp lệ: "
            f"{resolution.get('single_target_label', 0):,} nhóm đúng một nhãn mục tiêu, "
            f"{resolution.get('ambiguous_target_labels', 0):,} nhóm nhiều nhãn mục tiêu, "
            f"{resolution.get('no_target_label', 0):,} nhóm ngoài phạm vi. "
            f"Sau kiểm tra cleaner, lưu {resolution['resolved_usable_documents']:,} văn bản trong resolved_specialties.csv; "
            f"{resolution['resolved_usable_documents'] - len(processed):,} văn bản không vào benchmark "
            "được giữ trong resolved_low_support.csv. Đếm nhóm độc lập khác với đếm dòng CSV; "
            "không diễn giải việc bỏ bản sao là mất tương ứng số bệnh án độc lập."
        )
        paragraphs[5][1][0] = (
            "Benchmark chỉ đánh giá các trường hợp có nhãn mục tiêu duy nhất trong phạm vi đã chọn. "
            "Không còn nhóm cùng clean_text khác nhãn, nhưng điều này không xác nhận nhãn đúng về lâm sàng. "
            "Việc bỏ các trường hợp đa chuyên khoa làm thay đổi phân bố và có thể khiến bài toán dễ hơn. "
            "Kết quả không đại diện cho toàn bộ 40 nhãn, các lớp hiếm hoặc bệnh án đa chuyên khoa. "
            "Không so sánh trực tiếp điểm với benchmark v2 hoặc paper dùng tập/chia tập khác. "
            "Sau khi sửa phạm vi hoặc duyệt nhãn phải tạo phiên bản dữ liệu và split mới trước huấn luyện."
        )
        paragraphs[5][1][1] = (
            f"Chỉ số nhất quán nhãn (deterministic clean-text accuracy ceiling) bằng "
            f"{summary['deterministic_clean_text_accuracy_ceiling']:.2%} sau giải quyết phạm vi. "
            "Đây chỉ là phép kiểm tra xem đầu vào giống nhau có bị gán nhãn khác nhau hay không; "
            "không phải accuracy đạt được, không dự báo hiệu năng và không xác nhận chất lượng nhãn lâm sàng."
        )
    title = "Báo cáo preprocessing — EDA và tiền xử lý dữ liệu bệnh án"
    md = [f"# {title}\n"]
    html_parts = [f"<h1>{html.escape(title)}</h1>"]
    for heading, texts in paragraphs:
        md += [f"## {heading}\n", *[t + "\n" for t in texts]]
        html_parts.append(f"<h2>{html.escape(heading)}</h2>")
        html_parts += [f"<p>{html.escape(t)}</p>" for t in texts]
    for heading, frame in [("Số mẫu theo từng bước", stages), ("Số mẫu theo nhãn và tập", split_counts),
                           ("Điều kiện chọn lớp", pd.DataFrame(summary["class_support"])),
                           ("Lý do loại bản ghi", removed.groupby("reason").size().rename("rows").reset_index())]:
        md += [f"## {heading}\n", markdown_table(frame) + "\n"]
        html_parts += [f"<h2>{html.escape(heading)}</h2>", frame.to_html(index=False, border=0)]
    md.append("## Biểu đồ và diễn giải\n")
    html_parts.append("<h2>Biểu đồ và diễn giải</h2>")
    for name, caption in FIGURE_CAPTIONS.items():
        file = output / "figures" / f"{name}.png"
        if not file.exists():
            continue
        md += [f"### {name}\n", f"{caption}\n", f"![{caption}](figures/{name}.png)\n"]
        encoded = base64.b64encode(file.read_bytes()).decode("ascii")
        html_parts.append(f'<figure><img loading="lazy" src="data:image/png;base64,{encoded}" '
                          f'alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    (output / "report.md").write_text("\n".join(md), encoding="utf-8")
    style = """body{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#1e293b;background:#f8fafc}
    h1,h2{color:#165b70}table{border-collapse:collapse;background:white;width:100%;margin:24px 0}
    td,th{border:1px solid #d9e2e8;padding:8px;text-align:left}figure{margin:32px 0;padding:20px;background:white;border-radius:12px}
    img{max-width:100%;height:auto}figcaption{padding:12px 0;color:#475569}@media print{figure{break-inside:avoid}body{background:white}}"""
    (output / "report.html").write_text(
        f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
        f'<style>{style}</style></head><body>{"".join(html_parts)}</body></html>', encoding="utf-8")
