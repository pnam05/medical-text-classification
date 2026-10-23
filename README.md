# Phân loại văn bản bệnh án - bộ bàn giao ML/DL

Đầu ra hiện tại nằm trong `outputs/preprocessing/`.
Xem [hướng dẫn sử dụng dữ liệu](docs/preprocessing.md).

- 26 nhãn mục tiêu trong `configs/specialty_scope.json`.
- Sau giải quyết nhãn và ngưỡng support 100: 1.182 văn bản, 6 lớp.
- Train 708 / validation 237 / test 237. Huấn luyện theo holdout cố định, không dùng CV.
- ML: TF-IDF unigram hoặc unigram+bigram, nhãn NPY và vectorizer đã fit train.
- DL: văn bản gốc `transcription` và nhãn `label_id` trong ba CSV; tokenizer tùy mô hình.
- Giữ metadata/hash trong summary.json để tái lập. Không còn nhóm trùng văn bản
  theo quy tắc hiện tại; chưa dò trùng gần hoặc cách ly theo bệnh nhân.

## Cấu trúc giữ lại

| Thư mục | Vai trò |
|---|---|
| outputs/preprocessing/data | Chỉ train.csv, validation.csv, test.csv |
| outputs/preprocessing/features | X sparse, y, source_rows cho ba tập |
| outputs/preprocessing/artifacts | Vectorizer, bộ mã hóa nhãn, chính sách xử lý |
| src, scripts, configs | Mã nguồn và cấu hình để tái tạo |
| tests | Kiểm tra làm sạch, nhãn, nhóm và chia tập |
| notebooks | Mã nguồn EDA; cần tái tạo đầu ra đầy đủ trước khi chạy |

Đã dọn bảng EDA, biểu đồ, báo cáo HTML/Markdown/PDF, catalog và các dữ liệu
trung gian, manifest CV, vocabulary CSV và class weights xuất sẵn. Dữ liệu nguồn
mtsamples.csv, paper, đề cương và mã nguồn được giữ lại.

## Tái tạo khi cần

```powershell
.\.venv\Scripts\python.exe -m scripts.run_preprocessing
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Lệnh pipeline hiện tại tái tạo **bộ đầy đủ**, gồm cả các đầu ra EDA/CV đã dọn.
Nó chỉ chuẩn bị dữ liệu, không huấn luyện bằng CV. Sau khi tái tạo bộ đầy đủ,
notebook EDA và scripts/create_summary_pdf.py có thể dùng lại.
Môi trường cài bằng requirements.txt; các kiểm thử legacy cần WordNet trong .nltk_data.

Chỉ học trên train, chọn mô hình/siêu tham số bằng validation và đánh giá cuối
trên test. Không fit lại TF-IDF hoặc vocabulary tự học trên validation/test.

## Clone và chạy trên máy khác

Repository lưu mã nguồn, cấu hình, tài liệu, notebook không kèm output và các file requirements.
Dataset, môi trường Python, corpus NLTK và các đầu ra sinh tự động không đưa lên GitHub.

1. Chuẩn bị Python 3.11+ và đặt `mtsamples.csv` ở thư mục gốc (Medical Transcriptions
   Dataset từ Kaggle/MTSamples, gồm cột `transcription` và `medical_specialty`).
2. Tạo môi trường và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.run_preprocessing
```

`requirements-lock.txt` ghi phiên bản môi trường đã dùng; `requirements.txt` là
các khoảng phiên bản hỗ trợ cài đặt. Pipeline mặc định không cần WordNet.
Để chạy các test profile legacy, tải một lần:
`python -m nltk.downloader -d .nltk_data wordnet` trong môi trường dự án.
Script PDF dùng thêm `reportlab` và `pymupdf`, font Arial trên Windows.
Các đường dẫn `outputs/preprocessing/` chỉ xuất hiện sau khi chạy pipeline hoặc
nhận bộ dữ liệu bàn giao riêng. Không cần đưa dữ liệu và ma trận lớn vào commit.
