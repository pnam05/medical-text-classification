# Bộ dữ liệu huấn luyện ML và DL

6 lớp; train 708, validation 237, test 237. Dùng holdout cố định, không chạy CV.

## DL hoặc ML tự xây đặc trưng

`data/train.csv`, `data/validation.csv`, `data/test.csv`:
- `transcription`: văn bản gốc cho tokenizer DL.
- `clean_text`: bản chuẩn hóa cho TF-IDF.
- `label_id`: nhãn số 0-5; `medical_specialty`: tên nhãn.
- `source_row`, `raw_group`, `text_group`: truy vết và kiểm tra trùng.

Chưa tạo token IDs/tensor DL vì phụ thuộc tokenizer của mô hình.
Vocabulary tự học chỉ fit trên train; Transformer dùng tokenizer của checkpoint.

## ML dùng TF-IDF đã xuất

`features/X_{split}_unigram.npz` và `X_{split}_unigram_bigram.npz` là hai lựa chọn
đặc trưng. Mỗi bộ có 5.000 cột, cùng thứ tự dòng với CSV tương ứng.
`features/y_{split}.npy` là nhãn; `source_rows_{split}.npy` để kiểm tra thứ tự dòng.
Không cần nối hai bộ đặc trưng.

```python
from scipy.sparse import load_npz
import numpy as np
from pathlib import Path

root = Path('outputs/preprocessing')
X_train = load_npz(root / 'features/X_train_unigram_bigram.npz')
y_train = np.load(root / 'features/y_train.npy')
X_val = load_npz(root / 'features/X_validation_unigram_bigram.npz')
y_val = np.load(root / 'features/y_validation.npy')
```

Fit mô hình trên train, chọn cấu hình bằng validation, đánh giá cuối trên test.
TF-IDF đã fit riêng trên train. Không fit lại vectorizer trên validation/test.

## Artifacts

- `tfidf_*.joblib`: vectorizer cho transform văn bản mới, sau khi làm sạch bằng
  MedicalTextCleaner với cấu hình trong summary.json.
- `label_mapping.json`, `label_encoder.joblib`: chuyển nhãn tên/số.
- `specialty_scope.json`: snapshot 26 nhãn mục tiêu và hash.
- `stopwords.json`: chính sách giữ/bỏ từ, bảo vệ phủ định.
- `summary.json` ở thư mục gốc: cấu hình, hash CSV, phiên bản, số lượng và kiểm tra dữ liệu.

Đã dọn đầu ra EDA, báo cáo, dữ liệu trung gian và manifest CV. Muốn tái tạo bộ đầy
đủ, chạy `python -m scripts.run_preprocessing` (sẽ sinh lại các đầu ra đã dọn).
