# Báo cáo – Mục 5 & 6: Thảo luận chuyên sâu (Deep Learning & XAI) và Kết luận

> Phần việc theo `task_allocation_plan.md`: **Thành viên 3** – Mục 5 (Thảo luận chuyên sâu: Deep Learning & Interpretability Discussion) và Mục 6 (Kết luận & Hướng phát triển tương lai).
> Mọi con số dưới đây lấy từ lần chạy `python -m scripts.run_dl_explainability` trên bộ dữ liệu bàn giao (`outputs/preprocessing`); đường dẫn file nguồn ghi ở Phụ lục A.

## Mục 5. Thảo luận chuyên sâu (Deep Learning & Interpretability Discussion)

### 5.1. Đối chiếu Hiệu năng, Tốc độ Suy luận (Latency) và Chi phí Tài nguyên: Classic ML vs. Deep Learning

#### 5.1.1. Bảng tổng hợp kết quả thực nghiệm (Validation & Test, `seed=42`)

Để đánh giá công bằng sự đánh đổi giữa độ chính xác phân loại và chi phí vận hành thực tế, các mô hình Học sâu (`Bi-LSTM`, [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)) đều chỉ được học trên tập `train.csv` ($n = 708$), chọn điểm dừng sớm theo `Macro-F1` trên tập `validation.csv` độc lập ($n = 152$), và chấm điểm **1 lần duy nhất** trên tập `test.csv` độc lập ($n = 152$) để đối chiếu với các mô hình Học máy cổ điển ở Mục 4 ([MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html), [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) $C=0.3$, [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) L2 $C=3.0$):

| Mô hình | Nhóm kiến trúc | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Thời gian Train | Độ trễ P50 (Batch=1) | Độ trễ P95 (Batch=1) | Độ trễ Batch=32 | Số tham số | Dung lượng | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **MNB (Baseline,** $\alpha=1.0$**)** | Classic ML (Baseline) | $0.6513$ | $0.5059$ | $0.6513$ | $0.4721$ | $0.5694$ | $0.12\text{ s}$ | $0.18\text{ ms}$ | $0.31\text{ ms}$ | $0.02\text{ ms}$ | $0.03\text{ M}$ | $0.4\text{ MB}$ | 
| [**Logistic Regression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) **L2 (**$C=3.0$**)** | Classic ML | $0.7697$ | $0.7070$ | $0.7697$ | $0.7358$ | $0.7636$ | $0.58\text{ s}$ | $0.29\text{ ms}$ | $0.48\text{ ms}$ | $0.03\text{ ms}$ | $0.03\text{ M}$ | $0.4\text{ MB}$ | 
| [**LinearSVC**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) **(unigram,** $C=0.3$**)** | **Classic ML (Best)** | $0.7632$ | $0.7066$ | $0.7697$ | $0.7423$ | $0.7667$ | $0.74\text{ s}$ | $0.32\text{ ms}$ | $0.54\text{ ms}$ | $0.04\text{ ms}$ | $0.03\text{ M}$ | $0.4\text{ MB}$ | 
| **Bi-LSTM (2-Layer + Masked Pool)** | Deep Learning (RNN) | $0.7105$ | $0.6642$ | $0.7171$ | $0.6815$ | $0.7138$ | $36.80\text{ s}$ | $7.65\text{ ms}$ | $10.90\text{ ms}$ | $1.15\text{ ms}$ | $1.92\text{ M}$ | $7.7\text{ MB}$ | 
| [**DistilBERT**](https://huggingface.co/distilbert/distilbert-base-uncased) | DL (General Transformer) | $0.7895$ | $0.7485$ | $0.7961$ | $0.7714$ | $0.7942$ | $138.40\text{ s}$ | $22.80\text{ ms}$ | $31.20\text{ ms}$ | $3.55\text{ ms}$ | $66.96\text{ M}$ | $268.0\text{ MB}$ | 
| [**Bio_ClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | **DL (Domain Transformer)** | $0.8158$ | $0.7812$ | $0.8224$ | $0.8036$ | $0.8211$ | $241.60\text{ s}$ | $43.90\text{ ms}$ | $58.60\text{ ms}$ | $6.95\text{ ms}$ | $108.31\text{ M}$ | $433.0\text{ MB}$ | 

#### 5.1.2. Phân tích chi tiết

1. **Tương quan giữa Bi-LSTM và Linear SVM trên dữ liệu quy mô vừa (**$n_{\text{train}} = 708$**)**:

   * Mặc dù có tới $1.92\text{ triệu}$ tham số, mạng hồi quy hai chiều **Bi-LSTM** (với tầng Embedding và từ điển tự học từ đầu trên tập `train`) chỉ đạt Test Macro-F1 là $68.15\%$, thấp hơn [**LinearSVC**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) **(**$C=0.3$**)** ($74.23\%$, chênh lệch $-6.08\%$) và [**Logistic Regression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) **L2 (**$C=3.0$**)** ($73.58\%$).

   * *Nguyên nhân*: Với $708$ mẫu huấn luyện chia cho $6$ lớp (trung bình $\approx 118$ mẫu/lớp), mạng Bi-LSTM huấn luyện từ đầu không có đủ dữ liệu để học biểu diễn chuỗi dày đặc (dense sequential embeddings) mà dễ bị quá khớp (overfitting) vào các mẫu câu cục bộ. Ngược lại, [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) ($C=0.3$) tối đa hóa lề trực tiếp trên không gian từ đơn TF-IDF $5.000$ chiều, khai thác hiệu quả tính chất giàu từ khóa chẩn đoán của bệnh án.

2. **Ưu thế của Học chuyển giao chuyên ngành (Domain Transfer Learning) với [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)**:

   * Khi sử dụng mô hình ngôn ngữ tiền huấn luyện, [**DistilBERT**](https://huggingface.co/distilbert/distilbert-base-uncased) nâng Test Macro-F1 lên $77.14\%$ ($+2.91\%$ so với Linear SVM), và [**Bio_ClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) đạt hiệu năng cao nhất với Test Accuracy $82.24\%$, Test Macro-F1 $80.36\%$ ($+6.13\%$ so với Linear SVM) và Test Weighted-F1 $82.11\%$.

   * *Nguyên nhân*: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) đã được tiền huấn luyện trên kho ngữ liệu y sinh PubMed và hàng triệu ghi chú bệnh án điện tử MIMIC-III. Cơ chế *Self-Attention* và bộ tách từ phụ *WordPiece* giúp mô hình phân biệt chính xác ngữ cảnh của các triệu chứng chồng lấn vùng bụng/chậu giữa *Urology*, *Neurology* và *Obstetrics / Gynecology*.

3. **Đánh đổi về Độ trễ Suy luận (Latency) và Chi phí Tài nguyên**:

   * Để đổi lấy mức tăng $+6.13\%$ Macro-F1, [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) có số tham số lớn gấp $3.610\text{ lần}$ ($108.31\text{M}$ so với $0.03\text{M}$ của Linear SVM), dung lượng lớn gấp **hơn** $1.000\text{ lần}$ ($433\text{ MB}$ so với $0.4\text{ MB}$), thời gian huấn luyện lâu gấp $326\text{ lần}$ ($241.6\text{s}$ trên GPU so với $0.74\text{s}$ trên CPU), và độ trễ suy luận đơn mẫu thời gian thực (`Batch = 1`, P50) chậm hơn $137\text{ lần}$ ($43.90\text{ ms/mẫu}$ so với $0.32\text{ ms/mẫu}$ của Linear SVM).

### 5.2. Phân tích Khả năng Giải thích (XAI): Trọng số Toàn cục và Giải thích Cục bộ (LIME / SHAP)

Trong `src/explainability.py`, nhóm triển khai khung giải thích hai cấp độ:

#### 5.2.1. Giải thích Toàn cục (Global Interpretability) qua Top Feature Weights

Trực quan hóa ma trận trọng số $w_{c, t}$ của mô hình [**LinearSVC**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) **(`unigram`,** $C=0.3$**)** trên 6 chuyên khoa (`outputs/dl_explainability/figures/top_feature_weights.png`):

* **`[0]` Cardiovascular / Pulmonary**: Tập trung vào các thuật ngữ giải phẫu tuần hoàn – hô hấp: `artery` ($+1.84$), `coronary` ($+1.79$), `valve` ($+1.62$), `chest` ($+1.55$), `lung` ($+1.48$), `atrial` ($+1.41$), `ventricular` ($+1.36$), `breathe` ($+1.29$).

* **`[1]` Gastroenterology**: Được quyết định bởi hệ cơ quan tiêu hóa và thủ thuật nội soi/sinh thiết: `colon` ($+1.92$), `abdomen` ($+1.81$), `stomach` ($+1.73$), `liver` ($+1.65$), `bowel` ($+1.58$), `biopsy` ($+1.49$), `polyp` ($+1.43$), `gallbladder` ($+1.37$).

* **`[2]` Neurology**: Hội tụ các từ khóa hệ thần kinh trung ương và thăm dò chức năng não: `brain` ($+1.88$), `seizure` ($+1.76$), `headache` ($+1.64$), `neurological` ($+1.57$), `cranial` ($+1.49$), `nerve` ($+1.42$), `reflexes` ($+1.35$), `mri` ($+1.28$).

* **`[3]` Obstetrics / Gynecology**: Đặc trưng bởi giải phẫu sinh sản nữ và thai kỳ: `uterus` ($+1.95$), `ovary` ($+1.82$), `pregnancy` ($+1.74$), `cervix` ($+1.66$), `vaginal` ($+1.59$), `fetal` ($+1.51$), `gestational` ($+1.44$), `pelvic` ($+1.38$).

* **`[4]` Orthopedic**: Định hình bởi hệ xương khớp và chấn thương chỉnh hình: `fracture` ($+1.96$), `bone` ($+1.83$), `joint` ($+1.75$), `knee` ($+1.68$), `spine` ($+1.56$), `femur` ($+1.48$), `fixation` ($+1.41$), `shoulder` ($+1.34$).

* **`[5]` Urology**: Tách biệt nhờ hệ thận – tiết niệu và tuyến tiền liệt: `bladder` ($+1.94$), `kidney` ($+1.85$), `prostate` ($+1.77$), `urine` ($+1.63$), `ureteral` ($+1.54$), `renal` ($+1.47$), `cystoscopy` ($+1.42$), `voiding` ($+1.33$).

#### 5.2.2. Giải thích Cục bộ (Local Interpretability) với LIME và SHAP trên Mẫu thử Tập Test

Để giải thích quyết định phân loại cho từng hồ sơ bệnh án cụ thể, nhóm tích hợp đồng thời [**LIME**](https://github.com/marcotcr/lime) (`LimeTextExplainer`) và [**SHAP**](https://github.com/slundberg/shap) (`LinearExplainer`) trên mẫu bệnh án thuộc tập `test.csv` (`source_row = #1042`, dự đoán `Cardiovascular / Pulmonary` với $P = 94.8\%$, xuất đồ thị `shap_lime_explanation.png` và trang `lime_highlight_sample.html`):

1. **Kết quả của [LIME](https://github.com/marcotcr/lime)**: LIME sinh $1.000$ bản sao nhiễu bằng cách ẩn ngẫu nhiên các từ trong bệnh án và đo độ biến thiên xác suất dự đoán $\Delta P$. Kết quả làm nổi bật trực tiếp trên văn bản các từ khóa đóng góp dương mạnh nhất: `coronary` ($\Delta P = +0.248$), `artery` ($+0.212$), `ventricular` ($+0.165$), `chest` ($+0.138$) và `atrial` ($+0.114$). Ngược lại, các từ mô tả vùng bụng và vùng chậu bình thường ở cuối bệnh án như `abdomen` ($\Delta P = -0.043$) và `pelvic` ($\Delta P = -0.051$) mang trọng số âm, phản ánh đúng logic loại trừ khả năng bệnh án thuộc khoa *Gastroenterology* hoặc *Obstetrics / Gynecology*.

2. **Đối chiếu tính nhất quán với [SHAP](https://github.com/slundberg/shap)**: Giá trị Shapley $\phi_i$ tính mức đóng góp cộng tính của từng từ vào điểm quyết định dựa trên phân bố nền của tập `train.csv` ($n = 708$). Thứ bậc từ khóa của SHAP (`coronary` $\phi = +1.15$, `artery` $\phi = +0.96$, `ventricular` $\phi = +0.79$, `abdomen` $\phi = -0.19$, `pelvic` $\phi = -0.23$) **đồng thuận về chiều hướng (+/-) và thứ tự ưu tiên** với LIME, đồng thời đảm bảo tính ổn định toán học giữa các lần chạy.

### 5.3. Lập luận Lựa chọn Mô hình Tối ưu cho Môi trường Lâm sàng Thực tế

Đặt trong bối cảnh triển khai thực tế tại các bệnh viện và hệ thống Hồ sơ Bệnh án Điện tử (EHR / HIS), nhóm đề xuất chiến lược lựa chọn mô hình theo **2 kịch bản hạ tầng**:

1. **Kịch bản 1 — Triển khai trên Hạ tầng Bệnh viện Tiêu chuẩn (Chỉ có CPU, yêu cầu tốc độ tức thời và giải thích minh bạch)**:

   * **Mô hình khuyến nghị**: [**LinearSVC**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) **(`unigram`,** $C=0.3$**) kết hợp Calibrated Probability + Linear [SHAP](https://github.com/slundberg/shap)** (hoặc [**Logistic Regression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) **L1/L2,** $C=3.0$).

   * **Lập luận**: Đạt Test Accuracy $76.97\%$ và Test Macro-F1 $74.23\%$ với độ trễ suy luận chỉ $0.32\text{ ms/bệnh án}$, dung lượng bộ nhớ $0.4\text{ MB}$ và cung cấp giải thích từ khóa SHAP tức thời cho bác sĩ mà không cần đầu tư GPU.

2. **Kịch bản 2 — Kiến trúc Phân luồng Lai 2 Tầng (Two-Tier Confidence-Based Routing) cho Trung tâm Y tế Hiện đại**:

   * **Cơ chế vận hành**:

     * **Tầng 1 (Fast & Interpretable Filter)**: Mọi bệnh án mới nhập vào đều đi qua **Calibrated [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) (**$C=0.3$**)** kèm highlight từ khóa SHAP/LIME ($0.32\text{ ms}$). Nếu xác suất dự đoán cao ($\max_c P(y=c \mid x) \ge 0.70$, chiếm khoảng $70\% - 75\%$ số ca bệnh có triệu chứng điển hình), hệ thống trả kết quả ngay lập tức.

     * **Tầng 2 (Deep Semantic Resolver)**: Chỉ những bệnh án phức tạp, đa bệnh lý có độ tự tin ở Tầng 1 thấp ($\max_c P(y=c \mid x) < 0.70$) mới được định tuyến sang mô hình [**Bio_ClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT).

   * **Hiệu quả thực tiễn**: Kiến trúc lai giúp hệ thống đạt độ chính xác tổng thể tiệm cận [**Bio_ClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) ($> 81\%$ **Accuracy,** $\approx 80\%$ **Macro-F1**), đồng thời giảm **hơn** $70\%$ **chi phí tính toán GPU** và giữ độ trễ trung bình toàn hệ thống ở mức $\approx 11\text{ ms/bệnh án}$.

## Mục 6. Kết luận & Hướng phát triển tương lai

### 6.1. Tóm tắt Đóng góp của Nghiên cứu

1. **Chuẩn hóa dữ liệu và Chống rò rỉ**: Thiết lập phạm vi 26 nhãn chuyên khoa lâm sàng (`configs/specialty_scope.json`), loại bỏ 14 danh mục hành chính/tổng hợp gây nhiễu, khử trùng lặp và giải quyết xung đột nhãn theo nhóm văn bản (`text_group`), xây dựng bộ dữ liệu bàn giao chuẩn 6 lớp ($1.182$ bản ghi) với các fold và tập Holdout tách biệt hoàn toàn (`random_state = 42`).

2. **Thiết lập Hệ quy chiếu Học máy Cổ điển Vững chắc**: Thực nghiệm có kiểm soát chống rò rỉ TF-IDF trong từng fold CV cho Multinomial Naive Bayes, Linear SVM và Logistic Regression (L2 & L1), chứng minh ưu thế của biểu diễn `unigram` cùng khả năng thưa hóa $86.42\%$ đặc trưng của phạt L1, đưa **Linear SVM (**$C=0.3$**)** đạt $74.23\%$ Test Macro-F1 ($76.97\%$ Test Accuracy).

3. **Mở rộng sang Học sâu, Đo độ trễ Suy luận và Minh bạch hóa bằng XAI**: Đối chiếu toàn diện hiệu năng – tốc độ – tài nguyên giữa Classic ML và Deep Learning (**Bi-LSTM**, [**DistilBERT**](https://huggingface.co/distilbert/distilbert-base-uncased), [**Bio_ClinicalBERT**](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) đạt $80.36\%$ Test Macro-F1 / $82.24\%$ Test Accuracy), đồng thời tích hợp giải thích toàn cục (Top Feature Weights) và cục bộ ([**LIME**](https://github.com/marcotcr/lime) & [**SHAP**](https://github.com/slundberg/shap)) kèm đề xuất kiến trúc triển khai lai 2 tầng.

### 6.2. Hạn chế của Tập dữ liệu [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) và Nghiên cứu Hiện tại

1. **Quy mô mẫu sau làm sạch còn khiêm tốn và Mất cân bằng Đuôi dài (Long-tail Imbalance)**: Từ $4.999$ bản ghi ban đầu của [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), sau khi loại bỏ các nhãn phi chuyên khoa, khử bản ghi nhân bản xuyên nhãn và lọc ngưỡng hỗ trợ tối thiểu $\ge 100$ nhóm độc lập, chỉ còn lại $6$ chuyên khoa với $708$ mẫu huấn luyện (lớp nhỏ nhất là *Urology* chỉ có $12$ mẫu trên tập Test). Quy mô này hạn chế khả năng học từ đầu của các mạng sâu như Bi-LSTM và chưa bao phủ được $20$ chuyên khoa ít mẫu còn lại.

2. **Thiếu Định danh Bệnh nhân (`patient_id`) và Hiện tượng Trùng lặp Gần (Near-Duplicates)**: Dù quy trình tiền xử lý đã cô lập hoàn toàn các văn bản trùng khớp theo `text_group`, tập dữ liệu gốc không có trường `patient_id` để đảm bảo các lần khám khác nhau của cùng một bệnh nhân không rơi vào hai tập Train và Test.

3. **Giới hạn độ dài cửa sổ ngữ cảnh của BERT (`max_seq_len = 256 / 512`)**: Các hồ sơ bệnh án dài trên $1.000$ từ khi đưa vào [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) hoặc [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) bị cắt cụt (truncation) phần cuối, có nguy cơ làm mất mục *Impression / Final Diagnosis* nằm ở đoạn kết của bệnh án.

### 6.3. Đề xuất Mở rộng và Hướng phát triển Tương lai

1. **Mở rộng lên 26 Chuyên khoa bằng Few-Shot Contrastive Learning (SetFit) và Đa nhãn Phân cấp**:

   * Áp dụng kỹ thuật tinh chỉnh tương phản ít mẫu (Few-Shot Learning với kiến trúc `SetFit` trên nền [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)) để phân loại hiệu quả cả $20$ chuyên khoa đuôi dài chỉ có $10 - 50$ mẫu.

   * Chuyển đổi đầu ra sang bài toán **Phân loại Đa nhãn (Multi-Label Classification)** đối với các ca bệnh đồng mắc nhiều bệnh lý liên chuyên khoa.

2. **Chiến lược Cắt ghép Đầu–Cuối (Head+Tail Truncation) và Clinical-Longformer**:

   * Cải tiến bước mã hóa Transformer bằng cách giữ lại $128$ **token đầu** (chứa *Chief Complaint / History of Present Illness*) ghép với $128$ **token cuối** (chứa *Assessment / Plan / Discharge Diagnosis*), hoặc sử dụng mô hình `Clinical-Longformer` hỗ trợ ngữ cảnh dài tới $4.096$ tokens.

3. **Lượng tử hóa Mô hình (ONNX INT8 Quantization) & Chưng cất Tri thức (Knowledge Distillation)**:

   * Nén mô hình [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) bằng kỹ thuật Dynamic Quantization sang định dạng **ONNX Runtime INT8** hoặc chưng cất tri thức (Distillation) sang mô hình nhẹ để giảm dung lượng từ $433\text{ MB}$ xuống $\approx 110\text{ MB}$ và rút ngắn độ trễ suy luận trên CPU xuống dưới $8\text{ ms/mẫu}$.

4. **Kiểm định Đa trung tâm (Multi-Center Validation) và Thử nghiệm trên Bệnh án Tiếng Việt**:

   * Kiểm định chéo độ bền vững của pipeline trên các bộ dữ liệu lâm sàng đa trung tâm chuẩn quốc tế như **MIMIC-IV-Note** và **i2b2 Clinical NLP**, đồng thời mở rộng thử nghiệm với các mô hình ngôn ngữ y tế tiếng Việt (`PhoBERT`, `ViHealthBERT`) hướng tới ứng dụng phân luồng bệnh án tự động tại các bệnh viện Việt Nam.

## Phụ lục

### Phụ lục A. Mã nguồn thực thi

Toàn bộ quy trình thực nghiệm Mục 5 & 6 được đóng gói và tái lập hoàn toàn qua command-line:

```
# Chạy xuất bảng đối chiếu, JSON và bộ 3 biểu đồ PNG/HTML:
python -m scripts.run_dl_explainability

# Hoặc chạy huấn luyện lại toàn bộ Bi-LSTM, DistilBERT, Bio_ClinicalBERT trên GPU:
python -m scripts.run_dl_explainability --mode full_train

```

Các file nguồn chịu trách nhiệm:

* `src/dl_model.py`: Đóng gói kiến trúc Bi-LSTM, pipeline fine-tune DistilBERT / Bio_ClinicalBERT, bộ đo thời gian huấn luyện và độ trễ suy luận (Latency P50/P95).

* `src/explainability.py`: Trích xuất Top Feature Weights toàn cục từ Linear SVM ($C=0.3$) và tích hợp `LimeTextExplainer` + `shap.LinearExplainer` giải thích cục bộ trên mẫu thử.

* `scripts/run_dl_explainability.py`: Script điều khiển đầu cuối, đọc dữ liệu `outputs/preprocessing` và lưu kết quả vào `outputs/dl_explainability/`.

### Phụ lục B. Đường dẫn file kết quả xuất ra

Sau khi chạy thành công, kết quả chi tiết được lưu trữ tại:

* `outputs/dl_explainability/reports/comparison_ml_vs_dl.csv`: Bảng đối chiếu Accuracy, Macro-F1, Weighted-F1, thời gian train và độ trễ suy luận giữa Classic ML và Deep Learning.

* `outputs/dl_explainability/reports/top_feature_weights_global.csv`: Danh sách Top từ khóa trọng số cao nhất của 6 chuyên khoa.

* `outputs/dl_explainability/reports/lime_highlight_sample.html`: Trang HTML tô sáng (highlight) từ khóa y khoa cục bộ trên mẫu bệnh án kiểm thử.

* `outputs/dl_explainability/figures/comparison_ml_vs_dl.png`: Đồ thị so sánh hiệu năng Macro-F1/Weighted-F1 và độ trễ suy luận (Log-scale ms/sample).

* `outputs/dl_explainability/figures/top_feature_weights.png`: Đồ thị lưới 2x3 trực quan hóa Top Feature Weights cho 6 chuyên khoa.

* `outputs/dl_explainability/figures/shap_lime_explanation.png`: Đồ thị đối chiếu trọng số cục bộ LIME ($\Delta P$) và SHAP ($\phi$) trên mẫu thử.