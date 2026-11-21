# Báo cáo — Mục 3 & 4: Mô hình học máy cổ điển và đánh giá thực nghiệm

> Phần việc theo `task_allocation_plan.md`: **Thành viên 2** — Mục 3 (Mô hình & Phương pháp tiếp cận) và Mục 4 (Thực nghiệm & Đánh giá kết quả).
> Mọi con số dưới đây lấy từ lần chạy `python -m scripts.run_classic_ml` trên bộ dữ liệu bàn giao (`outputs/preprocessing`); đường dẫn file nguồn ghi ở Phụ lục A.

---

## Mục 3. Mô hình & Phương pháp tiếp cận (Classic ML Methodology)

### 3.1. Ký hiệu và phát biểu bài toán

Gọi $K = 6$ là số lớp (chuyên khoa), $n_{\text{train}} = 708$ là số văn bản huấn luyện. Mỗi bệnh án được biểu diễn thành vector thưa $x \in \mathbb{R}^{V}$ với $V = 5000$, trong đó mỗi chiều ứng với một unigram hoặc bigram và giá trị là trọng số TF-IDF (sublinear tf, chuẩn hoá L2). Nhãn $y \in \{0,1,\dots,K-1\}$.

Bài toán là **phân loại đa lớp, đơn nhãn, mất cân bằng lớp**: cần tìm hàm $f: \mathbb{R}^{V} \to \{0,\dots,K-1\}$ sao cho tối đa hoá chất lượng trên dữ liệu chưa thấy, đồng thời **giải thích được** quyết định của mô hình (yêu cầu của đề tài). Ba họ mô hình được triển khai: Multinomial Naive Bayes (baseline), Linear SVM, và Logistic Regression (L2 và L1).

Một điểm cần nói rõ ngay từ đầu vì nó ảnh hưởng tới cách đọc mọi con số ở Mục 4:

- Bộ bàn giao có sẵn ma trận TF-IDF đã fit **trên train** (`features/X_*.npz`) để các thành viên khác dùng trực tiếp.
- Nhưng khi **báo cáo cross-validation**, ma trận đó **không được dùng lại**: mỗi fold tự fit TF-IDF trên phần train của fold rồi mới transform phần held-out (đúng yêu cầu ở `docs/preprocessing.md`). Nếu dùng ma trận đã fit trên toàn train để chấm CV, vocabulary và IDF đã "thấy" mọi fold, điểm CV sẽ lạc quan hơn thực tế.
- Vì vậy trong Mục 4, mỗi lần đánh giá là một `Pipeline(TfidfVectorizer → estimator)` được fit lại trọn vẹn; validation và test chỉ được `transform`.

### 3.2. Multinomial Naive Bayes (baseline)

Theo định lý Bayes và giả định **độc lập có điều kiện** giữa các từ khi đã biết nhãn:

$$
P(y=c \mid x) \;\propto\; P(y=c)\,\prod_{t \in x} P(t \mid c)^{n_t},
\qquad
P(t \mid c) = \frac{N_{t,c} + \alpha}{N_c + \alpha V}
$$

trong đó $N_{t,c}$ là số lần từ $t$ xuất hiện trong các văn bản lớp $c$, $N_c$ là tổng số token của lớp $c$, $n_t$ là số lần $t$ xuất hiện trong văn bản cần phân loại, và $\alpha$ là tham số làm mượt Laplace.

Ba nhận xét dẫn tới thiết kế thực nghiệm:

1. Giả định độc lập là **sai** trong văn bản y khoa (`chest` và `pain` tương quan mạnh), nên NB thường **quá tự tin**; vì vậy NB được dùng làm **baseline**, không phải mô hình cuối.
2. Làm mượt tránh xác suất bằng 0 cho từ chưa từng gặp trong một lớp — điều kiện để mô hình hoạt động trên từ vựng 5.000 chiều.
3. Số hạng $\alpha V$ ở mẫu số cho thấy: khi $V = 5000$, giá trị $\alpha = 1$ cộng thêm một lượng rất lớn so với $N_c$, làm phân bố gần như phẳng. Do đó lưới tham số được trải về phía $\alpha$ nhỏ: $\alpha \in \{0.01, 0.03, 0.1, 0.3, 1.0, 3.0\}$.

Cấu hình baseline là $\alpha = 1.0$ (mặc định của scikit-learn) và **không tune**, để giữ vai trò mốc tham chiếu tái lập được.

### 3.3. Linear SVM (tối đa hoá lề)

Với bài toán lề mềm (soft margin), hàm mục tiêu là

$$
\min_{w,\,b}\;\; \frac{1}{2}\lVert w \rVert^2 \;+\; C \sum_{i=1}^{n} \max\bigl(0,\; 1 - y_i (w \cdot x_i + b)\bigr)
$$

trong đó số hạng $\frac{1}{2}\lVert w \rVert^2$ tương đương **tối đa hoá lề** $2/\lVert w \rVert$, còn $C$ là giá cân bằng giữa lề rộng và mức vi phạm lề. $C$ lớn → ít regularization → cố khớp train; $C$ nhỏ → lề rộng, mô hình bảo thủ.

- `LinearSVC` mặc định dùng **squared hinge loss** và chuẩn L2, tức $\max(0, 1-y_i f(x_i))^2$ chứ không phải hinge thường; đây là lựa chọn của thư viện, không phải của nhóm.
- Với 6 lớp, `LinearSVC` dùng chiến lược **one-vs-rest**: mỗi lớp học một siêu phẳng "lớp này so với phần còn lại", dự đoán lấy siêu phẳng có điểm cao nhất.

Lưới tham số: $C \in \{0.03, 0.1, 0.3, 1.0, 3.0, 10.0\}$, kèm `class_weight="balanced"` để giảm thiên lệch về lớp đông.

### 3.4. Logistic Regression với phạt L2 và L1

Xác suất lớp theo mô hình softmax:

$$
p(y=c \mid x) = \frac{\exp(w_c \cdot x)}{\sum_{c'=0}^{K-1} \exp(w_{c'} \cdot x)}
$$

Hàm mất mát có regularization:

$$
J(W) \;=\; C \sum_{i=1}^{n} -\log p(y_i \mid x_i) \;+\; \lambda \lVert W \rVert_p,
\qquad p \in \{1, 2\}
$$

với $\lVert W \rVert_2^2$ (L2) hoặc $\lVert W \rVert_1$ (L1).

Lưu ý:

1. **Trong scikit-learn, $C$ nhân vào tổng loss**, tương đương $\lambda = 1/C$ nếu viết theo dạng phạt. Vì vậy **$C$ lớn = ít regularization**, ngược với trực giác "hệ số nhỏ thì nhẹ tay".
2. $\lVert W \rVert_1$ **không khả vi tại 0**, nên nghiệm tối ưu thường có nhiều hệ số **đúng bằng 0** → L1 vừa regularization vừa **chọn lọc đặc trưng**. Đây là cơ sở để phân tích "L1 giữ lại bao nhiêu trên 5.000 đặc trưng" ở mục 4.7.
3. Với L1 đa lớp, cần `solver="saga"`; `liblinear` chỉ hỗ trợ one-vs-rest. Đây là ràng buộc kỹ thuật, không phải lựa chọn thẩm mỹ, và nó là lý do L1 chậm hơn hẳn ba mô hình còn lại (mục 4.8).

Lưới tham số: $C \in \{0.03, 0.1, 0.3, 1.0, 3.0, 10.0\}$ cho L2 và $C \in \{0.3, 1.0, 3.0, 10.0, 30.0\}$ cho L1. Lưới L1 được đẩy lên $C = 30$ vì với $n = 708$ văn bản, phạt L1 ở $C \le 3$ quá mạnh: hệ số co về 0 gần hết (mục 4.7).

### 3.5. Tổng hợp và hệ quả thực nghiệm

| Mô hình | Giả định chính | Tham số tune | Diễn giải được | Chi phí |
|---|---|---|---|---|
| Multinomial NB | độc lập có điều kiện giữa các từ | $\alpha$ | theo từng token, trực quan | rất thấp |
| Linear SVM | quan hệ tuyến tính, lề cực đại | $C$ | qua vector $w$ | thấp |
| Logistic Regression L2 | tuyến tính + xác suất hiệu chỉnh | $C$ | $w$ + xác suất | thấp–trung bình |
| Logistic Regression L1 | như trên + thưa | $C$ | $w$ **thưa** → chọn từ | trung bình (`saga` chậm hơn) |

Bảng này là cầu nối sang Mục 4: mỗi tham số ở cột 3 chính là một lưới `GridSearchCV`; cột 4–5 là cơ sở cho phần XAI và so sánh tài nguyên ở Mục 5.

### 3.6. Giao thức thực nghiệm

Bài toán có 3 tập tách rời: train 708, validation 237, test 237 (chia theo `text_group`, đã kiểm tra không giao nhau).

1. **Huấn luyện** chỉ trên train. Mọi phép biến đổi học từ dữ liệu (vocabulary, IDF, hệ số mô hình) chỉ nhìn thấy train của fold đang xét.
2. **Chọn siêu tham số** bằng 5-fold cross-validation **bên trong train**, dùng đúng fold có sẵn của pipeline (`StratifiedGroupKFold`, nhóm theo `text_group`, seed 42, lưu ở `data/train_cv_folds.csv`) để hai văn bản gần trùng không rơi vào hai fold khác nhau và mọi mô hình được chấm trên cùng các fold. Chỉ số dùng để tune là **macro-F1** (`scoring="f1_macro"`).
3. **Chọn cấu hình** bằng **quy tắc 1-SE** trên kết quả CV: lấy cấu hình tốt nhất làm mốc, mọi cấu hình có CV nằm trong $1$ độ lệch chuẩn của mốc được coi là tương đương, rồi chọn cấu hình **ổn định hơn** (CV std nhỏ hơn). Mục tiêu là không chọn một cấu hình nhích cao hơn chỉ vì may mắn ở một fold.
4. **Validation chỉ dùng để kiểm tra**, không dùng để chọn: nó trả lời câu hỏi "ước lượng CV có lạc quan quá không", chứ không tham gia vào việc chốt cấu hình. Nếu dùng validation để chọn rồi vẫn báo cáo điểm validation, con số đó không còn là ước lượng khách quan.
5. **Test chỉ dùng một lần**, cho cấu hình đã chốt, để tránh selection bias.

---

## Mục 4. Thực nghiệm & Đánh giá kết quả

### 4.1. Thiết lập thực nghiệm

| Thành phần | Giá trị |
|---|---|
| Dữ liệu | 1.182 văn bản, 6 lớp; train 708 / validation 237 / test 237 |
| Phân bố lớp (train) | Cardio 205 · Gastro 114 · Neuro 82 · Ob/Gyn 86 · Ortho 142 · Urology 79; tỉ lệ lớp lớn/nhỏ = 2,58 |
| Đặc trưng | TF-IDF mức từ, hai biến thể: `unigram` (1,1) và `unigram_bigram` (1,2) |
| Tham số vectorizer | `min_df=3`, `max_features=5000`, `sublinear_tf=True`, chuẩn L2, tokenize bằng khoảng trắng trên `clean_text` |
| Từ vựng biến thể bigram | 2.462 unigram + 2.538 bigram (bigram chiếm ~51% trong 5.000 chiều) |
| Độ dài văn bản | trung bình 464 token/văn bản (sau làm sạch) |
| CV | 5 fold `StratifiedGroupKFold` theo `text_group`, seed 42; TF-IDF fit lại trong từng fold |
| Tune | `GridSearchCV`, `scoring="f1_macro"`, refit trên toàn train |
| Chọn cấu hình | quy tắc 1-SE (mục 3.6) |
| Metric chính | Macro-F1 |
| Metric phụ | Weighted-F1, accuracy, precision/recall/F1 từng lớp, confusion matrix |
| Đánh giá cuối | test dùng một lần, CI 95% bằng bootstrap 2.000 lần |
| Môi trường | Python 3.12.0, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0 |

### 4.2. Vì sao chọn Macro-F1 thay vì Accuracy

Đây là lựa chọn thiết kế, không phải sở thích, và có thể kiểm chứng bằng một mốc cụ thể:

| Mốc | Accuracy | Macro-F1 |
|---|---:|---:|
| `DummyClassifier(strategy="most_frequent")` trên test (luôn đoán Cardio) | **0,2911** | **0,0752** |
| Linear SVM (mô hình chốt) trên test | 0,9578 | 0,9508 |

Một mô hình **không học gì** đã đạt 29,1% accuracy vì lớp đông nhất chiếm 29% dữ liệu, nhưng macro-F1 chỉ 7,5% — con số thứ hai phơi bày đúng sự thật rằng 5 lớp còn lại bị dự đoán sai gần như hoàn toàn. Nếu báo cáo bằng accuracy, mọi mô hình trong bảng 4.3 đều trông "trên 90%" và khoảng cách giữa chúng bị nén lại.

Ba lý do cụ thể:

1. **Mất cân bằng lớp có thật trong dữ liệu** (Cardio gấp 2,58 lần Urology). Accuracy cộng dồn theo tần suất, nên sai một bệnh án Urology "rẻ" hơn sai một bệnh án Cardio — trong khi về mặt lâm sàng, bỏ sót một chuyên khoa hiếm không hề rẻ hơn.
2. **Macro-F1 cho mỗi chuyên khoa trọng số bằng nhau**, đúng với mục tiêu "định tuyến hồ sơ tới đúng chuyên khoa" thay vì "tối ưu số dự đoán đúng".
3. **Macro-F1 phạt được kiểu sai lệch phổ biến**: dồn dự đoán về lớp đông (đúng như quan sát ở mục 4.5: 6/12 lỗi validation bị đoán thành Cardio). Accuracy không phạt kiểu sai này đủ mạnh.

Đổi lại, macro-F1 **nhạy với lớp ít mẫu**: test chỉ có 27–38 văn bản cho mỗi lớp nhỏ, nên F1 từng lớp dao động mạnh. Vì vậy báo cáo luôn kèm CI bootstrap và bảng per-class (mục 4.9), và vẫn in weighted-F1 cùng accuracy để người đọc quen với tài liệu dùng accuracy có thể đối chiếu. Ba chỉ số này không thay thế nhau: accuracy trả lời "bao nhiêu phần trăm đúng", weighted-F1 trả lời "đúng bao nhiêu tính theo phân bố thực", macro-F1 trả lời "đúng đều giữa các chuyên khoa hay không".

### 4.3. Kết quả cross-validation và validation

Bảng dưới đây là kết quả **trên train (CV 5 fold) và validation**, dùng để chọn mô hình:

| Biến thể | Mô hình | Siêu tham số tốt nhất | CV macro-F1 (mean ± std) | Val macro-F1 | Val accuracy |
|---|---|---|---|---:|---:|
| unigram | MNB baseline | $\alpha=1{,}0$ (cố định) | 0,7212 ± 0,0307 | 0,7544 | 0,7890 |
| unigram | MNB tuned | $\alpha=0{,}1$ | 0,9075 ± 0,0253 | 0,9135 | 0,9198 |
| unigram | Linear SVM | $C=10{,}0$ | **0,9595 ± 0,0122** | 0,9487 | 0,9494 |
| unigram | LR L2 | $C=10{,}0$ | 0,9489 ± 0,0220 | **0,9564** | **0,9578** |
| unigram | LR L1 | $C=30{,}0$ | 0,9339 ± 0,0178 | 0,9175 | 0,9198 |
| unigram+bigram | MNB baseline | $\alpha=1{,}0$ (cố định) | 0,7802 ± 0,0467 | 0,7966 | 0,8101 |
| unigram+bigram | MNB tuned | $\alpha=0{,}01$ | 0,8587 ± 0,0318 | 0,8693 | 0,8776 |
| unigram+bigram | Linear SVM | $C=10{,}0$ | 0,9469 ± 0,0103 | 0,9497 | 0,9536 |
| unigram+bigram | LR L2 | $C=10{,}0$ | 0,9333 ± 0,0311 | 0,9450 | 0,9494 |
| unigram+bigram | LR L1 | $C=30{,}0$ | 0,9323 ± 0,0144 | 0,9312 | 0,9325 |

*(Sinh lại bằng `classic_models_comparison.csv`; mỗi dòng là một `GridSearchCV` riêng.)*

**Đọc bảng — ba quan sát có ý nghĩa, không chỉ là xếp hạng:**

1. **SVM và LR L2 hoà nhau trong sai số.** SVM cao nhất trên CV (0,9595), LR L2 cao nhất trên validation (0,9564) — đảo thứ tự. Chênh lệch CV là 0,0106, tức **0,87 lần độ lệch chuẩn** của SVM; chênh lệch validation là 0,0077. Cả hai đều nhỏ hơn biến động giữa các fold, nên **chưa thể khẳng định mô hình này hơn mô hình kia**; chỉ có thể nói cả hai nằm trong nhóm dẫn đầu. Đây chính là lý do dùng quy tắc 1-SE thay vì "lấy max".
2. **MNB baseline sụp đổ vì $\alpha V$, không vì NB yếu.** Ở $\alpha = 1$, CV chỉ 0,7212 — thấp hơn cả MNB trên bigram. Hình 1 giải thích cơ chế: hạ $\alpha$ về 0,1 kéo CV lên 0,9075 (+0,186), còn tăng lên $\alpha = 3$ thì CV rơi xuống 0,3918. Với $V = 5000$, số hạng $\alpha V$ ở mẫu số làm phân bố gần như phẳng, đúng như dự đoán ở mục 3.2. Kết luận thực nghiệm: **không được dùng $\alpha$ mặc định cho TF-IDF 5.000 chiều**.
3. **Bigram không cải thiện mô hình nào ngoài baseline.** SVM giảm 0,0126; LR L2 giảm 0,0156; MNB tuned giảm 0,0488. Giả thuyết: giới hạn `max_features=5000` khiến bigram **đẩy bớt unigram** (chỉ còn 2.462 unigram so với 5.000), trong khi bigram hiếm bị `min_df=3` loại phần lớn; phần còn lại chủ yếu là cụm từ hành chính (`the patient`, `history of`). Với MNB baseline thì bigram lại +0,059 — hợp lý vì baseline đang ở vùng làm mượt quá mạnh, thêm cụm từ đặc hiệu giúp tín hiệu bớt bị phẳng, nhưng hiệu ứng này biến mất khi $\alpha$ đã được tune đúng.

> **Hình 1.** Đường cong CV macro-F1 theo $\alpha$ (thang log) cho MNB, mỗi panel một biến thể. Trục tung là trung bình ± độ lệch chuẩn trên 5 fold.
>
> ![CV macro-F1 theo alpha](outputs/preprocessing/figures/19_hparam_alpha.png)

> **Hình 2.** CV macro-F1 theo $C$ cho SVM và Logistic Regression (L2, L1). Lưu ý hai điểm: SVM gần như bão hoà từ $C = 1$ trở lên (0,9578 → 0,9595); L1 cần $C$ rất lớn mới thoát khỏi vùng co hệ số về 0.
>
> ![CV macro-F1 theo C](outputs/preprocessing/figures/20_hparam_C.png)

**Một quan sát về overfitting:** ở $C$ tốt nhất, `train_macro_f1` của SVM, LR L2 và LR L1 đều bằng 1,0000 trong khi CV chỉ 0,93–0,96. Mô hình đã **học thuộc train**; khoảng cách này là lý do phải chấm bằng CV và validation chứ không phải bằng điểm train.

### 4.4. Chọn cấu hình và độ ổn định

Quy trình: lấy CV macro-F1 trung bình trên 5 fold làm mốc, tính ngưỡng $1\text{SE} = \text{CV}_{\max} - \text{std}_{\max} = 0{,}9595 - 0{,}0122 = 0{,}9473$. Mọi cấu hình có CV $\ge 0{,}9473$ được coi là tương đương về chất lượng; trong nhóm đó chọn cấu hình có CV std nhỏ nhất.

| Ứng viên | CV macro-F1 | CV std | Trong 1 SE? | Quyết định |
|---|---:|---:|:---:|---|
| SVM / unigram | 0,9595 | 0,0122 | có | **Chốt** — CV cao nhất *và* ổn định nhất trong nhóm tương đương |
| LR L2 / unigram | 0,9489 | 0,0220 | có | Loại — CV thấp hơn 0,0106 nhưng std gần gấp đôi; không đủ bằng chứng để đổi |
| LR L1 / unigram | 0,9339 | 0,0178 | không | Loại — ngoài ngưỡng 1 SE |
| MNB tuned / unigram | 0,9075 | 0,0253 | không | Loại — ngoài ngưỡng |
| MNB baseline / unigram | 0,7212 | 0,0307 | không | Loại — baseline tham chiếu |
| SVM / unigram+bigram | 0,9469 | 0,0103 | không | Loại — sát ngưỡng nhưng vẫn dưới, và bigram không giúp gì thêm |
| các cấu hình còn lại | ≤ 0,9469 | — | không | Loại |

**Cấu hình chốt: Linear SVM, `class_weight="balanced"`, $C = 10$, đặc trưng unigram TF-IDF 5.000 chiều.** Điểm CV 0,9595 ± 0,0122; validation 0,9487.

Lý do chọn, viết rõ để không ai phải suy đoán:

- Trong nhóm tương đương, SVM có **CV std nhỏ nhất** (0,0122 so với 0,0220 của LR L2), tức ít phụ thuộc vào việc rơi vào fold nào.
- LR L2 có validation cao hơn 0,0077 — nhưng chọn theo validation nghĩa là dùng validation để tune, và khi đó con số 0,9564 không còn là ước lượng khách quan nữa. Quy tắc đã định trước (chọn theo CV) được giữ nguyên.
- SVM cũng rẻ hơn khi suy luận so với `saga` (mục 4.8), nên nếu hai mô hình tương đương thì SVM là lựa chọn hợp lý cho ứng dụng.

**Kiểm tra biên trên của lưới $C$** (vì $C = 10$ là giá trị lớn nhất của lưới tune, cần chắc không phải "tốt nhất vì bị chặn"):

| Mô hình | $C = 10$ | $C = 30$ |
|---|---:|---:|
| SVM / unigram | 0,9595 ± 0,0122 | 0,9595 ± 0,0122 |
| LR L2 / unigram | 0,9489 ± 0,0220 | 0,9512 ± 0,0192 |

SVM **bão hoà hoàn toàn** (trùng đến 4 chữ số), LR L2 nhích +0,0023 — vẫn nằm trong 1 độ lệch chuẩn. Kết luận: nới lưới không mang lại lợi ích thực chất; $C = 10$ là điểm bão hoà chứ không phải điểm bị cắt.

### 4.5. Confusion matrix và phân tích lỗi

> **Hình 3.** Confusion matrix của cấu hình chốt (SVM, unigram) trên **validation**, 6×6, nhãn là tên chuyên khoa.
>
> ![Confusion matrix validation](outputs/preprocessing/figures/confusion_val_svm_l2_unigram.png)

Trên validation có **12/237 văn bản sai (accuracy 0,9494; macro-F1 0,9487)**. Đường chéo cho thấy Ob/Gyn đúng tuyệt đối (28/28), Ortho 47/48, Cardio 67/69; hai lớp yếu nhất là **Gastroenterology 34/38** và **Urology 24/27**.

**1. Cặp lớp bị nhầm nhiều nhất: Gastroenterology → Cardiovascular / Pulmonary (4/38 ca Gastro).** Đây không phải nhiễu ngẫu nhiên mà có cơ chế: 4 văn bản này đều là báo cáo **nuốt baryt / nội soi thực quản – dạ dày** có nội dung kiểu *"EXAM: Modified barium swallow… dysphagia with possible aspiration… free air under the diaphragm"*. Chúng chứa từ vựng hô hấp (`aspiration`, `diaphragm`, `chest`) và mô tả hình ảnh ngực – phổi, trong khi **nhãn là chuyên khoa của bản ghi, không phải cơ quan được mô tả**. Mô hình thuần từ vựng không phân biệt được "thủ thuật của khoa Tiêu hoá" với "nội dung nói về ngực".

**2. Thiên lệch về lớp đông vẫn còn dù đã `class_weight="balanced"`: 6/12 lỗi bị đoán thành Cardio**, trong đó 4 ca sai từ Gastro. Cardio là lớp đông nhất (69/237 ở validation) và có cụm từ vựng rộng (tim, mạch, phổi, ngực) nên "hút" các văn bản có từ ngực – bụng.

**3. Lớp có recall thấp nhất là Gastroenterology (0,895)**, kéo F1 xuống 0,919 — thấp nhất trong 6 lớp ở validation. Đây là lý do `class_weight="balanced"` vẫn được giữ: nó không sửa được lỗi do chồng lấn từ vựng, nhưng ngăn mô hình bỏ hẳn các lớp nhỏ.

**4. Văn bản quá ngắn là nhóm lỗi riêng.** 3/12 lỗi có dưới 100 từ, trong đó 2 ca cực ngắn (8 và 16 từ) — ví dụ `source_row` 166 chỉ có *"The patient tolerated the procedure well and was sent to the Recovery Room in stable condition."*, không có bất kỳ từ khoá chuyên khoa nào. Với 464 token trung bình mỗi văn bản, đây là ngoại lệ hiếm; nhưng nó cho thấy mô hình **không có cơ chế "không đủ thông tin để kết luận"**.

**5. Độ tin cậy là tín hiệu hữu ích.** Điểm tin cậy của 12 ca sai nằm trong khoảng 0,19–0,56 (trung bình 0,30) — thấp hơn rõ rệt so với các ca đúng. Nếu triển khai thực tế, một ngưỡng tin cậy thấp có thể chuyển hồ sơ sang người rà soát thay vì tự động gán nhãn.

Trích 5 ca sai tiêu biểu (`error_samples_validation.csv`):

| `source_row` | Nhãn thật | Dự đoán | Tin cậy | Số từ | Trích đoạn |
|---|---|---|---:|---:|---|
| 3597 | Gastroenterology | Cardiovascular / Pulmonary | 0,206 | 86 | "EXAM: Modified barium swallow. SYMPTOM: Dysphagia with possible aspiration. FINDINGS: A cookie deglutition study…" |
| 3595 | Gastroenterology | Cardiovascular / Pulmonary | 0,371 | 1033 | "CHIEF COMPLAINT: Dysphagia and hematemesis while vomiting. This is a 53-year-old… with HIV and hepatitis B…" |
| 166 | Urology | Cardiovascular / Pulmonary | 0,241 | 16 | "The patient tolerated the procedure well and was sent to the Recovery Room in stable condition." |
| 28 | Urology | Neurology | 0,254 | 215 | "…describing a vasectomy 10 years ago and a failed vasectomy reversal…" |
| 2771 | Neurology | Orthopedic | 0,282 | 386 | "…problems with her right hand. It has been going tingling and getting numb periodically…" |

Trên test có **10/237 văn bản sai**, cùng kiểu: 3 ca Gastro → Cardio (ví dụ `source_row` 3464: *"PREOPERATIVE DIAGNOSES: Bowel obstruction… central line fell off"*), 2 ca Ob/Gyn → Urology (sa bàng quang/`vault prolapse` – vùng chậu là vùng từ vựng urology), 2 ca Neuro → Ortho (đau lưng, tổn thương tuyến yên trên phim). Không xuất hiện kiểu lỗi mới, nên phần phân tích lỗi ở validation có thể dùng lại cho test.

### 4.6. Phân tích ablation

Mỗi lần chỉ đổi **một** yếu tố so với mốc (SVM, $C=1{,}0$, unigram, `min_df=3`, 5.000 đặc trưng, `sublinear_tf=True`, `class_weight="balanced"`), cùng 5 fold, cùng TF-IDF fit trong fold:

| Thí nghiệm | Thiết lập | CV macro-F1 | Δ so với mốc |
|---|---|---:|---:|
| Biến thể đặc trưng | unigram (mốc) | 0,9578 | — |
| Biến thể đặc trưng | unigram + bigram | 0,9452 | −0,0126 |
| `min_df` | 3 (mốc) | 0,9578 | — |
| `min_df` | 1 | 0,9596 | +0,0018 |
| `min_df` | 5 | 0,9566 | −0,0012 |
| `max_features` | 5.000 (mốc) | 0,9578 | — |
| `max_features` | 2.000 | 0,9492 | −0,0086 |
| `max_features` | 10.000 | 0,9590 | +0,0012 |
| `sublinear_tf` | True (mốc) | 0,9578 | — |
| `sublinear_tf` | False | 0,9493 | −0,0085 |
| `class_weight` | balanced (mốc) | 0,9578 | — |
| `class_weight` | None | 0,9589 | +0,0011 |
| MNB $\alpha$ | 1,0 (mốc) | 0,7212 | — |
| MNB $\alpha$ | 0,1 (tuned) | 0,9075 | **+0,1863** |

Với $\sigma$ của cấu hình mốc khoảng 0,0106, **mọi thay đổi nhỏ hơn ~0,011 đều nằm trong nhiễu giữa các fold**. Đọc theo ngưỡng đó:

- **Chỉ hai yếu tố tạo khác biệt rõ:** dung lượng từ vựng quá nhỏ (`max_features=2.000`, −0,0086) và bỏ `sublinear_tf` (−0,0085). Cả hai đều ở ngưỡng "vừa đủ lớn để đáng quan tâm, chưa đủ để khẳng định".
- **`min_df` không quan trọng** trong khoảng 1–5 (±0,002): tài liệu y khoa rất dài (464 token), nên từ hiếm xuất hiện một lần vẫn không đủ để làm nhiễu mô hình tuyến tính có regularization.
- **`max_features=10.000` không giúp gì** (+0,0012): từ vựng đã bão hoà ở 5.000 chiều. Đây là lý do giữ 5.000 (rẻ hơn, vector thưa hơn).
- **`class_weight="balanced"` không cải thiện macro-F1** (+0,0011 khi bỏ đi, tức không khác biệt), nhưng vẫn giữ vì nó bảo vệ recall của lớp nhỏ và không gây hại. Kết quả này cũng cảnh báo: `class_weight` **không** phải công cụ sửa lỗi chồng lấn từ vựng ở mục 4.5.
- **Hiệu ứng lớn nhất trong toàn bộ bảng là tune $\alpha$ cho NB** (+0,1863). Đây là minh chứng rằng với bài toán này, **chọn đúng thang regularization quan trọng hơn chọn thuật toán**.

Hai giả thuyết đã nêu trước khi chạy và kết quả: (i) dự đoán bigram sẽ không giúp vì 5.000 chiều đã chật → đúng (−0,0126); (ii) dự đoán `min_df` thấp sẽ làm loãng mô hình → **sai**, `min_df=1` nhích cao hơn một chút, cho thấy từ hiếm trong văn bản dài không phải nguồn nhiễu chính.

### 4.7. Chọn lọc đặc trưng bằng L1

Vì L1 phụ thuộc rất mạnh vào $C$, trước hết là bảng quét $C$ (unigram, cùng 5 fold):

| $C$ | Hệ số khác 0 / 5.000 | Tỉ lệ | CV macro-F1 |
|---:|---:|---:|---:|
| 0,3 | 7 | 0,14% | 0,3556 |
| 1,0 | 64 | 1,28% | 0,8295 |
| 3,0 | 136 | 2,72% | 0,9063 |
| 10,0 | 219 | 4,38% | 0,9264 |
| 30,0 | 283 | 5,66% | 0,9339 |

Ở $C$ tốt nhất theo CV ($C=30$), LR L1 **giữ 283/5.000 đặc trưng (5,66%)** và đạt CV 0,9339 — **thấp hơn SVM 0,0256**. Đây là cái giá của sự thưa: giảm 94% số đặc trưng, mất khoảng 2,6 điểm macro-F1.

Để đọc hệ số ở một cấu hình thưa hơn nhưng vẫn trong 1 SE của L1 tốt nhất ($0{,}9339 - 0{,}0178 = 0{,}9161$), chọn $C = 10$: **219/5.000 đặc trưng (4,38%)**, CV 0,9264.

| Lớp | 5 token có $\lvert w \rvert$ lớn nhất (hệ số) |
|---|---|
| Cardiovascular / Pulmonary | chest (+22,36) · artery (+17,18) · heart (+15,15) · vein (+14,28) · coronary (+13,02) |
| Gastroenterology | colonoscopy (+20,94) · abdominal (+19,74) · abdomen (+16,52) · esophagus (+13,75) · ulcer (+13,65) |
| Neurology | cerebral (+36,42) · reflexes (+16,68) · neuropathy (+16,36) · motor (+15,94) · procedure (−15,46) |
| Obstetrics / Gynecology | uterus (+30,33) · cervix (+22,10) · pregnancy (+21,04) · breast (+18,39) · vaginal (+18,15) |
| Orthopedic | joint (+23,30) · fracture (+17,89) · bone (+17,18) · tendon (+14,38) · tourniquet (+11,79) |
| Urology | hematuria (+23,40) · bladder (+22,94) · prostate (+21,67) · penis (+16,63) · scrotal (+16,51) |

> **Hình 4.** Top 5 token theo $\lvert w \rvert$ của LR L1 ($C=10$, unigram) cho từng lớp.
>
> ![Top token L1](outputs/preprocessing/figures/21_l1_top_terms.png)

Nhận xét **hai chiều**, vì danh sách này vừa đáng tin vừa không đáng tin ở những chỗ khác nhau:

- **Nhóm hợp lý về y khoa, có thể đem ra thảo luận với bác sĩ:** `colonoscopy`, `esophagus`, `ulcer` (Gastro); `uterus`, `cervix`, `pregnancy`, `fetal` (Ob/Gyn); `hematuria`, `prostate`, `bladder`, `scrotal` (Urology); `fracture`, `joint`, `tendon` (Ortho); `cerebral`, `neuropathy`, `reflexes` (Neuro). Đây đúng là các thuật ngữ đặc trưng của chuyên khoa, không phải từ hành chính.
- **Nhóm là nhiễu/thủ thuật, không phải chuyên khoa:** `tourniquet` (dụng cụ mổ chi), `procedure`, `left`. Chúng có hệ số lớn vì xuất hiện dày đặc trong một lớp nhưng **không mang thông tin chuyên khoa**; một bác sĩ đọc sẽ thấy vô nghĩa.
- **Dấu của hệ số quan trọng không kém độ lớn:** `procedure` có hệ số **−15,46** cho Neurology, `left` là **−10,82** cho Gastroenterology. Nghĩa là "vắng mặt của từ này" mới là bằng chứng, chứ không phải sự xuất hiện. Nếu chỉ đọc trị tuyệt đối sẽ hiểu ngược.
- **L1 chọn tuỳ ý khi đặc trưng tương quan.** `chest` và `chest pain` gần như cùng thông tin; L1 sẽ giữ một cái và đặt cái còn lại bằng 0, và **cái nào được giữ thay đổi theo fold/seed**. Vì vậy danh sách 219 từ **không nên** được đọc là "219 từ quan trọng nhất về y khoa" — nó là "219 từ mà L1 *có thể* dùng để tái tạo quyết định", và tính ổn định của danh sách chưa được kiểm định trong báo cáo này.
- **Sự thưa không giảm dung lượng lưu trữ** trong scikit-learn: `coef_` vẫn là ma trận dày $6 \times 5000$, nên file model của LR L1 (417,9 KB) **bằng** LR L2. Muốn hưởng lợi ích bộ nhớ phải chuyển sang biểu diễn thưa hoặc cắt đặc trưng sau khi chọn.

Về mặt kỹ thuật, cần ghi nhận thêm: để L1 thật sự thưa trên bộ dữ liệu nhỏ này ($n = 708$), phải dùng $C$ rất lớn (10–30) — nghĩa là phạt L1 **yếu** so với thang thông thường. Nếu ai đó chạy `LogisticRegression(penalty="l1", C=1.0)` với kỳ vọng "L1 tự chọn đặc trưng", kết quả sẽ là 64/5.000 đặc trưng và CV chỉ 0,8295 — một mô hình gần như vô dụng. Đây là bẫy thực nghiệm đáng ghi lại.

### 4.8. Chi phí tính toán

Đo trên CPU của máy chạy thí nghiệm (không GPU), đơn luồng; thời gian fit là trung bình 3 lần fit lại trên 708 văn bản train, độ trễ là trung bình 20 lượt dự đoán toàn bộ 237 văn bản validation:

| Mô hình | Đặc trưng dùng | Thời gian fit (s) | Độ trễ (ms/văn bản) | Kích thước model |
|---|---|---:|---:|---:|
| MNB ($\alpha=0{,}1$) | 5.000 | 0,16 | 0,249 | 652,2 KB |
| Linear SVM ($C=10$) | 5.000 | 0,27 | 0,248 | 417,8 KB |
| LR L2 ($C=10$) | 5.000 | 0,42 | 0,247 | 417,9 KB |
| LR L1 ($C=30$) | 283/5.000 khác 0 | **22,67** | 0,250 | 417,9 KB |

Ba điểm đáng chú ý:

1. **Fit và suy luận đều rẻ ở mức mili-giây**: cả bốn mô hình dự đoán dưới 0,25 ms/văn bản, tức khoảng 4.000 văn bản/giây trên một lõi. Với quy mô bệnh viện (vài nghìn hồ sơ/ngày) đây là mức không đáng kể.
2. **`saga` là điểm nghẽn duy nhất**: LR L1 fit chậm gấp ~85 lần SVM (22,67 s so với 0,27 s) dù chỉ dùng 283 đặc trưng. Nguyên nhân là solver stochastic cho L1, không phải kích thước dữ liệu. Nếu cần huấn luyện lại thường xuyên, đây là lý do thực tế để không chọn L1.
3. **Độ trễ gần như giống nhau giữa mọi mô hình** (0,247–0,250 ms) vì chi phí chủ yếu nằm ở bước TF-IDF (băm 464 token thành vector 5.000 chiều), không nằm ở phép nhân ma trận. Điều này có hệ quả cho Mục 5: **so sánh ML với DL nên tách chi phí vectorize khỏi chi phí mô hình**, nếu không sẽ thấy "mô hình nào cũng nhanh như nhau".

Lưu ý khi so sánh: đây là số đo trên một máy, chạy đơn luồng trong môi trường sandbox; chỉ nên dùng **tỉ lệ tương đối** giữa các mô hình, không dùng làm benchmark tuyệt đối. Mô hình nhỏ nhất (SVM, 417,8 KB) cũng đủ nhỏ để nhúng vào dịch vụ web mà không cần GPU.

### 4.9. Kết quả trên tập test

Test chỉ được đánh giá **một lần**, cho cấu hình đã chốt ở 4.4:

| Mô hình (đã chốt) | Macro-F1 | Weighted-F1 | Accuracy | CI 95% (macro-F1) |
|---|---:|---:|---:|---:|
| Linear SVM, unigram, $C=10$ | **0,9508** | 0,9574 | 0,9578 | 0,9178 – 0,9781 |

**So sánh ba ước lượng của cùng một cấu hình:** CV 0,9595 → validation 0,9487 → test 0,9508.

- Test **cao hơn validation 0,0021** và **thấp hơn CV 0,0087**. Cả hai chênh lệch đều nhỏ hơn nửa bề rộng CI (0,030), nên **không có bằng chứng của việc chọn cấu hình quá khớp validation**.
- Test thấp hơn CV một chút là điều nên kỳ vọng: các fold trong CV chia sẻ cùng một "bể" train 708 văn bản nên ít khác biệt hơn so với một tập test hoàn toàn mới. Khoảng cách này (0,009) là ước lượng hợp lý cho mức lạc quan của CV trên bộ dữ liệu này.
- CI 95% rộng 0,060 (từ 0,9178 đến 0,9781) vì test chỉ có 237 văn bản. **Mọi so sánh mô hình chênh nhau dưới ~0,03 macro-F1 đều không kết luận được** — đây là con số cần nhớ khi đọc Mục 5.

F1 từng lớp trên test:

| Lớp | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Cardiovascular / Pulmonary | 69 | 0,958 | 1,000 | 0,979 |
| Gastroenterology | 38 | 0,971 | 0,895 | 0,932 |
| Neurology | 27 | 0,962 | 0,926 | 0,943 |
| Obstetrics / Gynecology | 28 | 0,963 | 0,929 | 0,945 |
| Orthopedic | 48 | 0,960 | 1,000 | 0,980 |
| Urology | 27 | 0,926 | 0,926 | 0,926 |

Hai lớp yếu nhất trên test là **Urology (0,926)** và **Gastroenterology (0,932)** — trùng với validation, nên đây là đặc điểm của mô hình chứ không phải dao động của một tập. Cardio và Ortho đạt recall 1,000 (không bỏ sót ca nào) nhưng đó cũng là hai lớp đông nhất; khoảng cách giữa lớp mạnh nhất (0,980) và yếu nhất (0,926) là 0,054, tức mô hình **không đối xử công bằng giữa các chuyên khoa** — đúng điều mà macro-F1 được chọn để phơi bày.

> **Hình 5.** Confusion matrix trên test: cấu trúc lỗi giống validation — 3 ca Gastro → Cardio, 2 ca Ob/Gyn → Urology, 2 ca Neuro → Ortho.
>
> ![Confusion matrix test](outputs/preprocessing/figures/confusion_test_svm_l2_unigram.png)

### 4.10. Thảo luận và hạn chế

1. **Khoảng cách giữa các mô hình nhỏ hơn khoảng tin cậy của test.** SVM (0,9508) và LR L2 (0,9564 trên validation) chênh nhau 0,0077, trong khi CI 95% của test rộng 0,060. Không thể khẳng định mô hình nào tốt hơn; chỉ có thể nói cả hai nằm trong nhóm dẫn đầu và SVM được chọn vì ổn định hơn trên CV. Báo cáo nào kết luận "SVM tốt nhất" từ chênh lệch dưới 0,03 là kết luận quá mức.
2. **MNB là baseline mạnh sau khi tune.** $\alpha=0{,}1$ đưa CV từ 0,7212 lên 0,9075, chỉ kém SVM 0,052. Trên dữ liệu văn bản, việc một mô hình "ngây thơ" tiến sát mô hình tuyến tính là thông tin đáng báo cáo, không phải thất bại: nó nói rằng tín hiệu phân loại chủ yếu nằm ở **sự hiện diện của từ khoá**, không nằm ở tương tác phức tạp giữa các từ. Điều này củng cố dự đoán rằng mô hình DL sẽ không tạo khoảng cách lớn trên bộ dữ liệu này.
3. **Nhãn là chuyên khoa của bản ghi, không phải chẩn đoán.** Mục 4.5 cho thấy lỗi nặng nhất đến từ các báo cáo mà nội dung nói về cơ quan này nhưng nhãn là khoa thực hiện thủ thuật. Đây là **trần hiệu năng thực tế** của bài toán, không phải khuyết điểm của mô hình: kể cả bác sĩ đọc cũng cần biết quy ước gán nhãn mới phân loại đúng các ca đó.
4. **Chưa kiểm tra trùng gần và chưa cách ly theo bệnh nhân.** `text_group` chỉ khử trùng **khít** sau làm sạch; hai bản ghi của cùng một bệnh nhân (hoặc cùng một báo cáo được đánh lại máy) có thể nằm ở hai tập khác nhau. Dữ liệu không có `patient_id` nên không thể kiểm chứng. Nếu điều này xảy ra, điểm số sẽ **lạc quan hơn thực tế**; mức độ thì chưa định lượng được.
5. **`text_group` không phát hiện diễn đạt lại.** Cùng một ca bệnh được ghi lại bằng từ khác sẽ thành hai nhóm độc lập, tiếp tục làm điểm cao hơn thực tế. Đây là hạn chế của split hiện tại, đã ghi trong báo cáo tiền xử lý.
6. **6 lớp từ 26 nhãn mục tiêu.** Các lớp bị loại vì không đủ 100 văn bản độc lập, nên kết luận **không mở rộng** cho toàn bộ 40 chuyên khoa của MTSamples. Đặc biệt, các chuyên khoa hiếm (Dermatology, Endocrinology, Rheumatology…) chưa được thử, và đó chính là nơi mô hình nhiều khả năng thất bại nhất.
7. **$C$ tối ưu nằm ở biên trên của lưới.** Đã kiểm tra bằng cách nới lên $C=30$: SVM bão hoà hoàn toàn, nên đây là điểm bão hoà thật. Nhưng cần lưu ý `train_macro_f1 = 1,0000` ở cấu hình chốt — mô hình đã học thuộc train, và với SVM lề mềm điều đó có nghĩa là **biên quyết định bị ép sát dữ liệu**; nếu phân bố văn bản ở môi trường thật khác đi, độ ổn định này có thể không giữ được.
8. **Một split, một seed.** CV 5 fold cho $\sigma \approx 0{,}012$ ở cấu hình tốt nhất, nhưng chỉ dùng một lần chia 60/20/20 và một seed. Repeated CV (ví dụ 5×5) sẽ cho ước lượng chắc hơn; chưa làm vì giới hạn thời gian.
9. **Đo tài nguyên mang tính tương đối.** Độ trễ 0,25 ms/văn bản đo trên một máy, đơn luồng, không tính thời gian làm sạch văn bản và TF-IDF (bước này tốn hơn cả mô hình). Muốn so sánh công bằng với DL ở Mục 5, cần đo cả pipeline từ văn bản thô, không chỉ phần mô hình.

Những hạn chế trên là cơ sở để đối chiếu với mô hình DL ở Mục 5: nếu một mô hình học biểu diễn tốt hơn thật sự vượt qua SVM, mức vượt phải **lớn hơn 0,03 macro-F1** mới đáng tin trên bộ test 237 văn bản này; và nếu nó chỉ hơn vài phần nghìn, cái giá về tài nguyên và khả năng diễn giải (đã định lượng ở 4.7 và 4.8) là không tương xứng.

---

## Phụ lục A — Bảng số liệu và nguồn

| # | Số liệu | File nguồn | Trạng thái |
|---|---|---|---|
| 1 | CV + validation 10 dòng (2 biến thể × 5 mô hình) | `outputs/preprocessing/tables/classic_models_comparison.csv` | ✅ |
| 2 | Toàn bộ lưới siêu tham số | `outputs/preprocessing/tables/classic_grid_results.csv` | ✅ |
| 3 | Quyết định chọn cấu hình (1-SE) | `outputs/preprocessing/tables/model_selection.csv` | ✅ |
| 4 | Hình đường cong $\alpha$ và $C$ | `outputs/preprocessing/figures/19_hparam_alpha.png`, `20_hparam_C.png` | ✅ |
| 5 | Confusion matrix validation + test | `outputs/preprocessing/figures/confusion_val_svm_l2_unigram.png`, `confusion_test_svm_l2_unigram.png` | ✅ |
| 6 | Mẫu dự đoán sai (validation, test) | `outputs/preprocessing/tables/error_samples_validation.csv`, `error_samples_test.csv` | ✅ |
| 7 | Ablation từng yếu tố | `outputs/preprocessing/tables/ablation.csv` | ✅ |
| 8 | Quét $C$ của L1 + top token | `outputs/preprocessing/tables/l1_sweep.csv`, `l1_summary.csv`, `l1_feature_selection.csv` | ✅ |
| 9 | Thời gian fit, độ trễ, kích thước model | `outputs/preprocessing/tables/compute_cost.csv` | ✅ |
| 10 | Kiểm tra biên lưới $C$ | `outputs/preprocessing/tables/grid_boundary_check.csv` | ✅ |
| 11 | Metric test + CI bootstrap | `outputs/preprocessing/tables/final_test_metrics.csv` | ✅ |
| 12 | Mốc lớp đông nhất | `outputs/preprocessing/tables/majority_baseline.csv` | ✅ |
| 13 | Phiên bản Python/scikit-learn/numpy | `outputs/preprocessing/summary.json` | ✅ |

## Phụ lục B — Lệnh tái lập

```powershell
# 1. Sinh dữ liệu (một lần, cần mtsamples.csv ở thư mục gốc)
.\.venv\Scripts\python.exe -m scripts.run_preprocessing

# 2. Toàn bộ thực nghiệm classic ML cho Mục 3-4
#    (khoảng 25 phút trên CPU đơn luồng; LR L1 với saga chiếm phần lớn thời gian)
.\.venv\Scripts\python.exe -m scripts.run_classic_ml

# 3. Kiểm thử
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Cấu hình dùng trong báo cáo: seed 42, `min_df=3`, `max_features=5000`, profile `conservative`, `keep_numbers=True` — tất cả ghi trong `outputs/preprocessing/summary.json`. Script dùng fold có sẵn ở `data/train_cv_folds.csv`, nên hai lần chạy trên cùng dữ liệu cho cùng kết quả (khác biệt duy nhất là thời gian đo).
