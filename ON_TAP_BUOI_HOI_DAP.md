# Ôn tập buổi hỏi đáp – CLIP / CIFAR

Tài liệu này trả lời 7 câu thầy có thể hỏi lại, nhưng **bám vào số liệu thật trong repo**
(`CLIP/demo_out_simple/results.json`, `CLIP/demo_out_ensemble/results.json`) chứ không trả lời
bằng lý thuyết chung. Cuối tài liệu có phần **sửa lại một số chỗ chưa chính xác trong ghi chú
của buổi học** và **việc cần làm trước buổi sau**.

---

## 0. Số liệu thật của nhóm (dùng chung cho mọi câu trả lời)

Mô hình: **CLIP ViT-B/32**, backend `openai/CLIP`, chạy **zero-shot trên CPU**, toàn bộ tập test
(n = 10.000 ảnh mỗi dataset).

| Dataset | Prompt | Top-1 (%) | Top-5 (%) | Thời gian (s) |
|---|---|---|---|---|
| CIFAR-10 | 1 template | 88,30 | 99,24 | 1101 |
| CIFAR-10 | 18 template (ensemble) | 89,83 | 99,61 | 1036 |
| CIFAR-100 | 1 template | **64,48** | 88,40 | 1088 |
| CIFAR-100 | 18 template (ensemble) | **65,08** | 89,23 | 1097 |

Con số "64–65%" thầy nhắc trong buổi học chính là **Top-1 của CIFAR-100**.

Ngoài ra còn một lần chạy thử trên 128 ảnh đầu tiên (`CLIP/demo_out/`) cho CIFAR-100 = 75,0%.
**Con số 75% này không được dùng để báo cáo** — lý do ở mục 8.

---

## 1. CIFAR-10 và CIFAR-100 khác nhau thế nào, vì sao chọn CIFAR-100?

### Khác biệt về bản chất, không chỉ về số lớp

Hai tập **cùng nguồn ảnh** (tập con của *80 Million Tiny Images*), **cùng kích thước ảnh** 32×32,
**cùng tổng số ảnh** 60.000. Khác biệt thật sự nằm ở ba chỗ:

1. **Độ mịn của nhãn (label granularity).** CIFAR-10 có 10 lớp *loại trừ lẫn nhau rõ ràng*
   (airplane vs. ship vs. frog). CIFAR-100 có 100 lớp mịn, gom thành 20 superclass — nghĩa là
   **thiết kế có chủ đích để các lớp gần nhau về ngữ nghĩa** (crocodile / lizard / snake / turtle
   cùng nằm trong *reptiles*). Đây là điểm quan trọng nhất.
2. **Số ảnh mỗi lớp giảm 10 lần**: 6.000 → 600 (500 train / 100 test).
3. **Mỗi ảnh có 2 nhãn**: fine (1/100) và coarse (1/20).

### Trả lời câu "tại sao không chọn tập nhỏ hơn?"

Câu trả lời đúng phải theo **mục tiêu của từng giai đoạn**, và nhóm mình thực tế đã làm cả hai:

| Giai đoạn | Mục tiêu | Nên dùng gì | Nhóm đã làm |
|---|---|---|---|
| Kiểm tra pipeline có chạy đúng | Debug, chưa cần số liệu | Tập nhỏ / ít ảnh | 128 ảnh (`demo_out/`), ~15 s |
| Báo cáo kết quả | Số liệu có ý nghĩa thống kê | Toàn bộ test set | 10.000 ảnh, ~18 phút/dataset |

Còn lý do chọn **CIFAR-100 thay vì chỉ CIFAR-10** phải là một lý do *khoa học*, không phải
"vì bài báo dùng":

- **CIFAR-10 đã gần bão hòa với CLIP**: 88–90% Top-1, và Top-5 = 99,2–99,6%. Ở mức đó,
  metric gần như không còn phân biệt được model nào tốt hơn model nào → không quan sát được gì.
- **CIFAR-100 tạo ra hiện tượng cần giải thích**: cùng model, cùng pipeline, Top-1 tụt từ 88,3%
  xuống 64,48%. Chênh lệch **~24 điểm** này là *đối tượng nghiên cứu*. Nếu chỉ chạy CIFAR-10
  thì không có gì để phân tích.
- **CIFAR-100 cho phép kiểm tra đúng điểm yếu đã biết của CLIP zero-shot**: phân biệt các lớp
  mịn cùng superclass phụ thuộc vào việc *tên lớp* có tách biệt trong không gian text embedding
  hay không — đúng thứ mà kiến trúc CLIP quyết định.

> **Cách nói khi thầy hỏi:** "Em dùng CIFAR-10 để xác nhận pipeline đúng, và dùng CIFAR-100 vì
> CIFAR-10 đã bão hòa nên không đo được gì. Chênh lệch 24 điểm giữa hai tập trên *cùng một
> pipeline* mới là dữ liệu để em phân tích."

### Lỗi thường gặp

- Nói "CIFAR-100 khó hơn vì nhiều lớp hơn" — **chưa đủ**. Nhiều lớp hơn chỉ hạ baseline ngẫu
  nhiên (10% → 1%). Cái làm nó khó là **các lớp giống nhau về hình ảnh** + **ít ảnh mỗi lớp**.
- Nói "dataset nhỏ nên dễ" — 32×32 là *khó hơn*, không phải dễ hơn: ở độ phân giải đó nhiều
  đặc trưng phân biệt (vảy, họa tiết, chi tiết mặt) bị xóa mất.

---

## 2. Zero-shot là gì?

### Trực giác

Mô hình giải một bài toán phân loại mà **không được xem một ảnh có nhãn nào từ tập dữ liệu đích**.
Bộ phân loại được "dựng ra" từ *mô tả bằng ngôn ngữ* của các lớp, ngay tại lúc suy luận.

### Hình thức

Bài toán phân loại thông thường: học $f_\theta: \mathcal{X} \to \{1,\dots,C\}$ từ tập huấn luyện
$D_{train} = \{(x_i, y_i)\}$ **của chính dataset đó**.

Zero-shot theo nghĩa của CLIP: cho trước encoder ảnh $g$ và encoder text $h$ đã pre-train
(trên dữ liệu **khác**), và tên lớp $\{c_1,\dots,c_C\}$ dưới dạng chuỗi:

$$\hat{y}(x) = \arg\max_{c}\ \cos\big(g(x),\, h(\text{prompt}(c))\big)$$

Không có bước tối ưu nào trên CIFAR. $|D_{train}^{CIFAR}| = 0$.

### Ví dụ đúng với bài của nhóm

Ảnh index 3637 của CIFAR-100, ground truth = `snake`:

| Lớp | Xác suất |
|---|---|
| snake | 0,674 |
| turtle | 0,165 |
| lizard | 0,021 |
| crocodile | 0,017 |
| snail | 0,014 |

Không hề có classifier nào được train trên CIFAR-100. 100 vector text embedding của
`"a photo of a {class}"` **chính là** trọng số của bộ phân loại.

### Hai nhầm lẫn rất dễ bị thầy bắt

1. **"Zero-shot nghĩa là model chưa từng thấy khái niệm đó."** → **Sai.** Từ "snake" và ảnh rắn
   gần như chắc chắn xuất hiện rất nhiều trong 400 triệu cặp ảnh–text của WIT. Zero-shot ở đây
   chỉ có nghĩa là **không dùng ảnh có nhãn của CIFAR**. Chính bài báo CLIP (Section 5, phần
   *Data Overlap Analysis*) phải đo độ trùng lặp giữa dữ liệu pre-train và tập đánh giá
   (trung vị 2,2%) đúng vì lý do này.
2. **Zero-shot của CLIP ≠ zero-shot learning (ZSL) cổ điển.** Trong ZSL truyền thống
   (Lampert et al. 2009; Xian et al. 2019) có sự phân chia nghiêm ngặt *seen class* / *unseen
   class*, và mô hình chuyển giao qua **attribute** do người định nghĩa. CLIP dùng "zero-shot"
   theo nghĩa rộng hơn: *không cần ví dụ có nhãn cho task đích*. Bài báo CLIP có nói rõ điều này.
   Nếu thầy hỏi "zero-shot theo định nghĩa nào?" thì đây là câu trả lời cần có.

### Liên hệ

Zero-shot → few-shot (k ví dụ/lớp) → linear probe (đóng băng encoder, train classifier tuyến
tính) → fine-tune toàn bộ. Bài báo CLIP báo cáo một hiện tượng phản trực giác đáng nhớ:
**few-shot logistic regression với k nhỏ có thể *kém hơn* zero-shot**, vì vài ví dụ nhiễu còn
tệ hơn tri thức ngôn ngữ đã có.

---

## 3. CLIP classification khác classifier truyền thống ở đâu?

### Bảng đối chiếu

| | Classifier truyền thống | CLIP zero-shot |
|---|---|---|
| Đầu ra cuối | Lớp fully-connected $W \in \mathbb{R}^{C \times d}$ **học từ ảnh có nhãn** | $W = [h(t_1); \dots; h(t_C)]$ **sinh ra từ text encoder** |
| Tập lớp | Cố định tại lúc train | Thay được lúc inference, chỉ cần viết câu mô tả |
| Thêm 1 lớp mới | Phải thu thập dữ liệu + train lại | Thêm 1 chuỗi text |
| Giám sát | Nhãn rời rạc (ảnh → id lớp) | Ngôn ngữ tự nhiên (ảnh ↔ caption) |

### Điểm cần nhấn mạnh (mức cao học)

**Lúc suy luận, CLIP zero-shot *là* một classifier tuyến tính.** Nó không phải một cơ chế khác
về mặt tính toán: $\text{logits} = \tau^{-1} \cdot \hat{I} W^\top$ với $\hat I$ là embedding ảnh
đã chuẩn hóa L2. Khác biệt duy nhất là **$W$ đến từ đâu**: text encoder đóng vai trò
**hypernetwork** sinh trọng số classifier từ ngôn ngữ, thay vì học $W$ bằng gradient descent
trên ảnh có nhãn. Nói được câu này là nói được "cơ chế", không phải "CLIP nhận diện ảnh".

Hệ quả là các cặp lớp mà *tên* của chúng nằm gần nhau trong không gian text embedding sẽ khó
phân biệt, dù ảnh có phân biệt được hay không — đây là một nguồn lỗi **không tồn tại** trong
classifier truyền thống. (Giả thuyết H4 ở mục 7.)

### Về temperature $\tau$ — một câu bẫy

Code của nhóm (`demo_clip.py`) nhân similarity với hằng số `100.0`; đó là
$\tau^{-1} = \exp(\text{logit\_scale}) \approx 100$ của checkpoint đã train (CLIP khởi tạo
$\tau = 0{,}07$ và **clip $\tau^{-1}$ ở mức 100** để ổn định huấn luyện).

**Quan trọng:** $\tau$ là một phép biến đổi **đơn điệu** trên logits ⇒ **không làm thay đổi
thứ tự các lớp** ⇒ **không làm thay đổi Top-1 hay Top-5**. Nó chỉ thay đổi *độ tự tin*
(xác suất sau softmax) và *giá trị loss*. Nếu thầy hỏi "đổi temperature thì accuracy có đổi
không?" → **không**, với điều kiện $\tau$ là cùng một hằng số cho mọi lớp.

---

## 4. Hàm loss: ý nghĩa, giá trị nhỏ nhất/lớn nhất

### ⚠️ Điều phải nói trước tiên: bài chạy của nhóm KHÔNG tính loss

Đây là chỗ dễ bị hỏi gãy nhất. Nhóm chạy **zero-shot inference**: không train, không backward,
**không có loss nào được tính**. Trong `demo_clip.py` chỉ có `softmax` + `topk`, không có
hàm loss.

Vì vậy khi thầy hỏi "hàm loss của em là gì?", câu trả lời trung thực là:

> "Bài của em không tối ưu loss nào. Loss mà em trình bày là **loss của giai đoạn pre-train
> CLIP** do OpenAI thực hiện trên 400 triệu cặp ảnh–text; em chỉ dùng lại checkpoint đó."

Nói "em dùng cross-entropy" khi không train gì là **sai sự thật** và sẽ bị bắt ngay khi thầy
hỏi "loss của em bao nhiêu?".

### a) Cross-entropy (nếu/khi nhóm fine-tune)

$$p_{i,c} = \frac{e^{z_{i,c}}}{\sum_{j} e^{z_{i,j}}}, \qquad
\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i} \log p_{i, y_i}$$

| Câu hỏi của thầy | Trả lời |
|---|---|
| Loss dùng để làm gì? | Biến "dự đoán sai/đúng" thành một đại lượng **khả vi** để lấy gradient. Accuracy không khả vi nên không tối ưu trực tiếp được. |
| Giá trị nhỏ nhất | **0**, đạt khi $p_{i,y_i} = 1$ với mọi $i$ (tự tin tuyệt đối và đúng). |
| Giá trị lớn nhất | **Không bị chặn trên** ($\to +\infty$ khi $p_{i,y_i} \to 0$). Đây là lý do một ảnh bị dự đoán sai *rất tự tin* có thể chi phối cả batch. |
| Mốc tham chiếu quan trọng | Model đoán ngẫu nhiên đều ⇒ $\mathcal{L} = \ln C$. Với CIFAR-100: $\ln 100 = 4{,}61$. Với CIFAR-10: $\ln 10 = 2{,}30$. **Loss cao hơn $\ln C$ nghĩa là model đang tệ hơn đoán bừa.** |
| Loss nhỏ nghĩa là gì? | Xác suất gán cho lớp đúng cao. Nhưng loss nhỏ trên *train* + loss lớn trên *val* = overfitting. |

Ví dụ thầy đưa (ảnh thật là cat; cat 0,7 / dog 0,2 / bird 0,1):
$\mathcal{L} = -\ln 0{,}7 = 0{,}357$. **Chỉ xác suất của lớp đúng đi vào công thức** — nếu đổi
dog↔bird thành 0,1/0,2 thì loss **không đổi**. Đây chính là ý "loss dùng thông tin của
ground-truth class".

Tính trên dữ liệu thật của nhóm (ảnh snake, $p = 0{,}674$): $-\ln 0{,}674 = 0{,}395$.
Ảnh CIFAR-10 index 3637 (ground truth `bird` nhưng model đoán `cat` 0,809; $p_{bird} = 0{,}0295$):
$-\ln 0{,}0295 = 3{,}52$ — **gấp 9 lần**. Một ảnh sai tự tin "đắt" hơn nhiều ảnh đúng vừa phải.

Gradient theo logit: $\partial \mathcal{L} / \partial z_{i,c} = p_{i,c} - y_{i,c}$ — gọn đúng
như vậy, đây là lý do cặp softmax + CE được dùng chứ không phải softmax + MSE.

### b) Loss thật của CLIP: symmetric InfoNCE (contrastive)

Với batch $N$ cặp $(\text{ảnh}_i, \text{text}_i)$, embedding đã chuẩn hóa L2,
$s_{ij} = \hat I_i \cdot \hat T_j / \tau$:

$$\mathcal{L}_{CLIP} = \frac{1}{2}\Big[
\underbrace{-\frac{1}{N}\sum_i \log \frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}}_{\text{ảnh} \to \text{text}}
+ \underbrace{-\frac{1}{N}\sum_i \log \frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}}_{\text{text} \to \text{ảnh}}
\Big]$$

- **Bản chất**: đây là cross-entropy trên bài toán phân loại $N$ lớp, trong đó "lớp đúng" của
  ảnh $i$ là text $i$. $N$ cặp đúng trên đường chéo, $N^2 - N$ cặp sai ngoài đường chéo.
- **Vì sao dùng loss này thay vì sinh caption?** Bài báo (Figure 2) cho thấy dự đoán chính xác
  từng từ của caption là bài toán *quá khó và dư thừa* (một ảnh có vô số caption đúng). Chỉ cần
  học *cặp nào đi với cặp nào* thì hiệu quả tính toán cao hơn ~4 lần so với mục tiêu
  bag-of-words, và ~12 lần so với sinh caption.
- **Giá trị nhỏ nhất**: 0 (lý thuyết). **Mốc ngẫu nhiên**: $\ln N$ — với batch size 32.768 của
  CLIP thì $\ln 32768 = 10{,}4$. Đây là chi tiết đáng nhớ: **loss của InfoNCE phụ thuộc batch
  size**, nên *không so sánh được giá trị loss giữa hai lần train khác batch size*. Batch lớn =
  bài toán khó hơn = tín hiệu học tốt hơn, và đó là một lý do CLIP cần batch cực lớn.
- **Tính đối xứng**: hai chiều là cần thiết vì retrieval theo cả hai hướng (tìm text cho ảnh
  *và* tìm ảnh cho text).

### Liên hệ

InfoNCE (Oord et al. 2018) là nền của cả SimCLR, MoCo. CLIP có thể xem như contrastive learning
với "augmentation" là **chuyển modality** thay vì crop/color-jitter.

---

## 5. Top-1 và Top-5 accuracy

$$\text{Acc@}k = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[y_i \in \text{TopK}(p_i, k)\big]$$

- **Top-1**: lớp có score cao nhất phải đúng.
- **Top-5**: ground truth nằm trong 5 lớp score cao nhất là tính đúng.

**Tính chất phải nhớ:** $\text{TopK}(p,1) \subseteq \text{TopK}(p,3) \subseteq \text{TopK}(p,5)$
⇒ $\text{Acc@}1 \le \text{Acc@}3 \le \text{Acc@}5$, **luôn luôn**, và $\text{Acc@}C = 100\%$.
Đây là hệ quả của định nghĩa tập hợp, **không phải** tính chất của model. Nếu ai báo cáo
Top-5 < Top-1 thì đó là **bug trong code**, không phải kết quả.

Kiểm tra bằng số liệu nhóm: 64,48 ≤ 88,40 ✓ và 88,30 ≤ 99,24 ✓.

---

## 6. Vì sao Top-5 có thể vô nghĩa?

### Lập luận định lượng (mạnh hơn "gần như 100%")

Baseline đoán ngẫu nhiên: $\text{Acc@}k^{random} = k/C$.

| Dataset | $C$ | Random Top-1 | Random Top-5 | Nhóm đo Top-5 | Khoảng cải thiện còn lại |
|---|---|---|---|---|---|
| CIFAR-10 | 10 | 10% | **50%** | 99,24% | 0,76 điểm |
| CIFAR-100 | 100 | 1% | **5%** | 88,40% | 11,6 điểm |

- Trên **CIFAR-10**, Top-5 chỉ yêu cầu model loại được 5 trong 10 lớp. Baseline ngẫu nhiên đã là
  50%, và model đạt 99,24% — **sàn thì cao, trần thì đã chạm**. Metric này không phân biệt được
  hai model bất kỳ. Đưa nó vào báo cáo không sai, nhưng **không mang thông tin**.
- Trên **CIFAR-100**, Top-5 yêu cầu chọn đúng 5 trong 100 lớp; baseline chỉ 5%. Việc model đạt
  88,40% là một thông tin thực sự.

### Trường hợp suy biến

Nếu $C = 5$ và dùng Top-5 thì $\text{Acc@}5 = 100\%$ **chính xác tuyệt đối, theo định nghĩa** —
không phải "gần như 100%". Metric mất hoàn toàn khả năng phân biệt. Quy tắc: **chỉ dùng Top-$k$
khi $k \ll C$.**

### Điểm quan trọng nhất: Top-5 trả lời một câu hỏi *khác*

- **Top-1** = "hệ thống phải đưa ra một câu trả lời duy nhất, nó có đúng không?" — đúng cho hệ
  thống tự động, không có người xem lại.
- **Top-5** = "model có đưa đáp án đúng vào nhóm ứng viên tốt nhất không?" — chỉ có ý nghĩa
  *nghiệp vụ* nếu hệ thống thật sự **hiển thị k ứng viên cho người dùng chọn** (gợi ý tag,
  retrieval, hỗ trợ chẩn đoán có bác sĩ xác nhận).
- Với **CIFAR-100 của nhóm**, khoảng cách 64,48% → 88,40% mang một thông tin chẩn đoán rất cụ
  thể: **trong ~24% ảnh mà model sai ở Top-1, đáp án đúng vẫn nằm trong 5 ứng viên đầu**. Nghĩa
  là model *không* mù hoàn toàn về lớp đó — nó đã khoanh đúng vùng ngữ nghĩa nhưng không xếp
  hạng nổi lớp đúng lên đầu. Đây là **bằng chứng ủng hộ giả thuyết "lẫn giữa các lớp gần nhau"**
  (H1/H4 mục 7), không phải giả thuyết "feature extraction hỏng".

> Đây là cách biện luận cho việc dùng Top-5: **không phải vì bài báo dùng, mà vì hiệu
> Top-5 − Top-1 là một công cụ chẩn đoán lỗi.**

---

## 7. Model chỉ đạt ~65% — làm gì để tìm nguyên nhân?

### Bước 0: kiểm tra xem 65% có thật sự là "thấp" không (bắt buộc làm trước)

Đây là bước hầu hết mọi người bỏ qua. **Trước khi đi tìm lỗi, phải so với số liệu đã công bố.**

CLIP ViT-B/32 zero-shot trên CIFAR-100 trong bài báo gốc (Table 11) ở **khoảng 65%** — tức là
kết quả 64,48% / 65,08% của nhóm **khớp với số đã công bố**.

> ⚠️ *Tôi không chắc chắn con số chính xác đến chữ số thập phân — nhóm PHẢI tự tra Table 11 của
> bài báo CLIP và ghi lại đúng con số + số hiệu bảng.* Nếu khớp, đây là câu trả lời mạnh nhất có
> thể có:
>
> "65% không phải là kết quả kém — đó là **reproduce đúng** con số của bài báo cho ViT-B/32.
> Câu hỏi đúng không phải 'sao code của em tệ' mà là **'vì sao bản thân CLIP ViT-B/32 chỉ đạt
> mức đó trên CIFAR-100, trong khi đạt 88–90% trên CIFAR-10?'**"

Bước 0 này cũng **loại bỏ ngay** các giả thuyết "preprocessing sai", "code lỗi", "chia sai
train/test" — nếu code sai thì không thể ra đúng số của bài báo.

### Các giả thuyết còn lại, và dữ liệu hiện có nói gì

| # | Giả thuyết | Dữ liệu hiện có | Kết luận tạm |
|---|---|---|---|
| H0 | Pipeline sai / preprocessing sai | Khớp số bài báo; CIFAR-10 = 88,3% hợp lý | **Đã loại** |
| H1 | Các lớp fine quá giống nhau | Top-5 = 88,4% ≫ Top-1 = 64,5%: đáp án đúng *có* trong top-5 | **Ủng hộ mạnh** |
| H2 | Prompt chưa tốt | 1 template 64,48% → 18 template 65,08%, chỉ **+0,60 điểm** | **Ảnh hưởng nhỏ** (xem mục 8) |
| H3 | Ảnh 32×32 mất đặc trưng | ViT-B/32 chia patch 32×32 trên ảnh resize 224 ⇒ ảnh CIFAR bị upsample 7× từ 32→224, không tạo thêm thông tin | **Cần thí nghiệm** |
| H4 | Tên lớp gần nhau trong text embedding space | Chưa đo | **Cần thí nghiệm** |
| H5 | Nhãn sai trong test set (label noise) | Chưa đo | **Cần thí nghiệm** (xem mục 9) |
| H6 | Model quá nhỏ | Chưa thử model khác | **Cần thí nghiệm** |

### Thí nghiệm cụ thể để kiểm chứng (thiết kế được ngay với repo hiện tại)

**E1 — Confusion matrix + nhóm lỗi theo superclass (ưu tiên cao nhất).**
Lưu dự đoán từng ảnh, rồi tính: *trong các ảnh sai, bao nhiêu % bị nhầm sang một lớp **cùng
superclass**?* Baseline ngẫu nhiên: nếu lỗi phân bố đều trên 99 lớp sai, tỉ lệ cùng superclass
chỉ là $4/99 = 4{,}0\%$. Nếu đo được tỉ lệ cao hơn nhiều (ví dụ > 20%) thì **H1 được xác nhận
với bằng chứng định lượng**. Đây là thí nghiệm rẻ nhất và mạnh nhất.

**E2 — Đánh giá bằng nhãn coarse (20 superclass).** Chạy lại zero-shot với 20 tên superclass.
Nếu accuracy nhảy lên rất cao (dự đoán > 85%) thì lỗi nằm ở tầng phân biệt *mịn*, không phải ở
khả năng nhìn. Đây là cách tách bạch "không nhìn thấy gì" khỏi "nhìn thấy nhưng không phân
biệt nổi".

**E3 — Đo khoảng cách giữa các text embedding.** Tính ma trận $\cos(h(t_i), h(t_j))$ cho 100 tên
lớp, đối chiếu các cặp similarity cao nhất với các cặp bị nhầm nhiều nhất ở E1. Nếu tương quan
mạnh ⇒ **H4 đúng**: nghẽn ở *phía ngôn ngữ*, không phải phía ảnh. Kiểm tra được bằng hệ số
tương quan Spearman giữa (độ giống tên lớp) và (số lần bị nhầm).

**E4 — Đổi model, giữ nguyên mọi thứ khác (ablation cho H6).** ViT-B/32 → ViT-B/16 → ViT-L/14.
Nếu accuracy tăng đáng kể thì nghẽn là *dung lượng model*. Đây là ablation một biến, đúng
phương pháp.

**E5 — Kiểm tra H3.** So sánh accuracy khi resize 32→224 bằng bicubic (mặc định) với các cách
nội suy khác, hoặc so ảnh CIFAR với ảnh cùng lớp ở độ phân giải cao. Nếu ảnh phân giải cao cho
kết quả tốt hơn nhiều ⇒ nghẽn là độ phân giải, và đó là **giới hạn của dataset, không phải của
model**.

**E6 — Soi lỗi bằng mắt.** Xuất 50 ảnh sai tự tin nhất (xác suất lớp sai cao nhất). Con người
xem và phân loại: (a) nhãn dataset sai, (b) ảnh thật sự mơ hồ, (c) model sai rõ ràng. Tỉ lệ
(a) là estimate cho label noise (H5).

### Quy trình cần trình bày (đúng tinh thần thầy yêu cầu)

```
Result (64,48%)
  → So với baseline đã công bố  ← BƯỚC HAY BỊ BỎ QUÊN
  → Đặt giả thuyết (H1…H6)
  → Thiết kế thí nghiệm một-biến (E1…E6)
  → Chạy, báo cáo cả thí nghiệm bác bỏ giả thuyết
  → Kết luận dựa trên bằng chứng
```

Điểm mấu chốt: **một thí nghiệm bác bỏ giả thuyết của mình cũng là kết quả có giá trị** và phải
được ghi vào báo cáo.

---

## 8. Phần thống kê — chỗ dễ bị hỏi gãy nhất

### 8.1. Con số 75% trên 128 ảnh không được dùng để báo cáo

`run_eval()` trong `demo_clip.py` lấy **128 ảnh ĐẦU TIÊN** của tập test
(`for i in range(start, stop)`), **không phải mẫu ngẫu nhiên**.

Hệ quả: 75,0% đó **không phải là một ước lượng không chệch** cho accuracy trên toàn tập, và
**không được phép** gắn khoảng tin cậy nhị thức vào nó. Đối chiếu: 75,0% (n=128) vs 64,48%
(n=10.000) lệch hơn 10 điểm — nếu là mẫu ngẫu nhiên thì đó là biến cố ~2,7σ. Cách giải thích
hợp lý hơn: **slice 128 ảnh đầu không đại diện**.

Nếu thầy hỏi "sao hai lần chạy khác nhau 10 điểm?" → đây là câu trả lời. Và bài học phương pháp:
**muốn dùng tập con thì phải lấy mẫu ngẫu nhiên có seed cố định**, không phải lấy N ảnh đầu.

### 8.2. Phải báo cáo khoảng tin cậy, không phải một con số trần

Với accuracy là tỉ lệ nhị thức, $SE = \sqrt{p(1-p)/n}$, khoảng tin cậy 95% ≈ $p \pm 1{,}96\,SE$:

| Kết quả | $n$ | $SE$ | 95% CI |
|---|---|---|---|
| CIFAR-100 Top-1, 1 template | 10.000 | 0,48 điểm | **[63,54 ; 65,42]** |
| CIFAR-100 Top-1, 18 template | 10.000 | 0,48 điểm | **[64,15 ; 66,01]** |
| CIFAR-10 Top-1, 1 template | 10.000 | 0,32 điểm | [87,67 ; 88,93] |
| CIFAR-100 (slice 128 ảnh) | 128 | 3,8 điểm* | — *(không hợp lệ: không phải mẫu ngẫu nhiên)* |

Viết "64,48%" mà không có CI là ngầm khẳng định độ chính xác đến 2 chữ số thập phân — điều mà
n = 10.000 không cho phép. Nên viết **64,5 ± 0,5%**.

### 8.3. Cải thiện +0,60 điểm của prompt ensemble CÓ Ý NGHĨA THỐNG KÊ KHÔNG?

Hai CI ở trên **chồng lấn nhau rất nhiều**. Kiểm định thô (coi hai mẫu độc lập):
$SE_{diff} = 0{,}68$ điểm, $z = 0{,}60/0{,}68 = 0{,}89$ ⇒ **không có ý nghĩa thống kê**
($p \approx 0{,}37$).

Nhưng kiểm định đó **quá bảo thủ**, vì hai lần chạy **trên đúng 10.000 ảnh giống nhau** —
đây là dữ liệu **cặp (paired)**. Kiểm định đúng phải là **McNemar's test** trên bảng chéo
(ảnh mà A đúng/B sai và A sai/B đúng).

**Vấn đề:** McNemar cần dự đoán từng ảnh, và `demo_clip.py` **chỉ giữ biến đếm `top1`, `top5`,
rồi bỏ toàn bộ dự đoán từng ảnh**. Vì vậy hiện tại nhóm **không thể**:

- vẽ confusion matrix (⇒ không làm được E1),
- biết lớp nào bị nhầm với lớp nào (⇒ không làm được error analysis nào),
- kiểm định xem +0,60 điểm là thật hay là nhiễu.

> **Đây là việc cần sửa đầu tiên trước buổi sau**: lưu lại `(index, ground_truth, top5_pred,
> top5_prob)` cho từng ảnh ra file (CSV/NPZ). Toàn bộ mục 7 phụ thuộc vào việc này. Nếu không
> có nó, mọi câu trả lời "tại sao 65%" đều chỉ là suy đoán.
>
> Nếu chưa lưu được, **đừng kết luận "prompt ensemble giúp cải thiện"** — với dữ liệu hiện có,
> +0,60 điểm chưa phân biệt được với nhiễu.

---

## 9. Label noise — nói cho đúng

Ghi chú buổi học viết: *"model có thể học theo label sai"*. **Với bài của nhóm thì câu này không
áp dụng**, vì zero-shot **không học gì từ nhãn CIFAR**. Phải phân biệt rõ hai tác động:

| | Ảnh hưởng của nhãn sai |
|---|---|
| **Khi train** (fine-tune) | Model học theo nhãn sai ⇒ hỏng cả tham số. Nghiêm trọng. |
| **Khi zero-shot** (bài của nhóm) | Không ảnh hưởng tham số. Chỉ làm **méo phép đo**: model dự đoán đúng nhưng bị tính là sai. |

Với zero-shot, label noise **đặt một trần trên accuracy đo được**: nếu $\epsilon$ là tỉ lệ nhãn
sai trong test set, accuracy đo được không thể vượt quá khoảng $1 - \epsilon$.

Northcutt et al. (2021), *Pervasive Label Errors in Test Sets*, ước lượng tỉ lệ nhãn sai trong
test set CIFAR-100 ở mức **khoảng 6%** (CIFAR-10 thấp hơn nhiều, dưới 1%).
*Tôi không chắc con số chính xác — nhóm nên tra lại bài báo đó hoặc labelerrors.com trước khi
trích dẫn.* Nếu con số đúng cỡ đó thì trần đo được của CIFAR-100 chỉ khoảng 94%, và **một phần
trong 35% lỗi của nhóm là lỗi không thể sửa được bằng bất kỳ model nào**.

Điều này *không* giải thích được khoảng cách 65% → 94%, nên **đừng dùng label noise làm cái cớ
chính**. Nó là một thành phần cần định lượng (qua E6), không phải câu trả lời.

---

## 10. Sửa lại một số chỗ trong ghi chú buổi học

| Ghi chú viết | Cần sửa thành |
|---|---|
| "Bài toán chỉ có 5 class mà dùng Top-5 → **gần như** chắc chắn 100%" | **Chính xác bằng 100%**, theo định nghĩa. Không phải "gần như". |
| "Loss nhỏ nhất/lớn nhất là gì" (không nêu số) | Min = 0. **Max không bị chặn** ($+\infty$). Mốc đoán bừa = $\ln C$ (CIFAR-100: 4,61). Với InfoNCE, mốc = $\ln N$ và **phụ thuộc batch size**. |
| "model có thể học theo label sai" | Chỉ đúng khi *train*. Zero-shot: label noise làm **méo phép đo**, không làm hỏng model. |
| "CIFAR-100 khó hơn vì nhiều lớp hơn" | Nhiều lớp hơn chỉ hạ baseline (10%→1%). Cái làm nó khó là **lớp cùng superclass giống nhau** + **500 ảnh/lớp** + **32×32**. |
| Không nhắc temperature | Cần biết: $\tau^{-1} \approx 100$ trong code, và $\tau$ **không đổi Top-k** (biến đổi đơn điệu), chỉ đổi độ tự tin và giá trị loss. |
| Chưa nói rõ "bài của em không có loss" | Phải nói rõ: zero-shot ⇒ **không tính loss nào**. Loss trình bày là loss pre-train của OpenAI. |
| "Result → hypothesis → experiment" | Thiếu **Bước 0: so với số liệu đã công bố**. 65% khớp bài báo ⇒ không phải bug ⇒ đổi hoàn toàn hướng phân tích. |

---

## 11. Việc cần làm trước buổi sau (theo thứ tự ưu tiên)

1. **Lưu dự đoán từng ảnh** ra CSV/NPZ trong `run_eval()`. Mọi thứ ở mục 7 phụ thuộc vào đây.
2. **Tra Table 11 bài báo CLIP**, ghi lại con số zero-shot CIFAR-100 của ViT-B/32 + số hiệu bảng.
   Đây là câu trả lời mạnh nhất cho "tại sao 65%".
3. **Chạy E1** (confusion matrix + tỉ lệ lỗi cùng superclass, so với baseline 4/99 = 4,0%).
4. **Chạy E2** (đánh giá bằng 20 nhãn coarse) — rẻ, và tách bạch rất rõ nguồn lỗi.
5. **Thêm khoảng tin cậy** vào mọi con số trong báo cáo; **bỏ hoặc ghi rõ** con số 75% (n=128).
6. Sửa lại kết luận về prompt ensemble: chưa chứng minh được +0,60 điểm là thật.
7. Chuẩn bị nói được: *"Bài của em là inference, không có loss."*

---

## 12. Bảy câu hỏi – phiên bản trả lời một câu

| Câu hỏi | Trả lời cốt lõi |
|---|---|
| CIFAR-10 vs CIFAR-100, tại sao chọn? | Cùng ảnh 32×32, khác ở **độ mịn nhãn** (lớp cùng superclass giống nhau) và **ít ảnh/lớp**. Chọn CIFAR-100 vì CIFAR-10 đã bão hòa (Top-5 = 99,2%) nên không quan sát được gì; chênh 24 điểm giữa hai tập trên cùng pipeline mới là dữ liệu để phân tích. |
| Zero-shot là gì? | Phân loại **không dùng ảnh có nhãn nào của dataset đích**; classifier được sinh từ tên lớp. **Không** có nghĩa model chưa từng thấy khái niệm đó. |
| CLIP khác classifier truyền thống? | Lúc inference CLIP **là** classifier tuyến tính; khác ở chỗ $W$ do **text encoder sinh ra** (hypernetwork) thay vì học từ ảnh có nhãn ⇒ đổi tập lớp không cần train lại. |
| Loss nghĩa là gì, sao càng thấp càng tốt? | Là đại lượng **khả vi** thay cho accuracy. CE: min 0, **max vô cực**, mốc đoán bừa $\ln C$. Loss thấp = xác suất gán cho **lớp đúng** cao. *Bài của nhóm là inference nên không tính loss.* |
| Top-1 / Top-5? | $\text{Acc@}k$ = ground truth nằm trong $k$ lớp điểm cao nhất; luôn có Acc@1 ≤ Acc@5 theo định nghĩa. |
| Sao Top-5 có thể vô nghĩa? | Baseline ngẫu nhiên là $k/C$: với $C=10$ là **50%**, với $C=100$ là 5%. $k$ gần $C$ ⇒ metric không phân biệt được model; $k = C$ ⇒ đúng 100%. |
| Model chỉ ~65%, làm gì? | **Trước tiên so với số bài báo** (khớp ⇒ không phải bug). Sau đó: Top-5 = 88,4% ≫ Top-1 = 64,5% ⇒ giả thuyết "lẫn giữa các lớp gần nhau" ⇒ kiểm chứng bằng confusion matrix + đánh giá theo nhãn coarse + đo khoảng cách text embedding. |

---

## Tài liệu tham khảo

1. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*
   (CLIP). ICML 2021. — Section 2.3 (loss), Section 3.1 (zero-shot), Table 11 (số liệu per-dataset).
2. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Tech. report,
   University of Toronto, 2009. — Bài báo giới thiệu CIFAR-10/100.
3. A. van den Oord, Y. Li, O. Vinyals. *Representation Learning with Contrastive Predictive
   Coding*. arXiv:1807.03748, 2018. — Nguồn gốc InfoNCE.
4. C. G. Northcutt, A. Athalye, J. Mueller. *Pervasive Label Errors in Test Sets Destabilize
   Machine Learning Benchmarks*. NeurIPS 2021 Datasets & Benchmarks. — Label noise của CIFAR.
5. B. Recht et al. *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* arXiv:1806.00451, 2018.
6. Y. Xian, C. H. Lampert, B. Schiele, Z. Akata. *Zero-Shot Learning — A Comprehensive
   Evaluation of the Good, the Bad and the Ugly*. IEEE TPAMI, 2019. — Định nghĩa ZSL cổ điển.
7. Q. McNemar. *Note on the sampling error of the difference between correlated proportions*.
   Psychometrika, 1947. — Kiểm định đúng cho hai model trên cùng tập test.
8. A. Kumar et al. *Fine-Tuning can Distort Pretrained Features and Underperform
   Out-of-Distribution*. ICLR 2022.
