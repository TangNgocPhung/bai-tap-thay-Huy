# Bài tập thầy Huy – Trả lời phần lý thuyết

Trả lời các câu hỏi thầy yêu cầu bổ sung: CIFAR-10/100, quy trình thu thập, bài báo gốc, hàm Loss, Top-k, lớp *snake*, số epoch khi fine-tune CLIP.

> Những chỗ ghi **[nhóm điền]** cần thay bằng số liệu thực tế của nhóm.

---

## 1. CIFAR-10 và CIFAR-100 khác nhau thế nào

| | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Tổng số ảnh | 60.000 ảnh màu 32×32 | 60.000 ảnh màu 32×32 |
| Số lớp | 10 | 100 lớp con (fine), gom thành 20 lớp lớn (coarse) |
| Ảnh mỗi lớp | 6.000 (5.000 train + 1.000 test) | 600 (500 train + 100 test) |
| Nhãn | 1 nhãn | 2 nhãn: fine và coarse |
| Độ khó | Các lớp khác nhau rõ (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) | Khó hơn nhiều: ít ảnh mỗi lớp, nhiều lớp rất giống nhau trong cùng lớp lớn |

Ví dụ: lớp lớn *reptiles* gồm crocodile, dinosaur, lizard, **snake**, turtle. Cả 10 lớp của CIFAR-10 loại trừ nhau, ví dụ *automobile* và *truck* không chồng lên nhau.

**Mục đích:** tạo một bộ benchmark nhỏ, có nhãn tốt và cân bằng để so sánh các thuật toán học đặc trưng và phân loại ảnh. Ảnh nhỏ nên train nhanh. CIFAR là viết tắt của *Canadian Institute For Advanced Research*, đơn vị tài trợ.

## 2. Quy trình thu thập (dataset lấy mẫu như thế nào)

1. **Nguồn ảnh:** CIFAR là tập con của **80 Million Tiny Images** (Torralba, Fergus, Freeman, 2008). Bộ này được thu bằng cách gõ các danh từ trong WordNet vào các công cụ tìm ảnh (Google, Flickr, …), rồi thu nhỏ ảnh về 32×32.
2. **Chọn ảnh ứng viên:** với mỗi lớp, lấy các ảnh tìm được bằng tên lớp và các từ con của nó (ví dụ lớp *ship* thì tìm cả các loại tàu).
3. **Gán nhãn thủ công:** sinh viên được trả tiền để kiểm tra từng ảnh. Một ảnh được giữ nếu:
   - tên lớp là câu trả lời hợp lý hàng đầu cho câu hỏi "Trong ảnh có gì?";
   - ảnh trông giống ảnh chụp thật (photo-realistic);
   - chỉ có **một** đối tượng chính của lớp đó (có thể bị che một phần hoặc ở góc nhìn lạ, miễn là vẫn nhận ra được).
4. **Loại ảnh trùng và gần trùng** (so khoảng cách L2 giữa các ảnh).
5. **Chia train/test:** tập test được **lấy ngẫu nhiên theo từng lớp** (CIFAR-10: 1.000 ảnh/lớp, CIFAR-100: 100 ảnh/lớp), phần còn lại là tập train.
   - Train và test **cân bằng tuyệt đối** giữa các lớp.

**Lưu ý:** đây không phải lấy mẫu ngẫu nhiên từ "thế giới thật". Ảnh được lấy theo từ khóa tìm kiếm rồi lọc thủ công, nên dataset mang thiên lệch của công cụ tìm ảnh. Recht và cộng sự (2018) đã làm lại đúng quy trình này để tạo **CIFAR-10.1**, và độ chính xác của các mô hình giảm khoảng 4–10%.

## 3. Bài báo đầu tiên sử dụng

**Alex Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*, Technical Report, University of Toronto.** Người hướng dẫn là Geoffrey Hinton. Đây là báo cáo giới thiệu CIFAR-10/100 và dùng chúng để huấn luyện RBM / Deep Belief Network.

## 4. Hàm Loss (có công thức)

### a) Cross-Entropy dùng cho phân loại

Mô hình cho ra vector logit $z_i \in \mathbb{R}^C$ với mỗi ảnh $x_i$, trong đó $C$ là số lớp. Softmax đổi logit thành xác suất:

$$p_{i,c} = \frac{e^{z_{i,c}}}{\sum_{j=1}^{C} e^{z_{i,j}}}$$

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c}\log p_{i,c} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}$$

- $N$: số ảnh trong batch. $y_{i,c}$: nhãn one-hot, bằng 1 nếu ảnh $i$ thuộc lớp $c$.
- Ý nghĩa: phạt mô hình khi xác suất dành cho lớp đúng thấp. Nếu $p=1$ thì loss bằng 0. Nếu $p\to 0$ thì loss tiến tới ∞.
- Gradient theo logit: $\frac{\partial \mathcal{L}}{\partial z_{i,c}} = p_{i,c} - y_{i,c}$. Mô hình tăng logit của lớp đúng và giảm logit của các lớp sai.

### b) Loss của CLIP (dành cho nhóm fine-tune CLIP)

Gọi $I_i$ là embedding ảnh và $T_j$ là embedding văn bản, cả hai đã chuẩn hóa L2. $\tau$ là temperature học được:

$$s_{ij} = \frac{I_i \cdot T_j}{\tau}$$

$$\mathcal{L}_{CLIP} = \frac{1}{2}\left[-\frac{1}{N}\sum_{i}\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} - \frac{1}{N}\sum_{i}\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\right]$$

- Đây là contrastive loss đối xứng (InfoNCE), gồm hai chiều ảnh→chữ và chữ→ảnh.
- Nó kéo cặp ảnh–chữ đúng lại gần nhau và đẩy các cặp sai ra xa.
- Khi fine-tune CLIP để phân loại, thường đặt $T_c$ là embedding của câu mô tả mỗi lớp, ví dụ *"a photo of a {class}"*. Khi đó loss trở thành Cross-Entropy ở mục (a) với logit $z_{i,c} = \cos(I_i, T_c)/\tau$.

**[Nhóm điền]:** ghi rõ nhóm dùng loss nào: CE thường, CE có label smoothing, hay contrastive.

## 5. Top-1, Top-5, Top-3 và "biến k"

$$\text{Acc@}k = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[y_i \in \text{TopK}(p_i, k)\right]$$

$\text{TopK}(p_i,k)$ là tập $k$ lớp có xác suất cao nhất.

**Tại sao top-5 luôn ≥ top-3 ≥ top-1:** tập top-1 nằm trong tập top-3, và tập top-3 nằm trong tập top-5. Ảnh nào đã đúng ở top-1 thì chắc chắn đúng ở top-5. Vì vậy Acc@k **không giảm khi k tăng**, và Acc@C = 100%. Đây là hệ quả toán học của định nghĩa, không phải do mô hình.

**Tại sao chọn top-1 và top-5 (theo bài báo):**
- Đây là quy ước của cuộc thi ImageNet ILSVRC (Russakovsky et al., 2015). AlexNet (Krizhevsky et al., 2012) báo cáo cả top-1 và top-5, và các bài sau (ResNet, CLIP, …) theo cùng chuẩn để so sánh được với nhau.
- Với CIFAR-100, top-5 có ý nghĩa vì nhiều lớp rất giống nhau, ví dụ snake/worm hay lizard/crocodile. Top-5 cho biết mô hình "gần đúng" hay sai hoàn toàn.

**Chạy thử top-3 (PyTorch):**

```python
def topk_acc(logits, y, ks=(1, 3, 5)):
    top = logits.topk(max(ks), dim=1).indices          # [N, maxk]
    hit = top.eq(y.view(-1, 1))                          # [N, maxk]
    return {k: hit[:, :k].any(1).float().mean().item() for k in ks}
```

Kết quả phải thỏa Acc@1 ≤ Acc@3 ≤ Acc@5. Nên vẽ thêm đường Acc@k với k = 1…10 để minh họa.

## 6. Tại sao lớp snake "lớn hơn" các lớp còn lại

Trong CIFAR-100, **số ảnh mỗi lớp bằng nhau** (500 train / 100 test). Vì vậy nếu biểu đồ cho thấy snake "lớn hơn", đó **không phải do dữ liệu mất cân bằng**. Cần xác định biểu đồ đang đo cái gì:

- **Nếu là số lần mô hình *dự đoán* là snake** (cột snake trong confusion matrix lớn): mô hình đang thiên về lớp này. Với CLIP zero-shot, câu prompt "a photo of a snake" có thể khớp với nhiều ảnh có vật dài và mảnh (worm, dây, cành cây). Đây là thiên lệch của text embedding.
- **Nếu là *lỗi* của lớp snake cao:**
  - ở 32×32, rắn chỉ là một nét mảnh và thường ngụy trang cùng màu nền;
  - hình dạng rất đa dạng (cuộn tròn hoặc duỗi thẳng);
  - dễ nhầm với các lớp cùng nhóm hoặc giống nhau: worm, lizard, crocodile.
- **Cách kiểm tra để chỉ ra được:**
  1. Đếm số ảnh mỗi lớp, sẽ thấy đều bằng 100.
  2. Vẽ confusion matrix và xem hàng/cột snake.
  3. Liệt kê top các lớp bị nhầm với snake.
  4. Hiển thị vài ảnh bị nhầm.

## 7. Fine-tune CLIP: chọn bao nhiêu epoch, tại sao

**[Nhóm điền số epoch thực tế].** Lập luận:

- CLIP đã được huấn luyện trước trên 400 triệu cặp ảnh–chữ, nên chỉ cần **ít epoch** (thường 5–10) với **learning rate nhỏ** (khoảng 1e-5), có warmup và cosine decay.
- Train nhiều epoch dễ bị **overfitting** (CIFAR-100 chỉ có 500 ảnh/lớp). Nó cũng làm **mất khả năng tổng quát và zero-shot** của CLIP: xem Kumar et al. 2022 *"Fine-Tuning can Distort Pretrained Features"* và Wortsman et al. 2022 *WiSE-FT*.
- Nên chọn số epoch **dựa trên thực nghiệm**:
  - vẽ đường train loss và validation loss/accuracy theo epoch;
  - chọn epoch mà validation accuracy đạt đỉnh hoặc bắt đầu giảm (early stopping);
  - đưa biểu đồ này vào báo cáo làm bằng chứng.

## Tài liệu tham khảo

1. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. Tech. report, University of Toronto, 2009.
2. A. Torralba, R. Fergus, W. T. Freeman. *80 Million Tiny Images*. IEEE TPAMI, 2008.
3. B. Recht et al. *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* arXiv:1806.00451, 2018.
4. A. Krizhevsky, I. Sutskever, G. Hinton. *ImageNet Classification with Deep Convolutional Neural Networks*. NeurIPS 2012.
5. O. Russakovsky et al. *ImageNet Large Scale Visual Recognition Challenge*. IJCV, 2015.
6. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021.
7. A. Kumar et al. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*. ICLR 2022.
8. M. Wortsman et al. *Robust Fine-Tuning of Zero-Shot Models* (WiSE-FT). CVPR 2022.
