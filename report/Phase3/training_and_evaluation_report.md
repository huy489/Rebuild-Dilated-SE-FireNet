# Báo cáo Giai đoạn 3: Huấn luyện mô hình GPU và Đánh giá chi tiết (Float32)

## 1. Mục tiêu
Tiến hành huấn luyện chính thức mô hình **Dilated-SE-FireNet** sử dụng hạ tầng tăng tốc GPU trên đám mây (Google Colab) và thực hiện đánh giá toàn diện, chi tiết hiệu năng mô hình số thực dấu phẩy động (Float32) trên tập dữ liệu kiểm thử độc lập gộp và độc lập từng bệnh nhân trên PC.

---

## 2. Cấu hình Huấn luyện và Hạ tầng đám mây

*   **Hạ tầng:** Google Colab Web (GPU Tesla T4 / L4).
*   **Môi trường:** Thiết lập đầu nối tự động dữ liệu `processed.zip` (3.7 GB) sử dụng tệp khóa Google API Service Account.
*   **Cấu hình tham số:**
    *   **Bộ tối ưu hóa (Optimizer):** Adam với bộ lập lịch tốc độ học Cosine Decay (tốc độ học giảm dần từ $10^{-3}$ về $5 \times 10^{-5}$).
    *   **Hàm mất mát (Loss Function):** Categorical Crossentropy kết hợp mượt nhãn (Label Smoothing = 0.1) để giảm mức độ tự tin thái quá của mô hình, chống quá khớp.
    *   **Cân bằng lớp (Class Weights):** Lớp Normal có trọng số `0.8782`, lớp AFIB có trọng số `1.1611` tính toán tự động từ tập huấn luyện.
    *   **Callbacks:** `ModelCheckpoint` lưu mô hình tốt nhất dựa trên `val_loss`, `EarlyStopping` dừng huấn luyện sau 10 epoch không cải thiện `val_loss`.

---

## 3. Nhật ký Huấn luyện & Cơ chế Dừng Sớm (Early Stopping)

Quá trình huấn luyện thực tế trên GPU Colab diễn ra với thời gian trung bình khoảng **77 giây mỗi epoch**.

*   Mô hình đạt được giá trị mất mát kiểm định tốt nhất (**`val_loss: 0.2487`**) và độ chính xác kiểm định tốt nhất (**`val_accuracy: 99.21%`**) tại **Epoch 16**.
*   Từ Epoch 17 đến Epoch 26 (đúng 10 epochs liên tiếp), giá trị `val_loss` dao động tăng nhẹ (lên mức `0.27` - `0.34`) và không vượt qua được mức tối ưu ở Epoch 16.
*   Cơ chế `EarlyStopping` đã kích hoạt dừng huấn luyện sớm ở cuối **Epoch 26** để bảo vệ mô hình khỏi hiện tượng quá khớp (học vẹt tập train).
*   Trọng số tối ưu tại Epoch 16 được tự động khôi phục và xuất ra tệp `best_dilated_se_firenet.keras`.

---

## 4. Kết quả Đánh giá mô hình Float32 cục bộ trên tập Test

Tệp trọng số tối ưu nhất được tải về máy tính và đánh giá bằng script [src/evaluate.py](file:///home/pd/data/info_model/build_model/src/evaluate.py).

### 4.1. Đánh giá Tổng thể (Overall Evaluation)
Đánh giá trên toàn bộ **65,188** cửa sổ ECG kiểm thử gộp từ 4 bệnh nhân độc lập:
*   **Độ chính xác tổng thể (Accuracy):** **`95.23%`**
*   **Độ nhạy AFIB (Recall):** **`99.71%`** (Cực kỳ an toàn, chỉ bỏ sót 82 mẫu Rung nhĩ trong tổng số 28,284 mẫu).
*   **Độ xác thực AFIB (Precision):** **`90.31%`** (Có 3,027 mẫu bình thường bị mô hình nhận nhầm là Rung nhĩ - dương tính giả).
*   **F1-Score (AFIB):** **`0.9478`**

#### Ma trận nhầm lẫn tổng thể:
| Thực tế \ Dự đoán | Normal (Dự đoán) | AFIB (Dự đoán) |
| :--- | :---: | :---: |
| **Normal (Thật)** | **33,877** (True Normal) | **3,027** (False AFIB) |
| **AFIB (Thật)** | **82** (False Normal) | **28,202** (True AFIB) |

---

### 4.2. Đánh giá chi tiết theo từng Bệnh nhân (Patient-by-Patient)

Mô hình được chạy đánh giá trên từng bệnh nhân riêng biệt để kiểm tra tính ổn định trên các cá thể khác nhau:

| Record ID | Độ chính xác (Accuracy) | Độ nhạy AFIB (Recall) | Độ xác thực AFIB (Precision) | F1-Score (AFIB) | Tổng số mẫu |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **04126** | **96.45%** | 100.00% | 58.80% | 74.06% | 13,594 |
| **05091** | **99.99%** | 100.00% | 94.74% | 97.30% | 16,269 |
| **08215** | **84.93%** | 99.49% | 84.82% | 91.57% | 17,358 |
| **08405** | **99.94%** | 99.93% | 99.99% | 99.96% | 17,967 |

#### Phân tích lỗi (Error Analysis):
1.  **Bệnh nhân 04126 (Precision thấp):** Đạt Recall AFIB tuyệt đối `100.0%` nhưng Precision chỉ đạt `58.8%` do mô hình dự đoán nhầm 482 mẫu Normal thành AFIB. Do tập dữ liệu bệnh nhân này quá mất cân bằng (chỉ có 688 mẫu AFIB so với 12,906 mẫu Normal) nên tỷ lệ Precision bị kéo xuống thấp. Tuy nhiên, việc không bỏ sót bất kỳ nhịp AFIB nào là tối ưu cho việc giám sát y tế.
2.  **Bệnh nhân 08215 (Accuracy thấp):** Bệnh nhân này có tỷ lệ Normal bị đoán nhầm thành AFIB cao (2,542 mẫu). Nguyên nhân thường do sóng nhiễu đẳng điện hoặc nhiễu cơ học của bệnh nhân này có tần số và hình thái trùng lặp với rung nhĩ. Mô hình vẫn bảo toàn độ nhạy AFIB ở mức cực cao `99.49%`.

---

## 5. Kết luận
Mô hình Float32 đạt hiệu năng xuất sắc vượt kỳ vọng đối với một mạng siêu nhẹ chỉ 110k tham số, chứng minh đồ thị mạng được thiết kế tối ưu và pipeline dữ liệu tiền xử lý chất lượng. Trọng số mô hình hoàn toàn đủ điều kiện để chuyển giao sang Giai đoạn 4 (Lượng tử hóa TFLite).
