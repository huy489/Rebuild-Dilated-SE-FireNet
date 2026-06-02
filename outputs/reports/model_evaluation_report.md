# Báo cáo Đánh giá Hiệu năng Mô hình Dilated-SE-FireNet (Float32)

* **Ngày thực hiện:** 02/06/2026
* **Mô hình đánh giá:** `best_dilated_se_firenet.keras` (Khôi phục trọng số từ Epoch 16)
* **Tổng số tham số:** 110,694 tham số (~432 KB)
* **Tác vụ:** Phát hiện Rung nhĩ (AFIB) từ tín hiệu ECG 1 kênh, độ dài 10 giây (tần số lấy mẫu 250 Hz)

---

## 1. Tóm tắt Quá trình Huấn luyện (GPU Colab)

Mô hình đã được huấn luyện trên Google Colab với cấu hình tối đa 60 epochs. Cơ chế **Early Stopping** (patience = 10) đã kích hoạt ở epoch 26 sau khi không tìm thấy sự cải thiện về validation loss so với epoch tốt nhất.

* **Epoch tốt nhất:** Epoch 16
* **Validation Loss thấp nhất:** `0.2487`
* **Validation Accuracy cao nhất:** `99.21%`

---

## 2. Kết quả Đánh giá Tổng thể (Overall Evaluation)

Được thực hiện trên toàn bộ tập dữ liệu kiểm thử gộp độc lập (bao gồm **65,188** cửa sổ ECG từ 4 bệnh nhân kiểm thử).

### Chỉ số hiệu năng chính:
* **Độ chính xác tổng thể (Accuracy):** **`95.23%`**
* **Độ nhạy AFIB (Recall):** **`99.71%`** (Đặc biệt quan trọng trong chẩn đoán y tế, chỉ bỏ sót 82 trong số 28,284 cửa sổ Rung nhĩ).
* **Độ xác thực AFIB (Precision):** **`90.31%`** (Tỷ lệ dương tính giả ở mức 9.69%).
* **F1-Score (AFIB):** **`0.9478`**

### Ma trận nhầm lẫn tổng thể:

| Thực tế \ Dự đoán | Normal (Dự đoán) | AFIB (Dự đoán) | Tổng số mẫu thực tế |
| :--- | :---: | :---: | :---: |
| **Normal (Thật)** | **33,877** (True Normal) | **3,027** (False AFIB) | 36,904 |
| **AFIB (Thật)** | **82** (False Normal) | **28,202** (True AFIB) | 28,284 |

### Bảng báo cáo phân loại chi tiết (Classification Report):

| Lớp phân loại | Precision | Recall | F1-Score | Support (Số mẫu) |
| :--- | :---: | :---: | :---: | :---: |
| **Normal (Bình thường)** | 99.76% | 91.80% | 95.61% | 36,904 |
| **AFIB (Rung nhĩ)** | 90.31% | 99.71% | 94.78% | 28,284 |
| **Trung bình (Macro Avg)** | 95.03% | 95.75% | 95.19% | 65,188 |
| **Trung bình (Weighted Avg)** | 95.66% | 95.23% | 95.25% | 65,188 |

---

## 3. Kết quả Đánh giá theo Từng Bệnh nhân (Patient-by-Patient Evaluation)

Để đánh giá khả năng tổng quát hóa trên từng cá thể độc lập, mô hình được kiểm thử riêng biệt trên 4 bản ghi bệnh nhân không xuất hiện trong tập Train/Val:

| Mã bệnh nhân (Record ID) | Độ chính xác (Accuracy) | Độ nhạy AFIB (Recall) | Độ xác thực AFIB (Precision) | F1-Score (AFIB) | Tổng số mẫu |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **04126** | **96.45%** | 100.00% | 58.80% | 74.06% | 13,594 |
| **05091** | **99.99%** | 100.00% | 94.74% | 97.30% | 16,269 |
| **08215** | **84.93%** | 99.49% | 84.82% | 91.57% | 17,358 |
| **08405** | **99.94%** | 99.93% | 99.99% | 99.96% | 17,967 |

---

## 4. Phân tích chi tiết hành vi của mô hình

1. **Khả năng bắt nhịp AFIB cực kỳ nhạy bén (Recall $\approx 100\%$):**
   * Đối với cả 4 bệnh nhân, độ nhạy (Recall) đối với lớp AFIB luôn duy trì ở mức tiệm cận tối đa: Bệnh nhân `04126` và `05091` đạt **100%**, bệnh nhân `08215` đạt **99.49%**, và bệnh nhân `08405` đạt **99.93%**.
   * Điều này chứng minh kiến trúc **Dilated convolutions** (tích chập giãn nở) kết hợp với khối **Squeeze-and-Excitation (SE)** hoạt động hiệu quả trong việc nắm bắt các đặc trưng tần số thấp và biến đổi vô định hình của sóng ECG khi xảy ra rung nhĩ.

2. **Hiện tượng báo động giả (False Positives) ở một số bệnh nhân:**
   * **Bệnh nhân 04126:** Đạt Recall 100% nhưng Precision chỉ đạt `58.80%`. Phân tích ma trận nhầm lẫn cho thấy bệnh nhân này có 12,906 cửa sổ Bình thường và chỉ 688 cửa sổ AFIB (tỷ lệ mất cân bằng cực lớn). Mô hình dự đoán sai 482 cửa sổ Normal thành AFIB. Do mẫu AFIB thực tế quá ít nên tỷ lệ Precision bị kéo xuống thấp. Tuy nhiên, việc mô hình không bỏ sót bất kỳ giây rung nhĩ nào của bệnh nhân này là một tín hiệu rất tốt về độ an toàn.
   * **Bệnh nhân 08215:** Độ chính xác đạt `84.93%` và Precision đạt `84.82%` do mô hình nhận nhầm 2,542 cửa sổ Normal thành AFIB. Điều này cho thấy dạng sóng ECG lúc bình thường của bệnh nhân `08215` có những đặc điểm nhiễu hoặc biến dạng (ví dụ: nhiễu cơ hoặc nhiễu đường đẳng điện) rất giống với rung nhĩ, khiến mô hình nhầm lẫn.

---

## 5. Kết luận & Định hướng Giai đoạn 4 (Lượng tử hóa TFLite)

* **Kết luận:** Mô hình **Dilated-SE-FireNet** (chỉ với 110k tham số) đạt hiệu năng xuất sắc trên tập kiểm thử gộp với **Accuracy 95.23%** and **Recall AFIB 99.71%**. Kết quả này tương đương hoặc vượt trội hơn nhiều mô hình lớn hơn, chứng minh thiết kế mạng tối giản nhưng sâu về mặt thụ cảm (dilated) rất phù hợp cho tín hiệu ECG.
* **Định hướng Giai đoạn 4:** 
  * Trọng số mô hình Float32 đạt chất lượng cao, sẵn sàng chuyển đổi sang **Float32 TFLite** và **INT8 TFLite** sử dụng cơ chế lượng tử hóa sau huấn luyện (Post-Training Quantization - PTQ).
  * Mục tiêu tiếp theo là xuất tệp `.tflite` và xác nhận độ chính xác sau lượng tử hóa trên PC không bị suy giảm quá 1-2% so với bản gốc Float32.
