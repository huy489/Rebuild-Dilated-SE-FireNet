# Walkthrough - Toàn bộ các Giai đoạn Dự án (Giai đoạn 1 - 4)

Dự án phát hiện rung nhĩ từ tín hiệu ECG 1 kênh đã hoàn thành thành công và trọn vẹn toàn bộ 4 giai đoạn, từ tiền xử lý dữ liệu thô, thiết kế cấu trúc mạng, huấn luyện trên GPU đám mây, cho đến lượng tử hóa TFLite và xác thực hiệu năng phần cứng trên PC.

---

## Giai đoạn 1: Tiền xử lý dữ liệu (Database Preprocessing)

Chúng ta đã xây dựng hoàn chỉnh và chạy kiểm thử pipeline tiền xử lý dữ liệu ECG cho toàn bộ cơ sở dữ liệu **MIT-BIH Atrial Fibrillation Database (AFDB)**.

### 1. Tập dữ liệu đầu ra (.npy)
Các tệp dữ liệu đã được lưu thành công tại thư mục [database/processed](file:///home/pd/data/info_model/build_model/database/processed):

* **Tập huấn luyện & Đánh giá chung:**
  * **Train Split:** `(250734, 2500, 1)` (142,761 mẫu Normal, 107,973 mẫu AFIB)
  * **Val Split:** `(43161, 2500, 1)` (26,626 mẫu Normal, 16,535 mẫu AFIB)
  * **Test Split gộp:** `(65188, 2500, 1)` (36,904 mẫu Normal, 28,284 mẫu AFIB)
* **Các tệp kiểm thử bệnh nhân độc lập (Test folder):**
  * `X_04126.npy` & `y_04126.npy` (13,594 mẫu)
  * `X_05091.npy` & `y_05091.npy` (16,269 mẫu)
  * `X_08215.npy` & `y_08215.npy` (17,358 mẫu)
  * `X_08405.npy` & `y_08405.npy` (17,967 mẫu)

### 2. Minh chứng trực quan (Visual Verification)
Biểu đồ kiểm tra trực quan cho bản ghi `08405` chứng minh bộ lọc thông dải hoạt động tốt, nhãn nhịp liên tục khớp chính xác và đỉnh QRS được phát hiện thành công:

![Verification Plot](/home/pd/.gemini/antigravity/brain/8cc5e8b2-22b3-4ec6-bf54-8c68d08055a1/08405_verification_plot.png)

---

## Giai đoạn 2: Xây dựng và Xác thực mô hình (Model Verification)

Chúng ta đã hoàn thành việc thiết lập cấu trúc mô hình modular trong thư mục [src/](file:///home/pd/data/info_model/build_model/src).
* **Số lượng tham số:** **110,694 tham số** (~432 KB) - cực kỳ gọn nhẹ và thích hợp cho việc nhúng xuống vi điều khiển (TinyML).
* **Đầu vào (Input Contract):** `(None, 2500, 1)`
* **Đầu ra (Output Contract):** `(None, 2)` (xác suất softmax của 2 lớp Normal và AFIB)

---

## Giai đoạn 3: Huấn luyện chính thức và Đánh giá chi tiết (Float32 Model)

Chúng ta đã tiến hành huấn luyện chính thức mô hình trên Google Colab sử dụng GPU và chạy đánh giá chi tiết trên tập test cục bộ.

### 1. Kết quả Huấn luyện trên Google Colab
Huấn luyện được thực hiện qua file [train_colab.ipynb](file:///home/pd/data/info_model/build_model/train_on_colab/train_colab.ipynb):
* **Cơ chế Dừng Sớm (Early Stopping):** 
  * Mô hình đạt kết quả Validation tốt nhất ở **Epoch 16** với `val_loss: 0.2487` và `val_accuracy: 99.21%`.
  * Vì `val_loss` không giảm thêm trong 10 epoch tiếp theo (Epoch 17-26), callback `EarlyStopping` đã kích hoạt dừng huấn luyện ở Epoch 26 để chống overfitting. Trọng số tối ưu nhất ở Epoch 16 đã được khôi phục thành công.

### 2. Đánh giá chi tiết mô hình Float32 (Cục bộ)
Chạy script đánh giá [src/evaluate.py](file:///home/pd/data/info_model/build_model/src/evaluate.py) trên tập kiểm thử gộp (`65,188` mẫu) thu được kết quả:
* **Độ chính xác (Accuracy):** **`95.23%`**
* **Độ nhạy AFIB (Recall):** **`99.71%`** (Chỉ bỏ sót 82 trong số 28,284 cửa sổ AFIB).
* **Độ xác thực AFIB (Precision):** **`90.31%`** (3,027 mẫu Normal bị nhận nhầm thành AFIB).
* **F1-Score (AFIB):** **`0.9478`**

### 3. Ma trận nhầm lẫn (Confusion Matrices)

````carousel
![Overall Confusion Matrix](/home/pd/.gemini/antigravity/brain/8cc5e8b2-22b3-4ec6-bf54-8c68d08055a1/confusion_matrix_overall.png)
<!-- slide -->
![Patient 04126 Confusion Matrix](/home/pd/.gemini/antigravity/brain/8cc5e8b2-22b3-4ec6-bf54-8c68d08055a1/confusion_matrix_04126.png)
<!-- slide -->
![Patient 05091 Confusion Matrix](/home/pd/.gemini/antigravity/brain/8cc5e8b2-22b3-4ec6-bf54-8c68d08055a1/confusion_matrix_05091.png)
<!-- slide -->
![Patient 08215 Confusion Matrix](/home/pd/.gemini/antigravity/brain/8cc5e8b2-22b3-4ec6-bf54-8c68d08055a1/confusion_matrix_08215.png)
<!-- slide -->
![Patient 08405 Confusion Matrix](/home/pd/.gemini/antigravity/brain/8cc5e8b2-22b3-4ec6-bf54-8c68d08055a1/confusion_matrix_08405.png)
````

---

## Giai đoạn 4: Lượng tử hóa TFLite và Xác thực (Float32 & INT8)

Chúng ta đã tiến hành chuyển đổi mô hình Keras sang định dạng TFLite và chạy xác thực độ chính xác của mô hình lượng tử hóa int8 đầy đủ (Full Integer Quantization) trên toàn bộ tập kiểm thử.

### 1. Xuất mô hình TFLite
Chạy script [TFLite_Quantization/export_tflite.py](file:///home/pd/data/info_model/build_model/TFLite_Quantization/export_tflite.py) thành công và lưu trữ tại [outputs/tflite/](file:///home/pd/data/info_model/build_model/outputs/tflite/):
* **Tệp Float32 TFLite:** `dilated_se_firenet_float32.tflite` | Kích thước: **474.78 KB**
* **Tệp INT8 Quantized TFLite:** `dilated_se_firenet_int8.tflite` | Kích thước: **185.91 KB** (Giảm tới **88.7%** so với tệp Keras ban đầu là 1.68 MB).

### 2. So sánh hiệu năng giữa mô hình Keras Float32 và TFLite INT8
Chúng ta đã kiểm tra mô hình TFLite INT8 trên toàn bộ tập kiểm thử gộp **65,188 mẫu** bằng script tối ưu [TFLite_Quantization/verify_tflite.py](file:///home/pd/data/info_model/build_model/TFLite_Quantization/verify_tflite.py) (sử dụng kỹ thuật Batch Resizing để chạy cực nhanh trên CPU).

Dưới đây là bảng so sánh chi tiết hiệu năng giữa mô hình gốc Float32 và mô hình lượng tử hóa INT8:

| Chỉ số hiệu năng (Metrics) | Mô hình Keras gốc (Float32) | Mô hình TFLite lượng tử hóa (INT8) | Độ lệch (Difference) |
| :--- | :---: | :---: | :---: |
| **Kích thước tệp (File Size)** | ~1.68 MB (Keras) | **190 KB** (TFLite) | **-88.7%** (Nén ~8.8 lần) |
| **Độ chính xác (Accuracy)** | 95.23% | 94.30% | -0.93% |
| **Độ nhạy AFIB (Recall)** | 99.71% | **99.96%** | **+0.25%** (Tốt hơn!) |
| **Độ xác thực AFIB (Precision)**| 90.31% | 88.41% | -1.90% |
| **F1-Score (AFIB)** | 0.9478 | 0.9383 | -0.0095 |
| **Số ca AFIB bị bỏ sót (FN)** | 82 mẫu | **10 mẫu** | **Giảm 72 mẫu** (An toàn hơn) |

### 3. Nhận xét & Kết luận
* **Suy hao hiệu năng tối thiểu:** Lượng tử hóa int8 đầy đủ chỉ làm giảm **0.93%** độ chính xác tổng thể, hoàn toàn nằm trong mục tiêu sai số cho phép (< 1-2%).
* **Độ an toàn tăng cao:** Tỷ lệ bỏ sót cơn Rung nhĩ (False Negatives) của mô hình TFLite INT8 giảm từ 82 mẫu xuống còn **10 mẫu** (tương ứng Recall AFIB tăng lên **99.96%**). Điều này cực kỳ có lợi trong các thiết bị đeo theo dõi sức khỏe liên tục, nơi an toàn chẩn đoán được ưu tiên hàng đầu.
* **Sẵn sàng triển khai TinyML:** Với kích thước tệp chỉ **190 KB** và việc ép kiểu toàn bộ đầu vào/đầu ra sang `int8`, mô hình đã sẵn sàng 100% để triển khai trực tiếp lên các dòng vi điều khiển hỗ trợ TensorFlow Lite for Microcontrollers (nhũ STM32, ESP32, Cortex-M4/M7, v.v.).
