# Báo cáo Giai đoạn 4: Lượng tử hóa TFLite và Kiểm chứng hiệu năng (TinyML)

## 1. Mục tiêu
Chuyển đổi mô hình **Dilated-SE-FireNet** từ định dạng Keras (.keras, 32-bit Float) sang định dạng **TensorFlow Lite (TFLite)** phục vụ các thiết bị phần cứng hạn chế tài nguyên (Edge AI). Tiến hành lượng tử hóa số nguyên 8-bit toàn phần (Full Integer Quantization) và đánh giá độ suy hao độ chính xác so với mô hình gốc.

---

## 2. Quy trình Lượng tử hóa TFLite (Post-Training Quantization)

Quá trình chuyển đổi được thực hiện bằng script [TFLite_Quantization/export_tflite.py](file:///home/pd/data/info_model/build_model/TFLite_Quantization/export_tflite.py):

1.  **Xuất bản TFLite Float32:** Chuyển đổi đồ thị sang định dạng TFLite thông thường, giữ nguyên kiểu dữ liệu số thực Float32.
2.  **Lượng tử hóa INT8 toàn phần (Full Integer Quantization):**
    *   **Bộ tối ưu hóa:** Kích hoạt chế độ `tf.lite.Optimize.DEFAULT`.
    *   **Tập dữ liệu đại diện (Representative Dataset):** Sử dụng 1000 mẫu ECG từ tập train (`X_train.npy`) chạy qua mạng để thu thập phân phối đầu ra của các lớp activation.
    *   **Kiểu dữ liệu phần cứng:** Ép kiểu toàn bộ đầu vào (Input) và đầu ra (Output) của mô hình sang kiểu số nguyên 8-bit (`tf.int8`). Điều này đảm bảo mô hình không cần bất kỳ bộ đồng xử lý số thực (FPU) nào khi chạy thực tế trên vi điều khiển.

---

## 3. Kết quả nén dung lượng mô hình

| Định dạng mô hình (Model Format) | Kích thước tệp (File Size) | Tỷ lệ nén so với Keras | Trạng thái triển khai |
| :--- | :---: | :---: | :--- |
| **Keras (.keras)** | **1.68 MB** (1,682,782 bytes) | 100.0% (Gốc) | Dùng để nghiên cứu, train trên PC/GPU |
| **TFLite Float32** | **474.78 KB** (486,172 bytes) | 28.2% (Nén ~3.5 lần) | Dùng trên điện thoại, máy tính nhúng (Raspberry Pi) |
| **TFLite INT8 (Quantized)** | **185.91 KB** (190,376 bytes) | **11.3%** (Nén ~8.8 lần) | **Tối ưu nhất cho Vi điều khiển (TinyML)** |

---

## 4. So sánh hiệu năng kiểm chứng cục bộ trên tập Test

Sử dụng script xác thực tối ưu [TFLite_Quantization/verify_tflite.py](file:///home/pd/data/info_model/build_model/TFLite_Quantization/verify_tflite.py) chạy suy luận song song (Batch Resizing) trên toàn bộ **65,188 mẫu** dữ liệu kiểm thử.

Dưới đây là so sánh chi tiết hiệu năng giữa mô hình gốc Float32 và mô hình lượng tử hóa INT8:

| Chỉ số hiệu năng (Metrics) | Mô hình Keras gốc (Float32) | Mô hình TFLite lượng tử hóa (INT8) | Sai lệch (Difference) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác (Accuracy)** | 95.23% | 94.30% | -0.93% (Rất nhỏ) |
| **Độ nhạy AFIB (Recall)** | 99.71% | **99.96%** | **+0.25%** (Tốt hơn!) |
| **Độ xác thực AFIB (Precision)**| 90.31% | 88.41% | -1.90% |
| **F1-Score (AFIB)** | 0.9478 | 0.9383 | -0.0095 |
| **Số ca AFIB bị bỏ sót (FN)** | 82 mẫu | **10 mẫu** | **Giảm 72 mẫu** (An toàn hơn) |

### Phân tích chi tiết:
1.  **Độ suy giảm Accuracy tối thiểu:** Độ chính xác tổng thể chỉ giảm **0.93%**, chứng minh việc loại bỏ lớp augmentation và tối giản đồ thị Keras ở Giai đoạn 2 đã giúp quá trình lượng tử hóa diễn ra trơn tru mà không làm méo mó các đặc trưng học được.
2.  **Độ nhạy (Recall) tăng lên:** Một điểm thú vị là mô hình lượng tử hóa INT8 có độ nhạy AFIB tăng lên **99.96%** (chỉ bỏ sót 10 mẫu so với 82 mẫu của Float32). Lượng tử hóa 8-bit làm mờ nhẹ ranh giới quyết định (decision boundary), khiến mô hình có xu hướng dự đoán nhạy hơn đối với lớp bất thường (AFIB). Trong y tế, việc tăng độ nhạy (giảm bỏ sót bệnh) là một tín hiệu cực kỳ tích cực, dù phải đánh đổi bằng việc tăng nhẹ số ca dương tính giả (Precision giảm 1.90%).

---

## 5. Cấu hình lượng tử hóa đầu vào/đầu ra cho Phần cứng
Mô hình lượng tử hóa int8 đầy đủ yêu cầu bộ tham số ánh xạ toán học khi nhúng xuống code C/C++ trên vi điều khiển:

### 5.1. Đầu vào (ECG Input Tensor)
*   **Kiểu dữ liệu:** `int8` (giá trị từ -128 đến 127).
*   **Tham số:** $Scale = 0.03504893$, $Zero-Point = -16$.
*   **Công thức chuẩn bị dữ liệu đầu vào:**
    $$x_{\text{int8}} = \text{clip}\left(\text{round}\left(\frac{x_{\text{float32}}}{0.03504893}\right) - 16, \ -128, \ 127\right)$$

### 5.2. Đầu ra (Softmax Output Tensor)
*   **Kiểu dữ liệu:** `int8`.
*   **Tham số:** $Scale = 0.00390625$, $Zero-Point = -128$.
*   **Công thức giải lượng tử lấy xác suất thực:**
    $$\text{Probability} = 0.00390625 \times (y_{\text{int8}} + 128)$$

---

## 6. Kết luận
Mô hình **Dilated-SE-FireNet INT8 TFLite** đã hoàn thiện 100%. Với kích thước siêu nhẹ **190 KB**, độ chính xác cao **94.30%** và độ nhạy AFIB gần như tuyệt đối **99.96%**, mô hình đã sẵn sàng để tích hợp vào các nền tảng vi điều khiển hỗ trợ TensorFlow Lite for Microcontrollers (TFLM) như ESP32-S3, STM32H7, STM32F4/F7 hoặc Arduino Nano 33 BLE Sense để phát hiện Rung nhĩ theo thời gian thực tại biên.
