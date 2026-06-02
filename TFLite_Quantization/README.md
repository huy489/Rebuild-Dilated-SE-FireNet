# Hướng dẫn Lượng tử hóa Mô hình TFLite (Float32 & Full Integer INT8)

Thư mục này chứa toàn bộ các công cụ và mã nguồn phục vụ việc chuyển đổi và lượng tử hóa mô hình **Dilated-SE-FireNet** sang định dạng **TensorFlow Lite (TFLite)** phục vụ triển khai trên thiết bị Edge / Vi điều khiển (TinyML).

---

## 1. Tổng quan về Lượng tử hóa (Quantization)

Lượng tử hóa là quá trình chuyển đổi các trọng số (weights) và các hàm kích hoạt (activations) của mô hình từ kiểu dữ liệu số thực chính xác đơn (Float32, 32-bit) sang kiểu dữ liệu số nguyên có độ chính xác thấp hơn (INT8, 8-bit).

### Lợi ích:
*   **Giảm dung lượng mô hình:** Từ **1.68 MB** (.keras) giảm xuống chỉ còn **190 KB** (.tflite INT8) - nén ~8.8 lần.
*   **Tăng tốc độ suy luận (Inference Speed):** Phép tính nhân chập số nguyên (Integer Arithmetic) trên CPU/NPU nhanh hơn nhiều so với số thực Float32.
*   **Giảm tiêu thụ năng lượng:** Phép tính 8-bit tiêu thụ ít băng thông bộ nhớ và năng lượng hơn, rất quan trọng trên các thiết bị chạy pin.
*   **Tương thích phần cứng:** Nhiều vi điều khiển (như dòng Cortex-M sử dụng thư viện CMSIS-NN) hoặc Edge TPU chỉ hỗ trợ tính toán số nguyên INT8.

---

## 2. Cơ sở Toán học của Lượng tử hóa sau Huấn luyện (Post-Training Quantization)

TensorFlow Lite sử dụng cơ chế lượng tử hóa tuyến tính không đối xứng (Asymmetric Linear Quantization) để ánh xạ dải số thực sang số nguyên 8-bit.

### 2.1. Công thức Lượng tử hóa (Quantization Formula)
Để ánh xạ một giá trị thực $r \in [r_{\min}, r_{\max}]$ sang giá trị nguyên $q \in [q_{\min}, q_{\max}]$:

$$q = \text{clip}\left( \text{round}\left( \frac{r}{S} \right) + Z, \ q_{\min}, \ q_{\max} \right)$$

Trong đó:
*   $r$ (Real value): Giá trị số thực Float32 đầu vào.
*   $q$ (Quantized value): Giá trị số nguyên INT8 đầu ra sau lượng tử.
*   $S$ (Scale): Tham số tỉ lệ (kiểu số thực Float32), định nghĩa khoảng giá trị thực tương ứng với 1 đơn vị số nguyên.
*   $Z$ (Zero-Point): Điểm không (kiểu số nguyên), ánh xạ giá trị thực $0.0$ vào miền số nguyên. Điều này đảm bảo giá trị thực $0.0$ được biểu diễn chính xác mà không bị lệch (rất quan trọng cho các phép đệm zero-padding).
*   $\text{clip}(x, a, b)$: Hàm giới hạn giá trị để chống tràn số (overflow/underflow):
    $$\text{clip}(x, a, b) = \max(a, \min(x, b))$$
*   Đối với kiểu dữ liệu **INT8 có dấu (signed)**: $q_{\min} = -128$ và $q_{\max} = 127$.
*   Đối với kiểu dữ liệu **UINT8 không dấu (unsigned)**: $q_{\min} = 0$ và $q_{\max} = 255$.

### 2.2. Tính toán tham số $S$ (Scale) và $Z$ (Zero-Point)
Các tham số lượng tử hóa của mỗi lớp được xác định từ dải động số thực $[r_{\min}, r_{\max}]$ thu được trong quá trình chạy dữ liệu:

*   **Công thức tính Scale ($S$):**
    $$S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$$

*   **Công thức tính Zero-Point ($Z$):**
    $$Z = \text{round}\left( \frac{-r_{\min}}{S} \right) + q_{\min}$$

### 2.3. Giải lượng tử hóa (Dequantization)
Khi xuất kết quả dự đoán ra ngoài, ta giải lượng tử hóa từ số nguyên $q$ sang số thực $\hat{r}$ để lấy phân phối xác suất lớp:

$$\hat{r} = S \times (q - Z)$$

---

## 3. Vai trò của Tập dữ liệu Đại diện (Representative Dataset)

Trong mô hình mạng nơ-ron:
1.  **Trọng số (Weights):** Đã cố định sau huấn luyện, nên có thể dễ dàng tìm được $r_{\min}, r_{\max}$ tĩnh của chúng để lượng tử hóa trực tiếp.
2.  **Hàm kích hoạt (Activations):** Giá trị thay đổi liên tục tùy thuộc vào dữ liệu đầu vào (dynamic range). Do đó, ta không thể biết trước $r_{\min}, r_{\max}$ của các lớp trung gian.

Để giải quyết vấn đề này, TensorFlow Lite cần một **Tập dữ liệu đại diện (Representative Dataset)**:
*   Chúng ta cung cấp khoảng 1000 mẫu ECG từ tập Train (`X_train.npy`).
*   TFLite Converter sẽ chạy thử nghiệm (calibration) các mẫu này qua mô hình để thống kê phân phối đầu ra của các lớp kích hoạt trung gian, từ đó tính toán chính xác bộ tham số $(S, Z)$ cho từng lớp kích hoạt.
*   Cơ chế này được định nghĩa qua hàm `representative_dataset_gen()` trong file `export_tflite.py`.

---

## 4. Làm rõ các tham số thực tế thu được trong Mô hình của chúng ta

Khi kiểm tra mô hình `dilated_se_firenet_int8.tflite` bằng script xác thực, chúng ta thu được các chi tiết lượng tử hóa thực tế:

### 4.1. Đầu vào (Input Tensor Detail)
*   **Tên:** `serving_default_Input_ECG:0`
*   **Kiểu dữ liệu:** `int8` (yêu cầu dữ liệu đưa vào vi điều khiển phải là số nguyên 8-bit).
*   **Tham số lượng tử:** `Scale = 0.03504893`, `Zero-Point = -16`.
*   **Ý nghĩa:** Khi có tín hiệu ECG thực tế ở dạng số thực $x_{\text{float}}$, thiết bị ngoại vi cần lượng tử hóa dữ liệu trước khi đẩy vào mô hình:
    $$x_{\text{int8}} = \text{clip}\left(\text{round}\left(\frac{x_{\text{float}}}{0.03504893}\right) - 16, \ -128, \ 127\right)$$

### 4.2. Đầu ra (Output Tensor Detail)
*   **Tên:** `StatefulPartitionedCall_1:0`
*   **Kiểu dữ liệu:** `int8` (mô hình trả về điểm số nguyên).
*   **Tham số lượng tử:** `Scale = 0.00390625` (tương đương $\frac{1}{256}$), `Zero-Point = -128`.
*   **Ý nghĩa:** Điểm số nguyên đầu ra $y_{\text{int8}}$ sau đó được chuyển đổi ngược về dạng xác suất thực từ $0.0$ đến $1.0$:
    $$\text{Probability} = 0.00390625 \times (y_{\text{int8}} - (-128))$$

---

## 5. Cấu trúc thư mục

```directory
TFLite_Quantization/
├── export_tflite.py  # Script xuất mô hình Keras sang TFLite (Float32 & INT8)
├── verify_tflite.py  # Script đánh giá kiểm thử mô hình TFLite INT8 trên tập test
└── README.md         # Tài liệu hướng dẫn (tệp này)
```

---

## 6. Hướng dẫn Triển khai lượng tử hóa

Mọi thao tác đều được chạy từ thư mục gốc của dự án bằng môi trường ảo `.venv`.

### Bước 1: Xuất mô hình TFLite (Float32 & INT8)
Chạy script `export_tflite.py` để thực hiện lượng tử hóa tĩnh trọng số và thu thập dải động kích hoạt từ tập train:

```bash
.venv/bin/python3 -m TFLite_Quantization.export_tflite
```

*   **Đầu ra:** 2 tệp mô hình sẽ được sinh ra tại `outputs/tflite/`:
    *   `dilated_se_firenet_float32.tflite` (~475 KB)
    *   `dilated_se_firenet_int8.tflite` (~190 KB)

### Bước 2: Xác thực hiệu năng mô hình INT8 trên PC
Để kiểm tra độ suy hao độ chính xác sau lượng tử hóa, chạy script `verify_tflite.py` trên toàn bộ tập dữ liệu test gộp (`65,188` mẫu). Script này đã được tối ưu hóa bằng thuật toán **Batch Resizing** để chạy suy luận song song cực nhanh trên CPU:

```bash
.venv/bin/python3 -m TFLite_Quantization.verify_tflite
```

*   **Tùy chọn giới hạn số mẫu (chạy nhanh để test):**
    ```bash
    .venv/bin/python3 -m TFLite_Quantization.verify_tflite --max_samples 10000
    ```
*   **Đầu ra:** Báo cáo đánh giá chi tiết được hiển thị trên terminal và lưu trữ tự động tại `outputs/reports/evaluation_tflite.json` để so sánh trực quan với mô hình Float32 gốc.
