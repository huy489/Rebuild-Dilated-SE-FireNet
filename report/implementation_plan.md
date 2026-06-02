# Implementation Plan - Phase 4: TFLite Export & Verification (Float32 and INT8)

Mục tiêu của giai đoạn này là xuất mô hình **Dilated-SE-FireNet** sang định dạng **TFLite** (cả hai phiên bản: Float32 và lượng tử hóa INT8 đầy đủ) và thực hiện đánh giá kiểm thử mô hình INT8 trên PC để xác nhận độ chính xác.

## User Review Required

> [!IMPORTANT]
> **Lượng tử hóa INT8 đầy đủ (Full Integer Quantization):**
> * Để lượng tử hóa INT8, chúng ta cần một tập dữ liệu đại diện (Representative Dataset) khoảng 1000 mẫu lấy từ tập huấn luyện (`X_train.npy`). Điều này giúp TFLite Converter tính toán được dải động (Dynamic Range) của các lớp Activation để ánh xạ chính xác sang kiểu số nguyên 8-bit.
> * Cả đầu vào (Input) và đầu ra (Output) của mô hình TFLite INT8 sẽ được cấu hình ở kiểu dữ liệu `int8` (hoặc `uint8`), phù hợp hoàn hảo với các phần cứng vi điều khiển (TinyML) và bộ tăng tốc NPU/TPU Edge.

## Open Questions

Không có câu hỏi mở.

## Proposed Changes

Chúng ta sẽ tạo mới các tệp sau trong thư mục [src/](file:///home/pd/data/info_model/build_model/src):

---

### TFLite Components

#### [NEW] [export_tflite.py](file:///home/pd/data/info_model/build_model/src/export_tflite.py)
* **Chức năng:** Nạp mô hình `.keras` tốt nhất và xuất ra định dạng TFLite.
* **Các phương thức:**
  * `export_float32(model, output_path)`: Xuất mô hình TFLite float32 tiêu chuẩn.
  * `export_int8(model, X_rep, output_path)`: Áp dụng Full Integer Quantization bằng cách sử dụng `X_train` làm Representative Dataset. Cấu hình kiểu dữ liệu đầu vào/đầu ra là `int8`.
* **Đường dẫn đầu ra:** 
  * `outputs/tflite/dilated_se_firenet_float32.tflite`
  * `outputs/tflite/dilated_se_firenet_int8.tflite`

#### [NEW] [verify_tflite.py](file:///home/pd/data/info_model/build_model/src/verify_tflite.py)
* **Chức năng:** Nạp mô hình TFLite và chạy dự đoán trên tập dữ liệu kiểm thử.
* **Các phương thức:**
  * `quantize_input(x_float, input_detail)`: Thủ công lượng tử hóa dữ liệu đầu vào (từ float32 sang int8 sử dụng scale và zero_point của mô hình TFLite).
  * `dequantize_output(y_raw, output_detail)`: Giải lượng tử hóa kết quả đầu ra (từ int8 sang float32).
  * `run_tflite(tflite_path, X, max_samples)`: Khởi tạo TFLite Interpreter, đẩy dữ liệu qua mô hình và thu thập kết quả dự đoán.
* **Đầu ra báo cáo:** Lưu các chỉ số đánh giá của mô hình TFLite INT8 vào [outputs/reports/evaluation_tflite.json](file:///home/pd/data/info_model/build_model/outputs/reports/evaluation_tflite.json).

---

## Verification Plan

### Automated Tests
* Chạy xuất mô hình:
  ```bash
  .venv/bin/python3 -m src.export_tflite
  ```
* Xác nhận hai tệp `.tflite` được tạo ra trong `outputs/tflite/`.
* Chạy kiểm tra hiệu năng mô hình INT8 trên tập test:
  ```bash
  .venv/bin/python3 -m src.verify_tflite --max_samples 5000
  ```
  *(Sử dụng tùy chọn `--max_samples 5000` để chạy nhanh hơn khi kiểm tra, hoặc chạy toàn bộ tập test để so sánh trực tiếp).*
* So sánh độ chính xác (Accuracy, F1-Score) của mô hình TFLite INT8 với mô hình Float32 Keras để đảm bảo sai số lượng tử hóa là tối thiểu (thường < 1-2%).
