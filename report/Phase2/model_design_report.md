# Báo cáo Giai đoạn 2: Thiết kế và Xác thực Kiến trúc mạng Dilated-SE-FireNet

## 1. Mục tiêu
Thiết kế một kiến trúc mạng nơ-ron tích chập (CNN) siêu nhẹ, hiệu năng cao, tối ưu cho việc nhận diện Rung nhĩ (AFIB) từ tín hiệu ECG 1 kênh dài 10 giây. Mô hình cần có số lượng tham số nhỏ (< 150k) và không chứa các toán tử phức tạp để dễ dàng chuyển đổi sang TFLite lượng tử hóa INT8 phục vụ TinyML.

---

## 2. Kiến trúc mạng Dilated-SE-FireNet

Kiến trúc mạng được thiết kế theo mô hình phân cấp, kết hợp giữa mạng SqueezeNet (nhẹ, ít tham số), tích chập giãn nở (Dilated Convolution - tăng trường nhìn thời gian) và cơ chế chú ý kênh (Squeeze-and-Excitation - tập trung kênh đặc trưng quan trọng).

### 2.1. Sơ đồ khối tổng quát

Mô hình nhận đầu vào kích thước $(2500, 1)$ và đi qua các khối xử lý tuần tự:

```text
ECG Input (2500, 1)
       │
  [Stem Block] (Conv1D k=7, filters=12, stride=2, padding=same) ──> Output: (1250, 12)
       │
[Dilated-SE-Fire Block 1] (dilation=1, squeeze=12, expand=24, output_channels=48)
       │
[Learnable Downsample 1] (Conv1D k=3, stride=2, padding=same) ──> Output: (625, 48)
       │
[Dilated-SE-Fire Block 2] (dilation=2, squeeze=16, expand=32, output_channels=64)
       │
[Learnable Downsample 2] (Conv1D k=3, stride=2, padding=same) ──> Output: (313, 64)
       │
[Dilated-SE-Fire Block 3] (dilation=4, squeeze=24, expand=48, output_channels=96)
       │
[Learnable Downsample 3] (Conv1D k=3, stride=2, padding=same) ──> Output: (157, 96)
       │
[Dilated-SE-Fire Block 4] (dilation=8, squeeze=32, expand=64, output_channels=128)
       │
[Global Average Pooling 1D] ──> Output: (128)
       │
[Classifier] (Dropout 0.2 ──> Dense Softmax) ──> Output: (2) [Normal, AFIB]
```

---

## 3. Các thành phần cải tiến lõi

### 3.1. Khối Dilated Fire Module với Cơ chế chú ý SE
Khối này cải tiến từ khối Fire Module gốc của SqueezeNet bằng cách tích hợp tích chập giãn nở và khối chú ý Squeeze-and-Excitation:

1.  **Squeeze Layer:** Sử dụng tích chập 1D kích thước kernel $1 \times 1$ để giảm số lượng kênh (giảm chi phí tính toán).
2.  **Expand Layer:** Chia làm 2 nhánh song song:
    *   **Nhánh 1x1 Conv:** Học các tổ hợp đặc trưng kênh cục bộ.
    *   **Nhánh 3x3 Conv Dilated:** Sử dụng tích chập kích thước $3 \times 3$ kết hợp với hệ số giãn nở (dilation rate) $d \in \{1, 2, 4, 8\}$ tăng dần qua các block. Tích chập giãn nở giúp mở rộng trường cảm thụ (receptive field) theo chiều thời gian để mô hình "nhìn" được các khoảng cách R-R xa hơn mà không làm tăng số lượng trọng số (trọng số lọc chỉ đặt cách nhau $d-1$ khoảng trắng).
3.  **Concatenate:** Ghép hai nhánh đầu ra theo chiều kênh đặc trưng.
4.  **Squeeze-and-Excitation (SE) Block:**
    *   **Squeeze:** Ép chiều thời gian bằng Global Average Pooling về véc-tơ $(1, C)$.
    *   **Excitation:** Đi qua 2 lớp Dense giảm kênh 4 lần rồi khôi phục lại, kích hoạt bằng hàm Sigmoid để thu được bộ trọng số biểu thị tầm quan trọng của từng kênh.
    *   **Scale:** Nhân trọng số này lại với ma trận đặc trưng ban đầu để lọc nhiễu đường truyền.

### 3.2. Học giảm mẫu (Learnable Downsampling)
Thay vì dùng Max Pooling (làm mất thông tin pha thời gian của ECG), chúng ta sử dụng **Tích chập 1D với Stride=2** để mô hình tự học cách giảm phân giải thời gian hiệu quả nhất.

### 3.3. Kết nối tắt Residual Connection
Mỗi khối đều có nhánh kết nối tắt (Skip Connection) cộng trực tiếp đầu vào vào đầu ra. Khi số kênh đầu vào và đầu ra khác nhau, đầu vào được đi qua một lớp Conv 1D 1x1 để khớp số kênh, giúp tránh hiện tượng triệt tiêu gradient khi mô hình sâu hơn.

---

## 4. Loại bỏ các lớp Augmentation khỏi Đồ thị mô hình

*   **Vấn đề:** Ban đầu, các phép biến đổi dữ liệu (thêm nhiễu, co giãn biên độ) được định nghĩa dưới dạng một lớp Keras tùy chỉnh nằm trong đồ thị mô hình. Khi convert sang TFLite lượng tử hóa INT8, các phép tính ngẫu nhiên này không thể lượng tử hóa được và gây lỗi phần cứng nghiêm trọng.
*   **Giải pháp:** Di chuyển toàn bộ quá trình Augmentation dữ liệu ra ngoài mô hình, thực hiện trực tiếp trên CPU thông qua pipeline `tf.data.Dataset.map(augment_ecg)` trước khi nạp vào mô hình. Đồ thị mạng Keras chỉ nhận đầu vào chuẩn và xuất ra xác suất Softmax, đảm bảo tương thích 100% với TFLite.

---

## 5. Xác thực đồ thị mô hình (Model Verification)

Script kiểm tra [src/check_model.py](file:///home/pd/data/info_model/build_model/src/check_model.py) được xây dựng để xác minh:
*   Mô hình khởi tạo thành công với tổng số tham số là **110,694** (~432 KB).
*   Độ phân giải các chiều (Shape tensor) giảm chính xác qua các block và kết thúc ở kích thước đầu ra `(None, 2)`.
*   Chạy thử dữ liệu giả lập (dummy data) và dữ liệu thật từ tập test thành công, đầu ra Softmax chuẩn (tổng xác suất 2 lớp luôn bằng `1.0`), không chứa các giá trị lỗi `NaN`/`Inf`.
*   Mô hình sẵn sàng đưa vào pipeline huấn luyện chính thức.
