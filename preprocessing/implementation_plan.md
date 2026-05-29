# Implementation Plan - Phase 1: Database Preprocessing

Mục tiêu của giai đoạn này là xây dựng hoàn chỉnh pipeline tiền xử lý dữ liệu từ cơ sở dữ liệu thô **MIT-BIH Atrial Fibrillation Database (AFDB)** để tạo ra các tập dữ liệu `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` lưu dưới dạng file `.npy`. 

Công việc sẽ được thực hiện và kiểm thử từng bước trong notebook [pipeline.ipynb](file:///home/pd/data/info_model/build_model/preprocessing/pipeline.ipynb).

## User Review Required

> [!IMPORTANT]
> **1. Quy tắc chia tập bệnh nhân (Patient-wise Split):**
> Để tránh rò rỉ dữ liệu, chúng ta phân chia cứng các Record (bệnh nhân) như sau:
> * **Train (16 records):** `04043`, `04936`, `07162`, `07859`, `07879`, `08455`, `06426`, `05121`, `06995`, `05261`, `06453`, `04015`, `04908`, `04048`, `08434`, `08378`
> * **Val (3 records):** `04746`, `07910`, `08219`
> * **Test (4 records):** `04126`, `05091`, `08215`, `08405` (các record còn lại trong thư mục trừ `00735` và `03665` bị thiếu file tín hiệu `.dat`).
>
> **2. Thuật toán phát hiện đỉnh QRS và Hậu kiểm (Post-validation):**
> * Chạy phát hiện QRS toàn cục bằng `wfdb.processing.xqrs_detect` trên từng record trước khi cắt cửa sổ.
> * Cửa sổ Normal được lọc bằng điều kiện biến thiên RR (CV < 0.2) để tránh lẫn các đoạn ngoại tâm thu hoặc nhiễu nặng.
> * Cửa sổ AFIB được xác nhận bằng việc có tối thiểu 5 đỉnh QRS để tránh nhiễu mất kết nối điện cực (flatline).

## Open Questions

Không có câu hỏi mở ở giai đoạn này.

## Proposed Changes

Chúng ta sẽ thực hiện viết mã nguồn trong notebook [pipeline.ipynb](file:///home/pd/data/info_model/build_model/preprocessing/pipeline.ipynb) theo các phần chính sau:

### Preprocessing Component

#### [MODIFY] [pipeline.ipynb](file:///home/pd/data/info_model/build_model/preprocessing/pipeline.ipynb)
* **Khởi tạo và cấu hình:** Định nghĩa các đường dẫn thư mục, tham số ECG (tần số $f_s = 250$ Hz, độ dài cửa sổ $2500$ mẫu, độ dịch $500$ mẫu).
* **Bước 1: Đọc tín hiệu và nhãn nhịp thô:** Viết hàm đọc một record sử dụng `wfdb.rdrecord` và `wfdb.rdann`. Chỉ lấy tín hiệu kênh 0.
* **Bước 2: Xây dựng nhãn liên tục:** Sửa lỗi gán nhãn bằng cách chỉ trích xuất các nhãn đổi nhịp bắt đầu với dấu ngoặc `(` (ví dụ `(N`, `(AFIB`), sau đó gán nhãn cho toàn bộ độ dài tín hiệu từ mốc thời gian này đến mốc thời gian tiếp theo.
* **Bước 3: Lọc nhiễu thông dải (Bandpass Filter):** Sử dụng bộ lọc Butterworth 4th-order (0.5Hz - 40Hz) với hàm `scipy.signal.sosfilt`.
* **Bước 4: Phát hiện đỉnh QRS toàn cục:** Chạy phát hiện đỉnh QRS một lần trên toàn bộ tín hiệu để phục vụ kiểm tra chất lượng cửa sổ.
* **Bước 5: Cắt cửa sổ và Chuẩn hóa:** Duyệt qua tín hiệu để cắt các cửa sổ 10s chồng lấp 80%, chuẩn hóa Z-score, hậu kiểm chất lượng và gom dữ liệu.
* **Bước 6: Gộp và Lưu trữ dữ liệu:** Lưu kết quả thành các tệp `.npy` trong thư mục [database/processed](file:///home/pd/data/info_model/build_model/database/processed).

## Verification Plan

### Automated Tests
Chúng ta sẽ viết code kiểm thử trực quan (Visual verification) ngay trong notebook:
* Vẽ đồ thị tín hiệu ECG thô so sánh với tín hiệu sau khi lọc nhiễu để đánh giá bộ lọc.
* Vẽ đồ thị tín hiệu ECG kèm theo các đỉnh QRS phát hiện được và nhãn tương ứng trên một đoạn ngắn để kiểm tra độ chính xác của nhãn liên tục và thuật toán QRS.
* Kiểm tra `shape` và sự phân bổ nhãn (class distribution) của dữ liệu đầu ra sau khi chạy thử nghiệm trên 1 record.
