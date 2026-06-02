# Dự án Tái xây dựng & Lượng tử hóa Mô hình Dilated-SE-FireNet (Phát hiện Rung nhĩ ECG 1 kênh)

Dự án này tập trung vào việc xây dựng lại, huấn luyện GPU đám mây, đánh giá chi tiết và lượng tử hóa mô hình **Dilated-SE-FireNet** chuyên dụng để phát hiện Rung nhĩ (AFIB) từ tín hiệu điện tâm đồ (ECG) 1 kênh dài 10 giây (tần số lấy mẫu 250 Hz). 

Mô hình được thiết kế theo phong cách TinyML cực kỳ gọn nhẹ (chỉ **110,694 tham số**, kích thước tệp **190 KB** sau lượng tử hóa) để có thể triển khai trực tiếp lên các dòng vi điều khiển thu thập tín hiệu tại biên.

---

## 1. Cấu trúc thư mục dự án

```directory
build_model/
├── TFLite_Quantization/    # Giai đoạn 4: Lượng tử hóa TFLite & Xác thực
│   ├── export_tflite.py    # Script xuất mô hình Float32 và lượng tử hóa INT8
│   ├── verify_tflite.py    # Script kiểm chứng hiệu năng TFLite (Batch Resizing)
│   └── README.md           # Hướng dẫn chi tiết lý thuyết và toán học lượng tử hóa
├── database/               # Thư mục chứa cơ sở dữ liệu ECG (AFDB)
│   └── processed/          # Dữ liệu sạch .npy sau tiền xử lý (Train, Val, Test)
├── outputs/                # Thư mục chứa kết quả đầu ra
│   ├── checkpoints/        # Tệp trọng số mô hình (.keras) tốt nhất và cuối cùng
│   ├── reports/            # Biểu đồ ma trận nhầm lẫn và các báo cáo đánh giá JSON
│   └── tflite/             # Tệp mô hình lượng tử hóa .tflite (Float32 & INT8)
├── preprocessing/          # Giai đoạn 1: Tiền xử lý tín hiệu ECG
│   ├── pipeline.ipynb      # Notebook thử nghiệm lọc nhiễu và trích xuất đỉnh QRS
│   ├── run_batch.py        # Tiền xử lý toàn bộ dữ liệu AFDB sang tệp .npy
│   └── README.md           # Hướng dẫn chi tiết quy trình tiền xử lý tín hiệu ECG
├── report/                 # Báo cáo tổng kết dự án theo từng giai đoạn
│   ├── Phase1/             # Báo cáo Giai đoạn 1 (Tiền xử lý)
│   ├── Phase2/             # Báo cáo Giai đoạn 2 (Kiến trúc mô hình)
│   ├── Phase3/             # Báo cáo Giai đoạn 3 (Huấn luyện & Đánh giá Float32)
│   ├── Phase4/             # Báo cáo Giai đoạn 4 (Lượng tử hóa & TinyML)
│   ├── implementation_plan.md # Bản kế hoạch triển khai gốc
│   └── walkthrough.md      # Tài liệu tổng kết dự án chi tiết
├── src/                    # Thư mục mã nguồn chính của mô hình (Giai đoạn 2 & 3)
│   ├── config.py           # Siêu tham số mô hình và cấu hình đường dẫn
│   ├── utils.py            # Hàm tiện ích (load dữ liệu, khóa seed, tạo thư mục)
│   ├── model.py            # Định nghĩa đồ thị mạng Dilated-SE-FireNet
│   ├── check_model.py      # Kiểm thử tính đúng đắn của đồ thị mô hình
│   ├── train.py            # Pipeline huấn luyện chính thức (hỗ trợ LR decay, augmentation)
│   ├── evaluate.py         # Đánh giá chi tiết mô hình Float32 trên tập Test
│   └── README.md           # Tài liệu kỹ thuật chi tiết cấu trúc mô hình
├── train_on_colab/         # Thư mục cô lập luồng huấn luyện GPU đám mây
│   ├── train_colab.ipynb   # Jupyter Notebook chạy huấn luyện trên Google Colab GPU
│   ├── upload_to_drive.py  # Script upload dữ liệu lên Google Drive phục vụ Colab
│   └── dilated-se-fire-*.json # Khóa Service Account liên kết Drive (đã được bảo mật)
├── requirements.txt        # Các thư viện phụ thuộc của dự án
└── Dilated_SE_FireNet_TensorFlow_Project.md # Tài liệu đặc tả yêu cầu dự án gốc
```

---

## 2. Tóm tắt các Giai đoạn Phát triển

1.  **Giai đoạn 1: Tiền xử lý dữ liệu (ECG Preprocessing):** Lọc thông dải Butterworth (0.5 - 40 Hz), định vị đỉnh sóng QRS sử dụng giải thuật `gqrs`, phân cửa sổ trượt 10 giây (2500 mẫu), chuẩn hóa Z-Score, chia tập dữ liệu train/val/test độc lập để tránh rò rỉ dữ liệu (data leakage).
2.  **Giai đoạn 2: Định nghĩa cấu trúc mạng (Model Setup):** Định nghĩa mô hình tích chập giãn nở kết hợp khối chú ý kênh **Dilated-SE-FireNet** (110k tham số). Loại bỏ lớp augmentation ra khỏi đồ thị Keras để đảm bảo tương thích 100% với lượng tử hóa TFLite.
3.  **Giai đoạn 3: Huấn luyện đám mây & Đánh giá Float32:** Huấn luyện trên GPU Google Colab đạt độ chính xác validation tốt nhất **99.21%** ở Epoch 16. Cơ chế Early Stopping tự động dừng ở Epoch 26 để chống overfitting. Chạy đánh giá cục bộ trên tập Test đạt **Accuracy 95.23%** và **Recall AFIB 99.71%**.
4.  **Giai đoạn 4: Lượng tử hóa TFLite & Xác thực (TinyML):** Chuyển đổi mô hình sang kiểu dữ liệu lượng tử hóa số nguyên 8-bit toàn phần (Full Integer INT8) sử dụng 1000 mẫu đại diện để căn chỉnh (calibration). Đạt tỷ lệ nén **88.7%** (dung lượng mô hình chỉ còn **190 KB**) và độ chính xác kiểm thử hầu như giữ nguyên (**Accuracy 94.30%**, **Recall AFIB 99.96%**).

---

## 3. Hướng dẫn chạy và tái lập dự án (Step-by-step)

Mọi thao tác dưới đây đều được chạy từ thư mục gốc của dự án.

### Bước 1: Kích hoạt môi trường ảo Python
```bash
source .venv/bin/activate
```
*(Yêu cầu đã cài đặt đầy đủ các thư viện trong `requirements.txt` bằng lệnh `pip install -r requirements.txt`).*

### Bước 2: Kiểm thử đồ thị mô hình
Để đảm bảo mô hình tích chập được liên kết đúng cấu trúc và không có lỗi shape vật lý:
```bash
python -m src.check_model
```

### Bước 3: Huấn luyện mô hình
Do việc huấn luyện trên CPU máy tính thông thường mất rất nhiều thời gian (~50 phút/epoch), quy trình khuyên dùng là:
1.  Đăng tải tệp dữ liệu đã nén `processed.zip` lên Google Drive (sử dụng script `train_on_colab/upload_to_drive.py`).
2.  Mở notebook [train_on_colab/train_colab.ipynb](file:///home/pd/data/info_model/build_model/train_on_colab/train_colab.ipynb) trên **Google Colab**, chọn phần cứng **GPU T4** và chạy huấn luyện tự động.
3.  Sau khi huấn luyện thành công (khoảng 30 phút), tải tệp mô hình tốt nhất `best_dilated_se_firenet.keras` về máy tính và đặt vào thư mục `outputs/checkpoints/`.

### Bước 4: Chạy đánh giá chi tiết mô hình Float32
Đánh giá độ chính xác tổng thể và theo từng bệnh nhân trên tập test:
```bash
python -m src.evaluate
```
*Kết quả ma trận nhầm lẫn PNG và báo cáo JSON sẽ được lưu vào thư mục `outputs/reports/`.*

### Bước 5: Thực hiện lượng tử hóa TFLite
Xuất mô hình Keras sang phiên bản lượng tử hóa INT8 phục vụ TinyML:
```bash
python -m TFLite_Quantization.export_tflite
```
*Hai tệp mô hình `.tflite` mới sẽ xuất hiện tại thư mục `outputs/tflite/`.*

### Bước 6: Kiểm chứng hiệu năng mô hình lượng tử hóa INT8
Chạy kiểm chứng mô hình INT8 trên toàn bộ tập test gộp:
```bash
python -m TFLite_Quantization.verify_tflite
```
*Script sử dụng kỹ thuật batching thông minh giúp suy luận 65,000 mẫu chỉ trong chưa đầy 3 giây trên CPU.*

---

## 4. Kết quả so sánh hiệu năng thực tế

Dưới đây là bảng so sánh hiệu năng chi tiết giữa mô hình Float32 ban đầu và mô hình lượng tử hóa INT8 toàn phần trên tập test gộp 65,188 mẫu:

| Chỉ số (Metrics) | Mô hình Keras (Float32) | Mô hình TFLite (INT8) | Độ lệch (Difference) |
| :--- | :---: | :---: | :---: |
| **Kích thước tệp (File Size)** | ~1.68 MB | **190 KB** | **-88.7%** (Nén ~8.8 lần) |
| **Độ chính xác (Accuracy)** | 95.23% | 94.30% | -0.93% |
| **Độ nhạy AFIB (Recall)** | 99.71% | **99.96%** | **+0.25%** (Tốt hơn!) |
| **Độ xác thực AFIB (Precision)**| 90.31% | 88.41% | -1.90% |
| **F1-Score (AFIB)** | 0.9478 | 0.9383 | -0.0095 |
| **Số ca AFIB bị bỏ sót (FN)** | 82 mẫu | **10 mẫu** | **Giảm 72 mẫu** (An toàn hơn) |

Mô hình đã sẵn sàng triển khai thực tế trên các hệ thống vi điều khiển TinyML. Chi tiết kỹ thuật về thuật toán và ma trận nhầm lẫn của từng bệnh nhân có thể xem tại thư mục [report/](file:///home/pd/data/info_model/build_model/report/).
