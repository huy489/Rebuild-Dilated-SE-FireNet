# Giai đoạn 2: Xây dựng mô hình Dilated-SE-FireNet

## 1. Mục tiêu của giai đoạn

Giai đoạn này tập trung vào việc xây dựng bộ khung mô hình học sâu để phân loại rung nhĩ từ ECG đã được preprocessing.

Mục tiêu chính:

- Xác định rõ input và output của mô hình.
- Thiết kế kiến trúc Dilated-SE-FireNet.
- Đảm bảo mô hình tương thích với dữ liệu đã preprocessing.
- Kiểm tra mô hình có thể chạy forward pass.
- Kiểm tra mô hình có thể train thử ở mức tối thiểu.
- Chốt phiên bản kiến trúc đầu tiên trước khi bước sang training chính thức.

Giai đoạn này chưa tập trung vào tối ưu accuracy.

---

## 2. Điều kiện bắt đầu

Giai đoạn 2 chỉ bắt đầu khi Giai đoạn 1 đã hoàn tất.

Dữ liệu sau preprocessing cần có dạng:

```text
X_train: (N, 2500, 1)
y_train: (N,)

X_val:   (N, 2500, 1)
y_val:   (N,)

X_test:  (N, 2500, 1)
y_test:  (N,)
```

Quy ước nhãn:

```text
0 = Normal
1 = AFIB
```

Mỗi mẫu ECG đại diện cho:

```text
10 giây ECG
Sampling rate = 250 Hz
2500 điểm dữ liệu
1 kênh ECG
```

---

## 3. Flow tổng thể

```text
Dữ liệu đã preprocessing
        ↓
Xác nhận input/output contract
        ↓
Thiết kế kiến trúc model
        ↓
Xây dựng các khối kiến trúc chính
        ↓
Ghép thành mô hình hoàn chỉnh
        ↓
Kiểm tra shape từng tầng
        ↓
Kiểm tra forward pass bằng dữ liệu giả
        ↓
Kiểm tra forward pass bằng dữ liệu thật
        ↓
Compile model
        ↓
Train thử ngắn
        ↓
Chốt model version 1
```

---

## 4. Bước 1: Xác nhận input/output contract

Trước khi xây mô hình, cần xác nhận rõ mô hình sẽ nhận gì và trả ra gì.

Input của mô hình:

```text
Shape: (2500, 1)
```

Ý nghĩa:

```text
2500 = số điểm ECG trong 10 giây
1    = một kênh ECG
```

Output của mô hình:

```text
Shape: (2,)
```

Ý nghĩa:

```text
output[0] = xác suất Normal
output[1] = xác suất AFIB
```

Logic phân loại:

```text
Nếu output[0] > output[1] → Normal
Nếu output[1] > output[0] → AFIB
```

Kết quả cần chốt sau bước này:

```text
Input model  = ECG window 10 giây, shape (2500, 1)
Output model = vector xác suất 2 lớp [Normal, AFIB]
```

---

## 5. Bước 2: Xác định yêu cầu kiến trúc từ bệnh rung nhĩ

Các đặc điểm của rung nhĩ cần mô hình học được:

```text
1. Khoảng R-R không đều
2. Mất sóng P
3. Xuất hiện sóng f nhỏ và hỗn loạn
4. Nhịp bất thường kéo dài theo thời gian
5. Tín hiệu ECG single-lead dễ nhiễu
```

Từ đó dẫn đến các lựa chọn kiến trúc:

```text
R-R không đều
→ cần nhìn nhiều nhịp liên tiếp
→ dùng Dilated Convolution

Mất sóng P, sóng f nhỏ
→ cần học đặc trưng hình thái cục bộ
→ dùng Conv1D

Feature nhỏ, dễ bị nhiễu
→ cần chọn lọc feature quan trọng
→ dùng SE Block

Cần model nhẹ để deploy edge
→ cần giảm tham số
→ dùng Fire Module

Nhiều tầng xử lý có thể làm mất thông tin gốc
→ cần giữ thông tin qua các tầng
→ dùng Residual Connection
```

Kết quả cần chốt sau bước này:

```text
Kiến trúc chính = Conv1D + Fire Module + Dilated Conv + SE Block + Residual
```

---

## 6. Bước 3: Thiết kế kiến trúc tổng thể

Kiến trúc tổng thể của mô hình:

```text
Input ECG
    ↓
Augmentation nhẹ
    ↓
Stem Conv1D
    ↓
Dilated-SE-Fire Block 1
    ↓
Dilated-SE-Fire Block 2
    ↓
Dilated-SE-Fire Block 3
    ↓
Dilated-SE-Fire Block 4
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense Softmax
    ↓
Output [Normal, AFIB]
```

Ý nghĩa từng phần:

```text
Input ECG:
Nhận tín hiệu ECG đã preprocessing.

Augmentation:
Giúp model chịu được nhiễu và biến thiên biên độ.

Stem Conv1D:
Trích đặc trưng ban đầu và giảm chiều dài tín hiệu.

Dilated-SE-Fire Blocks:
Lõi chính của mô hình, học đặc trưng AFIB theo cả hình thái sóng và nhịp dài hạn.

Global Average Pooling:
Gom đặc trưng toàn bộ đoạn ECG 10 giây.

Dropout:
Giảm overfitting.

Dense Softmax:
Xuất xác suất Normal và AFIB.
```

---

## 7. Bước 4: Thiết kế Stem Conv1D

Stem là khối đầu tiên xử lý ECG.

Flow của Stem:

```text
Input ECG (2500, 1)
        ↓
Conv1D
        ↓
Batch Normalization
        ↓
LeakyReLU
        ↓
Feature map ban đầu
```

Mục đích:

```text
1. Bắt các đặc trưng cơ bản của ECG.
2. Nhận diện cạnh dốc, QRS, đỉnh R.
3. Giảm chiều dài tín hiệu để tiết kiệm tính toán.
4. Chuẩn bị feature map cho các block phía sau.
```

Kết quả mong muốn:

```text
Tín hiệu ECG raw-level được chuyển thành feature map ban đầu.
```

---

## 8. Bước 5: Thiết kế Dilated-SE-Fire Block

Mỗi Dilated-SE-Fire Block có flow:

```text
Input feature map
        ↓
Squeeze Conv1D 1x1
        ↓
Tách thành 2 nhánh
        ↓
Nhánh 1: Expand Conv1D 1x1
Nhánh 2: Expand Conv1D 3x3 Dilated
        ↓
Concatenate hai nhánh
        ↓
SE Block
        ↓
Residual Add
        ↓
Output feature map
```

Ý nghĩa từng phần:

```text
Squeeze 1x1:
Giảm số kênh để giảm tham số.

Expand 1x1:
Học tổ hợp đặc trưng theo kênh.

Expand 3x3 Dilated:
Học đặc trưng theo thời gian với vùng nhìn rộng hơn.

Concatenate:
Ghép đặc trưng từ hai nhánh.

SE Block:
Chọn lọc kênh đặc trưng quan trọng.

Residual Add:
Giữ lại thông tin gốc và giúp truyền gradient ổn định.
```

Kết quả mong muốn:

```text
Mỗi block tạo ra feature map giàu thông tin hơn nhưng vẫn giữ mô hình nhẹ.
```

---

## 9. Bước 6: Thiết kế dilation theo tầng

Dilation tăng dần qua các block:

```text
Block 1: dilation = 1
Block 2: dilation = 2
Block 3: dilation = 4
Block 4: dilation = 8
```

Ý nghĩa:

```text
Dilation = 1:
Học đặc trưng cục bộ như QRS, sóng P, sóng f.

Dilation = 2:
Nhìn xa hơn một chút giữa các nhịp.

Dilation = 4:
Bắt đầu học quan hệ giữa nhiều nhịp tim.

Dilation = 8:
Học bất thường dài hạn của khoảng R-R.
```

Kết quả mong muốn:

```text
Model vừa học được hình thái sóng ECG, vừa học được sự bất thường nhịp theo thời gian.
```

---

## 10. Bước 7: Thiết kế SE Block

SE Block có flow:

```text
Input feature map
        ↓
Global Average Pooling theo thời gian
        ↓
Dense giảm chiều
        ↓
Dense sigmoid
        ↓
Tạo trọng số cho từng kênh
        ↓
Nhân trọng số vào feature map ban đầu
```

Mục đích:

```text
1. Tăng trọng số các kênh chứa đặc trưng AFIB.
2. Giảm trọng số các kênh chứa nhiễu.
3. Giúp model tập trung vào feature quan trọng.
```

Kết quả mong muốn:

```text
Feature map sau SE có thông tin hữu ích rõ hơn và ít nhiễu hơn.
```

---

## 11. Bước 8: Thiết kế classifier cuối

Sau các Dilated-SE-Fire Blocks, mô hình cần chuyển feature map thành kết quả phân loại.

Flow classifier:

```text
Feature map cuối
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Dense 2 units
        ↓
Softmax
        ↓
[p_normal, p_afib]
```

Ý nghĩa:

```text
Global Average Pooling:
Tóm tắt toàn bộ thông tin của đoạn ECG 10 giây.

Dropout:
Giảm overfitting.

Dense 2 units:
Tạo 2 điểm số tương ứng 2 lớp.

Softmax:
Chuyển điểm số thành xác suất.
```

Kết quả mong muốn:

```text
Model trả ra vector xác suất gồm 2 giá trị:
[Normal, AFIB]
```

---

## 12. Bước 9: Kiểm tra shape toàn mô hình

Sau khi ghép model, cần kiểm tra shape đi qua từng phần.

Flow kiểm tra:

```text
Input
        ↓
Kiểm tra shape sau Stem
        ↓
Kiểm tra shape sau Block 1
        ↓
Kiểm tra shape sau Block 2
        ↓
Kiểm tra shape sau Block 3
        ↓
Kiểm tra shape sau Block 4
        ↓
Kiểm tra shape trước classifier
        ↓
Kiểm tra shape output
```

Yêu cầu cuối cùng:

```text
Input shape  = (None, 2500, 1)
Output shape = (None, 2)
```

Nếu output không phải `(None, 2)`, chưa được chuyển sang giai đoạn train.

---

## 13. Bước 10: Kiểm tra forward pass bằng dữ liệu giả

Trước khi dùng dữ liệu thật, cần kiểm tra model bằng một mẫu giả.

Flow kiểm tra:

```text
Tạo input giả shape (1, 2500, 1)
        ↓
Đưa qua model
        ↓
Nhận output
        ↓
Kiểm tra output shape
        ↓
Kiểm tra output có hợp lệ không
```

Output hợp lệ cần có:

```text
Shape = (1, 2)
Không có NaN
Không có Inf
Tổng xác suất xấp xỉ 1
```

Kết quả mong muốn:

```text
Model graph hoạt động đúng.
```

---

## 14. Bước 11: Kiểm tra forward pass bằng dữ liệu thật

Sau khi dữ liệu giả chạy được, kiểm tra bằng một batch dữ liệu thật.

Flow kiểm tra:

```text
Load một batch nhỏ từ X_train
        ↓
Kiểm tra shape batch
        ↓
Đưa batch qua model
        ↓
Nhận output
        ↓
Kiểm tra output shape
        ↓
So sánh số lượng output với số lượng label
```

Yêu cầu:

```text
X_batch shape = (batch_size, 2500, 1)
y_batch shape = (batch_size,)
output shape  = (batch_size, 2)
```

Kết quả mong muốn:

```text
Model tương thích với dữ liệu preprocessing.
```

---

## 15. Bước 12: Compile model

Sau khi forward pass ổn, tiến hành compile model.

Do output là softmax 2 lớp, logic huấn luyện cần thống nhất:

```text
Label dạng số nguyên:
0 = Normal
1 = AFIB
```

Loss phù hợp:

```text
Sparse Categorical Crossentropy
```

Output phù hợp:

```text
Dense(2, Softmax)
```

Metric cần theo dõi:

```text
Accuracy
Recall AFIB
Precision AFIB
F1-score
Confusion Matrix
```

Trong bài toán y tế, cần đặc biệt quan tâm:

```text
Recall AFIB
```

Vì bỏ sót AFIB nguy hiểm hơn báo nhầm AFIB.

---

## 16. Bước 13: Train thử ngắn

Giai đoạn này chỉ train thử để kiểm tra pipeline.

Flow train thử:

```text
Chọn một phần nhỏ dữ liệu train
        ↓
Chọn một phần nhỏ dữ liệu validation
        ↓
Train 1 đến 3 epoch
        ↓
Theo dõi loss
        ↓
Theo dõi accuracy
        ↓
Kiểm tra model có lưu được không
```

Kết quả hợp lệ:

```text
Loss không NaN
Loss có thay đổi
Accuracy không đứng im bất thường
Validation chạy được
Model lưu được checkpoint
```

Kết quả chưa cần đạt accuracy cao.

---

## 17. Bước 14: Kiểm tra lỗi cơ bản

Sau khi train thử, cần kiểm tra các lỗi thường gặp.

Checklist:

```text
Input shape có đúng không?
Output shape có đúng không?
Label có đúng 0 và 1 không?
Loss có bị NaN không?
Output softmax có tổng gần bằng 1 không?
Model có quá lớn không?
Model có lưu được không?
Model có load lại được không?
```

Nếu có lỗi, quay lại bước tương ứng:

```text
Sai shape
→ quay lại kiểm tra input/output contract.

Sai output
→ quay lại classifier.

Loss NaN
→ kiểm tra dữ liệu, normalization, learning rate.

Model quá lớn
→ giảm số filter hoặc số block.

Label sai
→ kiểm tra dữ liệu preprocessing.
```

---

## 18. Bước 15: Chốt model version 1

Khi model đã build được và train thử ổn, chốt phiên bản đầu tiên.

Thông tin cần ghi lại:

```text
Tên model: Dilated-SE-FireNet-v1

Input shape:
(2500, 1)

Output:
2 lớp softmax

Class:
0 = Normal
1 = AFIB

Các thành phần chính:
Stem Conv1D
Dilated-SE-Fire Block
SE Block
Residual Connection
Global Average Pooling
Dropout
Dense Softmax

Dilation:
1, 2, 4, 8
```

Kết quả cần có:

```text
Model architecture đã ổn định.
Model chạy được forward pass.
Model train thử được.
Model lưu được.
Model sẵn sàng chuyển sang Giai đoạn 3.
```

---

## 19. Điều kiện kết thúc Giai đoạn 2

Chỉ chuyển sang Giai đoạn 3 khi đạt đủ các điều kiện sau:

```text
[ ] Dữ liệu input có shape đúng: (N, 2500, 1)
[ ] Label có dạng đúng: 0 hoặc 1
[ ] Model input đúng: (None, 2500, 1)
[ ] Model output đúng: (None, 2)
[ ] Output là softmax [Normal, AFIB]
[ ] Các block Dilated-SE-Fire hoạt động đúng
[ ] Dilation tăng dần: 1, 2, 4, 8
[ ] SE Block hoạt động đúng
[ ] Residual Connection không lỗi shape
[ ] Forward pass bằng dữ liệu giả chạy được
[ ] Forward pass bằng dữ liệu thật chạy được
[ ] Compile model thành công
[ ] Train thử ngắn không NaN
[ ] Model lưu được checkpoint
[ ] Model load lại được
```

---

## 20. Kết quả đầu ra của Giai đoạn 2

Sau khi hoàn thành Giai đoạn 2, project cần có:

```text
1. Một kiến trúc model hoàn chỉnh.
2. Model nhận đúng ECG window 10 giây.
3. Model xuất đúng xác suất Normal / AFIB.
4. Model đã được kiểm tra shape.
5. Model đã được kiểm tra forward pass.
6. Model đã được train thử ngắn.
7. Model version 1 đã được chốt.
```

---

## 21. Tóm tắt một dòng

Giai đoạn 2 là giai đoạn biến dữ liệu ECG đã preprocessing thành một mô hình Dilated-SE-FireNet hoàn chỉnh, kiểm tra mô hình chạy đúng về shape, output và logic phân loại, trước khi bước sang training chính thức.