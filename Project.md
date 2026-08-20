# Dự đoán Thời gian Hoạt động Còn lại (RUL) cho Động cơ Phản lực Cánh quạt sử dụng Học sâu Lai — Có khả năng Lượng hóa Độ bất định và Giải thích được

## 1. Giới thiệu

### 1.1 Phát biểu Vấn đề
Dự đoán Thời gian Hoạt động Còn lại (Remaining Useful Life — RUL) là một bài toán quan trọng trong Bảo trì Dự đoán (Predictive Maintenance — PdM) cho các hệ thống công nghiệp. Ước lượng RUL chính xác giúp tối ưu hóa lịch bảo trì, giảm thời gian ngừng hoạt động ngoài kế hoạch và ngăn ngừa các sự cố thảm khốc. Trong ngành hàng không, sự suy giảm của động cơ phản lực cánh quạt (turbofan) ảnh hưởng trực tiếp đến an toàn bay và chi phí vận hành.

Tuy nhiên, các phương pháp học sâu hiện tại cho bài toán dự đoán RUL đang đối mặt với ba hạn chế chính:
1. **Dự đoán điểm, không có khoảng tin cậy** — Mô hình chỉ đưa ra một giá trị RUL duy nhất (ví dụ: "42 chu kỳ") mà không lượng hóa mức độ tin cậy, gây rủi ro cho các quyết định an toàn.
2. **Tính hộp đen** — Kỹ sư không thể hiểu được *tại sao* mô hình đưa ra một giá trị RUL nhất định, làm giảm niềm tin và khả năng áp dụng trong thực tế.
3. **Khả năng tổng quát hóa hạn chế** — Hầu hết mô hình chỉ được kiểm chứng trên một điều kiện vận hành và một chế độ lỗi duy nhất, chưa trả lời được câu hỏi về độ bền vững (robustness) trên các kịch bản đa dạng.

### 1.2 Mục tiêu
Dự án này nhằm phát triển một **framework học sâu lai (hybrid deep learning)** cho dự đoán RUL với các đặc tính:
- **Chính xác** — Đạt hiệu năng cạnh tranh so với các benchmark tiên tiến nhất (SOTA) trên bộ dữ liệu NASA C-MAPSS.
- **Lượng hóa được độ bất định** — Cung cấp khoảng dự đoán (ví dụ: "RUL = 42 ± 8 chu kỳ") thông qua Monte Carlo Dropout.
- **Giải thích được** — Xác định cảm biến nào đóng góp nhiều nhất vào dự đoán thông qua Attention Heatmap và/hoặc SHAP values.
- **Tổng quát hóa tốt** — Được kiểm chứng trên cả bốn tập con C-MAPSS (FD001–FD004), bao gồm đơn/đa điều kiện vận hành và đơn/đa chế độ lỗi.

### 1.3 Phạm vi
- **Trong phạm vi:** Tiền xử lý dữ liệu, trích xuất đặc trưng (feature engineering), các mô hình ML cơ sở (baseline), kiến trúc DL lai (CNN-LSTM), thí nghiệm với Transformer, lượng hóa độ bất định, giải thích mô hình, đánh giá toàn diện.
- **Ngoài phạm vi:** Triển khai thời gian thực, kiểm thử trên phần cứng (hardware-in-the-loop), các bộ dữ liệu ngoài CMAPSS.

---

## 2. Bộ dữ liệu

### 2.1 NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

| Tập con | Động cơ Train | Động cơ Test | Điều kiện Vận hành | Chế độ Lỗi | Độ phức tạp |
|:---|:---|:---|:---|:---|:---|
| FD001 | 100 | 100 | 1 (Mực nước biển) | 1 (Suy giảm HPC) | Thấp |
| FD002 | 260 | 259 | 6 | 1 (Suy giảm HPC) | Trung bình |
| FD003 | 100 | 100 | 1 (Mực nước biển) | 2 (Suy giảm HPC + Quạt) | Trung bình |
| FD004 | 248 | 249 | 6 | 2 (Suy giảm HPC + Quạt) | Cao |

Mỗi bản ghi động cơ bao gồm:
- **26 cột:** Mã động cơ, Chu kỳ, 3 Thiết lập Vận hành, 21 Giá trị Cảm biến
- **Quỹ đạo chạy đến hỏng (run-to-failure)** trong tập train (RUL thực tế được tính từ chu kỳ tối đa)
- **Quỹ đạo bị cắt ngắn** trong tập test kèm file RUL thực tế riêng

### 2.2 Đặc điểm Dữ liệu Chính (từ EDA)
- **Cảm biến có thông tin:** Cảm biến 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
- **Cảm biến không có thông tin (FD001/FD003):** Cảm biến 1, 5, 6, 10, 16, 18, 19 (phương sai gần bằng 0)
- **Biến ẩn — Bộ Phân biệt Chế độ Lỗi:** Cảm biến 7, 10, 12, 16, 20 phân biệt được lỗi HPC và lỗi Quạt trong FD003/FD004
- **Suy giảm phi tuyến:** Dạng tuyến tính từng đoạn (piecewise linear) — cảm biến ổn định ở giai đoạn đầu đời, suy giảm nhanh ở cuối đời
- **Bùng nổ phương sai:** Nhiễu cảm biến tăng khi động cơ gần hỏng (heteroscedasticity)
- **Trễ thời gian:** Các cảm biến khác nhau phản ứng với suy giảm ở các độ trễ thời gian khác nhau

---

## 3. Phương pháp

### 3.1 Quy trình Tiền xử lý Dữ liệu

```
Dữ liệu thô → Loại bỏ Cảm biến Không thông tin → Phân cụm Điều kiện Vận hành (FD002/FD004)
             → Chuẩn hóa Theo cụm (MinMaxScaler 0–1) → RUL Tuyến tính Từng đoạn (cap=125)
             → Trích xuất Đặc trưng → Cửa sổ Trượt Thích ứng → Chia tập Theo Động cơ
```

#### 3.1.1 Biến đổi Nhãn RUL
- **RUL Tuyến tính Từng đoạn (Piecewise Linear)** với giới hạn tối đa = 125 chu kỳ
- Lý do: Giá trị cảm biến ở giai đoạn đầu đời gần như không phân biệt được; việc giới hạn giúp mô hình tập trung vào giai đoạn suy giảm quan trọng

#### 3.1.2 Trích xuất Đặc trưng (3 Tầng)

| Tầng | Đặc trưng | Mô tả |
|:---|:---|:---|
| Tầng 1 — Lựa chọn | Loại bỏ cảm biến phương sai bằng 0 | Xóa các kênh không mang thông tin |
| Tầng 2 — Thống kê Trượt | Trung bình trượt, Độ lệch chuẩn trượt (window=5) | Nắm bắt xu hướng cục bộ và độ biến động |
| Tầng 3 — Nâng cao | Tốc độ thay đổi (đạo hàm bậc 1), Tỷ lệ chéo giữa cảm biến | Nắm bắt tốc độ suy giảm và mối quan hệ giữa các cảm biến |

#### 3.1.3 Chiến lược Chuẩn hóa
- **FD001, FD003** (một điều kiện vận hành): MinMaxScaler chuẩn, fit trên tập train
- **FD002, FD004** (sáu điều kiện vận hành): Phân cụm điều kiện vận hành trước (K-Means trên operational settings), sau đó chuẩn hóa trong từng cụm rồi mới áp dụng MinMaxScaler

#### 3.1.4 Chia tập Dữ liệu
- **Chia theo động cơ** (không rò rỉ dữ liệu): Các động cơ train vs các động cơ validation
- Tập test: File test chính thức của C-MAPSS kèm file RUL thực tế

### 3.2 Kiến trúc Mô hình

#### 3.2.1 Mô hình Cơ sở (Machine Learning)

| Mô hình | Mục đích |
|:---|:---|
| Linear Regression | Kiểm tra cơ bản — thiết lập baseline đơn giản nhất |
| Support Vector Regression (SVR) | Baseline phi tuyến, không ensemble |
| Random Forest | Baseline ensemble, dựa trên cây |
| XGBoost | Baseline ML mạnh nhất trước khi dùng deep learning |

#### 3.2.2 Mô hình Chính — CNN-LSTM Lai

```
Chuỗi đầu vào (cửa sổ thích ứng × đặc trưng)
    │
    ▼
┌─────────────────────────────┐
│   Các lớp CNN 1D             │  ← Trích xuất mẫu không gian cục bộ giữa các cảm biến
│   (Conv1D + ReLU + Pool)     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Các lớp LSTM / BiLSTM      │  ← Mô hình hóa phụ thuộc thời gian trong suy giảm
│   + Cơ chế Attention         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Các lớp Kết nối Đầy đủ     │  ← Đầu hồi quy (regression head)
│   + Dropout                  │
└──────────────┬──────────────┘
               │
               ▼
         Dự đoán RUL
```

- **Khối CNN:** Nắm bắt tương quan không gian cục bộ giữa các cảm biến tại mỗi bước thời gian
- **Khối LSTM/BiLSTM:** Nắm bắt các mẫu suy giảm theo thời gian trên toàn chuỗi
- **Cơ chế Attention:** Làm nổi bật các bước thời gian quan trọng nhất cho dự đoán
- **Dropout:** Cho phép Monte Carlo Dropout để lượng hóa độ bất định khi suy luận

#### 3.2.3 Mô hình Thí nghiệm — Dựa trên Transformer

Kiến trúc Transformer Encoder sẽ được khám phá như một thí nghiệm so sánh:
- Self-attention để nắm bắt phụ thuộc thời gian toàn cục
- Positional encoding cho thứ tự thời gian
- So sánh với CNN-LSTM để đánh giá hiệu quả trên dữ liệu nhỏ (100 động cơ)

### 3.3 Lượng hóa Độ Bất định — Monte Carlo Dropout

Thay vì đưa ra một giá trị RUL duy nhất, mô hình cung cấp một **phân phối dự đoán**:

1. Giữ các lớp dropout **bật** trong quá trình suy luận (inference)
2. Chạy N lần lan truyền thuận (N = 50–100) cho mỗi đầu vào
3. Tính: **Trung bình** (dự đoán điểm) và **Độ lệch chuẩn** (độ bất định)
4. Báo cáo: `RUL = μ ± 2σ` (khoảng tin cậy 95%)

Điều này cho phép người vận hành phân biệt giữa "RUL = 42 (độ tin cậy cao)" và "RUL = 42 (độ tin cậy thấp, có thể dao động từ 20 đến 65)."

### 3.4 Khả năng Giải thích Mô hình

Hai cách tiếp cận bổ sung cho nhau:
1. **Attention Heatmap:** Trực quan hóa trọng số attention từ lớp attention LSTM để thể hiện mô hình tập trung vào bước thời gian nào
2. **SHAP Values:** Phân tích tầm quan trọng đặc trưng hậu kỳ (post-hoc) để xếp hạng cảm biến nào chi phối dự đoán cho từng động cơ cụ thể

---

## 4. Đánh giá

### 4.1 Chỉ số Đánh giá

| Chỉ số | Công thức | Mục đích |
|:---|:---|:---|
| RMSE | √(Σ(ŷ - y)² / n) | Chỉ số so sánh chính |
| MAE | Σ\|ŷ - y\| / n | Bền vững trước ngoại lai (outlier) |
| NASA Score | Σ exp(-d/13)-1 nếu sớm; Σ exp(d/10)-1 nếu muộn | Phạt nặng dự đoán muộn (an toàn) |

### 4.2 Quy trình Đánh giá
- Huấn luyện trên tập train chính thức (chia theo động cơ cho tập validation)
- Kiểm thử trên tập test chính thức kèm RUL thực tế
- Báo cáo cả ba chỉ số trên **cả bốn tập con** (FD001–FD004)
- Nghiên cứu cắt bỏ (Ablation study): ảnh hưởng của từng tầng đặc trưng, kích thước cửa sổ, thành phần mô hình

### 4.3 So sánh Benchmark (Mục tiêu)

| Mô hình | FD001 | FD002 | FD003 | FD004 |
|:---|:---|:---|:---|:---|
| ML Cơ sở (LR/SVR/RF/XGB) | ~18–25 | ~25–35 | ~18–25 | ~28–38 |
| LSTM Cơ bản | ~14–16 | ~20–25 | ~14–16 | ~22–28 |
| **CNN-LSTM (dự án này)** | **~12–14** | **~18–22** | **~13–15** | **~20–25** |
| Transformer (thí nghiệm) | TBD | TBD | TBD | TBD |

---

## 5. Tiến độ Dự án (12 Tuần)

| Tuần | Giai đoạn | Công việc | Sản phẩm |
|:---|:---|:---|:---|
| 1 | Nền tảng | Hoàn thiện EDA notebook, đọc 5 paper quan trọng, chốt kế hoạch đặc trưng | EDA notebook, ghi chú tổng quan tài liệu |
| 2 | Nền tảng | Xây data pipeline (nạp → làm sạch → chuẩn hóa → cửa sổ → chia tập) | Scripts data pipeline tái sử dụng |
| 3 | Cơ sở | Triển khai LR, SVR, RF, XGBoost trên FD001 | Bảng so sánh baseline |
| 4 | Cơ sở | Mở rộng baseline sang FD002–FD004, triển khai LSTM baseline | Kết quả baseline mở rộng |
| 5 | Mô hình Chính | Thiết kế và triển khai kiến trúc CNN-LSTM | Lần chạy CNN-LSTM đầu tiên |
| 6 | Mô hình Chính | Tinh chỉnh CNN-LSTM, thêm cơ chế Attention | Kết quả CNN-LSTM tối ưu |
| 7 | Thí nghiệm | Triển khai biến thể Transformer, so sánh với CNN-LSTM | So sánh Transformer vs CNN-LSTM |
| 8 | Phân tích | Nghiên cứu cắt bỏ (đặc trưng, cửa sổ, thành phần) | Bảng kết quả ablation |
| 9 | Điểm nhấn | Triển khai MC Dropout cho lượng hóa độ bất định | Trực quan hóa độ bất định |
| 10 | Điểm nhấn | Triển khai Attention heatmap + SHAP giải thích mô hình | Trực quan hóa khả năng giải thích |
| 11 | Báo cáo | Viết bản nháp báo cáo, tạo hình và bảng | Bản nháp paper/báo cáo |
| 12 | Hoàn thiện | Chỉnh sửa báo cáo, dọn code, chuẩn bị thuyết trình | Nộp bài cuối cùng |

### Phân công (Nhóm 2 người)

| Thành viên | Trách nhiệm chính |
|:---|:---|
| Thành viên A | Data pipeline, trích xuất đặc trưng, mô hình baseline, viết báo cáo |
| Thành viên B | Kiến trúc DL (CNN-LSTM, Transformer), độ bất định, giải thích mô hình |

> Cả hai cùng hợp tác trong EDA, đánh giá, và thuyết trình cuối cùng.

---

## 6. Công nghệ Sử dụng

| Thành phần | Công cụ |
|:---|:---|
| Ngôn ngữ | Python 3.10+ |
| Học sâu | PyTorch |
| ML Cơ sở | scikit-learn, XGBoost |
| Xử lý Dữ liệu | pandas, NumPy |
| Trực quan hóa | matplotlib, seaborn, plotly |
| Giải thích Mô hình | SHAP, trực quan hóa attention tùy chỉnh |
| Tài nguyên Tính toán | Google Colab (GPU) hoặc máy chủ GPU |
| Quản lý Phiên bản | Git / GitHub |
| Báo cáo | LaTeX hoặc Markdown |

---

## 7. Đóng góp Kỳ vọng

1. **Kiến trúc CNN-LSTM lai** kết hợp attention cho dự đoán RUL, được kiểm chứng trên cả bốn tập con C-MAPSS
2. **Dự đoán có lượng hóa độ bất định** thông qua Monte Carlo Dropout, cung cấp khoảng tin cậy bên cạnh dự đoán điểm
3. **Kết quả giải thích được** chỉ ra cảm biến và bước thời gian nào chi phối dự đoán, kết nối giữa AI và kiến thức chuyên ngành
4. **So sánh benchmark toàn diện** giữa các baseline ML, LSTM, CNN-LSTM, và Transformer
5. **Phân tích xuyên tập dữ liệu** cho thấy hiệu năng mô hình suy giảm như thế nào khi độ phức tạp tăng (đa điều kiện, đa lỗi)

---

## 8. Quản lý Rủi ro

| Rủi ro | Khả năng xảy ra | Biện pháp Giảm thiểu |
|:---|:---|:---|
| Rò rỉ dữ liệu làm thổi phồng kết quả | Cao | Chia tập nghiêm ngặt theo động cơ, không xáo trộn ngẫu nhiên theo dòng |
| Transformer kém hiệu quả trên dữ liệu nhỏ | Trung bình | CNN-LSTM làm phương án dự phòng, Transformer chỉ là thí nghiệm |
| Colab GPU hết thời gian khi huấn luyện | Trung bình | Lưu checkpoint thường xuyên, dùng Colab Pro nếu cần |
| Không đủ thời gian viết báo cáo | Cao | Viết tích lũy từ Tuần 1, ghi chú mọi thí nghiệm |
| FD002/FD004 kết quả kém do chuẩn hóa sai | Trung bình | Chuẩn hóa theo cụm, kiểm tra bằng trực quan hóa |

---

## 9. Tài liệu Tham khảo

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. PHM08.
2. Heimes, F. O. (2008). *Recurrent Neural Networks for Remaining Useful Life Estimation*. PHM08.
3. Li, X., Ding, Q., & Sun, J. Q. (2018). *Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks*. Reliability Engineering & System Safety.
4. Mo, Y., et al. (2021). *Remaining Useful Life Estimation via Transformer Encoder Enhanced by a Gated Convolutional Unit*. Journal of Intelligent Manufacturing.

> Các tài liệu tham khảo bổ sung sẽ được thêm trong quá trình tổng quan tài liệu (Tuần 1).

---

*Khởi động dự án: Tháng 8/2026*
*Quy mô nhóm: 2 thành viên*
*Thời lượng: 12 tuần*
*Ngôn ngữ báo cáo: Tiếng Anh*
