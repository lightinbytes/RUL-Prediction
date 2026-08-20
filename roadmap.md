# Lộ trình Dự án Dự đoán RUL — Agile Scrum

> **Thời lượng:** 12 Tuần (Tháng 8 – Tháng 11/2026)
> **Nhóm:** 2 Thành viên (A, B)
> **Chu kỳ Sprint:** 1 Sprint = 1 Tuần
> **Nghi thức:** Lập kế hoạch Sprint (Thứ 2), Standup hàng ngày (bất đồng bộ), Đánh giá + Hồi cứu Sprint (Chủ nhật)

---

## Sprint 0 — Thiết lập Ban đầu

**Mục tiêu:** Thiết lập môi trường và công cụ làm việc

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 0.1 | Tạo GitHub repository, cấu trúc thư mục | A | ☐ |
| 0.2 | Thiết lập môi trường Python (requirements.txt) | B | ☐ |
| 0.3 | Cấu hình Google Colab / GPU runtime | B | ☐ |
| 0.4 | Thống nhất coding convention, branching strategy | A + B | ☐ |
| 0.5 | Tạo bảng quản lý dự án (GitHub Projects / Trello) | A | ☐ |

**Tiêu chí Hoàn thành:** Cả 2 thành viên đều có thể clone repo, chạy notebook trên Colab, push code.

**Cấu trúc Thư mục Đề xuất:**
```
RUL/
├── data/                  # Dữ liệu thô + đã xử lý
│   └── CMAPSSData/
├── notebooks/             # EDA, thí nghiệm
├── src/                   # Mã nguồn tái sử dụng
│   ├── data/              # Nạp dữ liệu, tiền xử lý
│   ├── features/          # Trích xuất đặc trưng
│   ├── models/            # Kiến trúc mô hình
│   ├── training/          # Vòng huấn luyện
│   └── evaluation/        # Chỉ số, trực quan hóa
├── results/               # Mô hình đã lưu, hình, bảng
├── reports/               # Bản nháp báo cáo
├── Project.md
├── roadmap.md
└── requirements.txt
```

---

## Sprint 1 — Hoàn thiện EDA & Tổng quan Tài liệu

**Mục tiêu Sprint:** Hoàn thiện phân tích dữ liệu và nắm vững bối cảnh nghiên cứu

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 1.1 | Hoàn thiện EDA notebook cho FD001 (phân phối, tương quan, đường cong suy giảm) | A | ☐ |
| 1.2 | Mở rộng EDA sang FD002, FD003, FD004 — so sánh xuyên tập dữ liệu | A | ☐ |
| 1.3 | Đọc và tóm tắt paper gốc: Saxena et al. 2008 | B | ☐ |
| 1.4 | Đọc và tóm tắt 2 paper CNN-LSTM cho RUL | B | ☐ |
| 1.5 | Đọc và tóm tắt 2 paper Transformer / Attention cho RUL | A | ☐ |
| 1.6 | Tổng hợp bảng so sánh tài liệu (mô hình, dataset, RMSE, phương pháp) | A + B | ☐ |

**Sản phẩm:**
- [ ] EDA notebook hoàn chỉnh (4 tập FD)
- [ ] Tóm tắt tổng quan tài liệu (≥ 5 paper)
- [ ] Kế hoạch trích xuất đặc trưng

**Câu hỏi Đánh giá Sprint:**
- Cảm biến nào nên loại bỏ ở mỗi FD?
- Paper nào có hướng tiếp cận gần nhất với mục tiêu?
- Khoảng trống nào trong tài liệu mà mình có thể lấp đầy?

---

## Sprint 2 — Xây dựng Data Pipeline

**Mục tiêu Sprint:** Xây dựng data pipeline hoàn chỉnh, tái sử dụng cho tất cả thí nghiệm

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 2.1 | Viết lớp `DataLoader`: nạp dữ liệu thô, gán tên cột | A | ☐ |
| 2.2 | Viết tính toán RUL: Tuyến tính Từng đoạn (cap=125) | A | ☐ |
| 2.3 | Viết logic lựa chọn cảm biến (loại bỏ phương sai bằng 0 theo FD) | A | ☐ |
| 2.4 | Viết phân cụm điều kiện vận hành (K-Means) cho FD002/FD004 | B | ☐ |
| 2.5 | Viết chuẩn hóa: MinMaxScaler (theo cụm cho FD002/FD004) | B | ☐ |
| 2.6 | Viết trích xuất đặc trưng: trung bình trượt, độ lệch chuẩn trượt, tốc độ thay đổi | A | ☐ |
| 2.7 | Viết cửa sổ trượt thích ứng + chia tập theo động cơ | B | ☐ |
| 2.8 | Viết PyTorch Dataset & DataLoader | B | ☐ |
| 2.9 | Kiểm thử: xác minh không rò rỉ dữ liệu, giá trị RUL đúng, kích thước đúng | A + B | ☐ |

**Sản phẩm:**
- [ ] Module `src/data/` hoàn chỉnh
- [ ] Module `src/features/` hoàn chỉnh
- [ ] Pipeline chạy đầu-cuối cho cả 4 tập FD
- [ ] Notebook xác minh: trực quan hóa dữ liệu đã xử lý

**Tiêu chí Hoàn thành:** `python -c "from src.data import load_and_preprocess; X, y = load_and_preprocess('FD001')"` chạy thành công.

---

## Sprint 3 — Các Mô hình ML Cơ sở

**Mục tiêu Sprint:** Thiết lập mức hiệu năng sàn với 4 mô hình ML

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 3.1 | Viết module đánh giá: RMSE, MAE, NASA Score | B | ☐ |
| 3.2 | Huấn luyện + đánh giá Linear Regression trên FD001 | A | ☐ |
| 3.3 | Huấn luyện + đánh giá SVR trên FD001 | A | ☐ |
| 3.4 | Huấn luyện + đánh giá Random Forest trên FD001 | B | ☐ |
| 3.5 | Huấn luyện + đánh giá XGBoost trên FD001 | B | ☐ |
| 3.6 | Tạo bảng so sánh baseline (RMSE, MAE, NASA Score) | A + B | ☐ |
| 3.7 | Chạy baseline tốt nhất trên FD002–FD004 | A + B | ☐ |
| 3.8 | Trực quan hóa: Biểu đồ phân tán RUL Dự đoán vs Thực tế | A | ☐ |

**Sản phẩm:**
- [ ] `src/evaluation/metrics.py` hoàn chỉnh
- [ ] Bảng so sánh baseline (4 mô hình × 4 tập × 3 chỉ số)
- [ ] Biểu đồ RUL Dự đoán vs Thực tế

**Câu hỏi Đánh giá Sprint:**
- Baseline nào tốt nhất? Tại sao?
- Đặc trưng nào quan trọng nhất (feature importance từ RF/XGBoost)?
- RMSE đạt được có hợp lý so với tài liệu?

---

## Sprint 4 — LSTM Cơ sở

**Mục tiêu Sprint:** Xây dựng deep learning baseline đầu tiên

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 4.1 | Viết mô hình LSTM cơ bản (PyTorch) | B | ☐ |
| 4.2 | Viết vòng huấn luyện: hàm mất mát, optimizer, dừng sớm | B | ☐ |
| 4.3 | Viết lưu/nạp checkpoint | B | ☐ |
| 4.4 | Huấn luyện LSTM trên FD001, điều chỉnh learning rate + hidden size | B | ☐ |
| 4.5 | Đánh giá LSTM trên FD001 (RMSE, MAE, NASA Score) | A | ☐ |
| 4.6 | Mở rộng LSTM sang FD002–FD004 | A + B | ☐ |
| 4.7 | So sánh LSTM vs ML baseline, cập nhật bảng tổng hợp | A | ☐ |
| 4.8 | Vẽ đường cong huấn luyện (loss theo epoch) | A | ☐ |

**Sản phẩm:**
- [ ] `src/models/lstm.py`
- [ ] `src/training/trainer.py`
- [ ] Kết quả LSTM trên 4 tập
- [ ] Bảng so sánh cập nhật: ML baseline + LSTM

**Tiêu chí Hoàn thành:** LSTM RMSE trên FD001 ≤ 16.0 (nếu không đạt → debug trước khi qua sprint tiếp).

---

## Sprint 5 — Kiến trúc CNN-LSTM (Phần 1)

**Mục tiêu Sprint:** Triển khai kiến trúc chính CNN-LSTM

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 5.1 | Thiết kế kiến trúc CNN-LSTM (số lớp, kích thước kernel, hidden units) | A + B | ☐ |
| 5.2 | Viết bộ trích xuất đặc trưng CNN (các lớp Conv 1D) | B | ☐ |
| 5.3 | Viết bộ mã hóa thời gian LSTM | B | ☐ |
| 5.4 | Tích hợp CNN + LSTM thành mô hình thống nhất | B | ☐ |
| 5.5 | Huấn luyện CNN-LSTM trên FD001 (hyperparameter ban đầu) | B | ☐ |
| 5.6 | So sánh CNN-LSTM vs LSTM cơ bản trên FD001 | A | ☐ |
| 5.7 | Bắt đầu viết báo cáo: phần Giới thiệu + Tổng quan Tài liệu | A | ☐ |

**Sản phẩm:**
- [ ] `src/models/cnn_lstm.py`
- [ ] Kết quả CNN-LSTM ban đầu trên FD001
- [ ] Bản nháp báo cáo: Giới thiệu + Tổng quan Tài liệu

**Câu hỏi Đánh giá Sprint:**
- CNN-LSTM có vượt LSTM cơ bản không?
- Nếu không → vấn đề ở đâu? (kiến trúc, hyperparameter, dữ liệu?)

---

## Sprint 6 — CNN-LSTM + Attention (Phần 2)

**Mục tiêu Sprint:** Thêm cơ chế Attention và tối ưu CNN-LSTM

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 6.1 | Viết lớp Attention (temporal attention trên đầu ra LSTM) | B | ☐ |
| 6.2 | Tích hợp Attention vào CNN-LSTM | B | ☐ |
| 6.3 | Điều chỉnh hyperparameter: learning rate, batch size, dropout rate, window size | A + B | ☐ |
| 6.4 | Huấn luyện + đánh giá CNN-LSTM-Attention trên FD001 | B | ☐ |
| 6.5 | Mở rộng sang FD002–FD004 | A + B | ☐ |
| 6.6 | Cập nhật bảng so sánh với kết quả CNN-LSTM-Attention | A | ☐ |
| 6.7 | Tiếp tục viết báo cáo: phần Phương pháp (Data Pipeline + CNN-LSTM) | A | ☐ |

**Sản phẩm:**
- [ ] `src/models/cnn_lstm_attention.py`
- [ ] Kết quả tối ưu trên cả 4 tập
- [ ] Bảng so sánh toàn diện cập nhật
- [ ] Bản nháp báo cáo: phần Phương pháp

**Tiêu chí Hoàn thành:** CNN-LSTM-Attention RMSE trên FD001 ≤ 14.0.

---

## Sprint 7 — Thí nghiệm Transformer

**Mục tiêu Sprint:** Triển khai biến thể Transformer và so sánh với CNN-LSTM

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 7.1 | Viết mô hình Transformer Encoder (positional encoding, multi-head attention) | B | ☐ |
| 7.2 | Huấn luyện Transformer trên FD001 | B | ☐ |
| 7.3 | Điều chỉnh hyperparameter cho Transformer | B | ☐ |
| 7.4 | Đánh giá Transformer trên FD001–FD004 | A | ☐ |
| 7.5 | So sánh Transformer vs CNN-LSTM-Attention (bảng + phân tích) | A + B | ☐ |
| 7.6 | Viết phân tích: tại sao Transformer tốt/kém hơn? | A | ☐ |
| 7.7 | **Cổng quyết định:** Chọn mô hình tốt nhất cho các sprint tiếp theo | A + B | ☐ |

**Sản phẩm:**
- [ ] `src/models/transformer.py`
- [ ] Bảng + phân tích so sánh Transformer vs CNN-LSTM
- [ ] Quyết định: mô hình nào dùng cho độ bất định + giải thích

**Câu hỏi Đánh giá Sprint:**
- Transformer có cạnh tranh không? Ở tập nào tốt/kém?
- RMSE tốt nhất hiện tại là bao nhiêu trên mỗi FD?
- Có cần thử lai Transformer-LSTM không?

---

## Sprint 8 — Nghiên cứu Cắt bỏ (Ablation Study)

**Mục tiêu Sprint:** Chứng minh mỗi thành phần đóng góp vào kết quả cuối

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 8.1 | Cắt bỏ: bỏ khối CNN, chỉ giữ LSTM → ảnh hưởng? | B | ☐ |
| 8.2 | Cắt bỏ: bỏ Attention → ảnh hưởng? | B | ☐ |
| 8.3 | Cắt bỏ: bỏ đặc trưng trượt (Tầng 2) → ảnh hưởng? | A | ☐ |
| 8.4 | Cắt bỏ: bỏ tốc độ thay đổi (Tầng 3) → ảnh hưởng? | A | ☐ |
| 8.5 | Cắt bỏ: thay đổi RUL cap (100, 125, 150) → ảnh hưởng? | A | ☐ |
| 8.6 | Cắt bỏ: thay đổi kích thước cửa sổ (20, 30, 50) → ảnh hưởng? | B | ☐ |
| 8.7 | Tổng hợp kết quả cắt bỏ thành bảng | A + B | ☐ |
| 8.8 | Thử nghiệm học chuyển giao: huấn luyện FD001 → đánh giá FD003 (bonus) | B | ☐ |

**Sản phẩm:**
- [ ] Bảng nghiên cứu cắt bỏ (thành phần × chỉ số × tập dữ liệu)
- [ ] Phân tích: thành phần nào quan trọng nhất?
- [ ] Kết quả học chuyển giao (bonus)

---

## Sprint 9 — Lượng hóa Độ Bất định

**Mục tiêu Sprint:** Triển khai Monte Carlo Dropout cho khoảng dự đoán

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 9.1 | Chỉnh mô hình tốt nhất: bật dropout khi suy luận | B | ☐ |
| 9.2 | Triển khai suy luận MC Dropout: N=100 lần lan truyền thuận | B | ☐ |
| 9.3 | Tính cho từng mẫu: trung bình dự đoán, độ lệch chuẩn, khoảng tin cậy 95% | B | ☐ |
| 9.4 | Đánh giá: % RUL thực nằm trong khoảng tin cậy dự đoán | A | ☐ |
| 9.5 | Trực quan hóa: khoảng dự đoán cho các động cơ được chọn | A | ☐ |
| 9.6 | Phân tích: độ bất định có tương quan với sai số không? | A | ☐ |
| 9.7 | Viết báo cáo: phần Lượng hóa Độ Bất định | A | ☐ |

**Sản phẩm:**
- [ ] Module suy luận MC Dropout
- [ ] Hình phân tích độ bất định (biểu đồ khoảng tin cậy, scatter độ bất định vs sai số)
- [ ] Kết quả hiệu chuẩn: tỷ lệ bao phủ khoảng tin cậy
- [ ] Bản nháp báo cáo: phần Độ Bất định

**Tiêu chí Hoàn thành:** ≥ 85% giá trị RUL thực nằm trong khoảng tin cậy 95% dự đoán.

---

## Sprint 10 — Khả năng Giải thích

**Mục tiêu Sprint:** Giải thích tại sao mô hình đưa ra quyết định

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 10.1 | Trích xuất + trực quan hóa Attention heatmap (tầm quan trọng bước thời gian) | B | ☐ |
| 10.2 | Triển khai phân tích SHAP cho mô hình tốt nhất | A | ☐ |
| 10.3 | Tạo biểu đồ tóm tắt SHAP: xếp hạng tầm quan trọng cảm biến | A | ☐ |
| 10.4 | Nghiên cứu tình huống: chọn 3 động cơ (dự đoán tốt, kém, biên) — phân tích sâu | A + B | ☐ |
| 10.5 | Đối chiếu kết quả SHAP với phát hiện EDA (cảm biến 11, 7, 4 là chí tử?) | A | ☐ |
| 10.6 | Viết báo cáo: phần Khả năng Giải thích | A | ☐ |
| 10.7 | Viết báo cáo: phần Thí nghiệm + Kết quả | A + B | ☐ |

**Sản phẩm:**
- [ ] Hình Attention heatmap
- [ ] Biểu đồ SHAP tóm tắt + giải thích từng mẫu
- [ ] 3 nghiên cứu tình huống
- [ ] Bản nháp báo cáo: Thí nghiệm + Kết quả

---

## Sprint 11 — Viết Báo cáo

**Mục tiêu Sprint:** Hoàn thiện bản nháp báo cáo đầy đủ

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 11.1 | Hoàn thiện Tóm tắt (Abstract) | A + B | ☐ |
| 11.2 | Chỉnh sửa Giới thiệu + Tổng quan Tài liệu | A | ☐ |
| 11.3 | Chỉnh sửa Phương pháp (sơ đồ data pipeline, kiến trúc mô hình) | B | ☐ |
| 11.4 | Hoàn thiện tất cả bảng kết quả và hình | A + B | ☐ |
| 11.5 | Viết Thảo luận: điểm mạnh, hạn chế, so sánh với SOTA | A | ☐ |
| 11.6 | Viết Kết luận + Hướng Phát triển Tương lai | B | ☐ |
| 11.7 | Tạo slide thuyết trình (bản nháp) | B | ☐ |
| 11.8 | Đánh giá chéo: A review phần của B, B review phần của A | A + B | ☐ |

**Sản phẩm:**
- [ ] Bản nháp báo cáo hoàn chỉnh (tất cả phần)
- [ ] Tất cả hình và bảng hoàn thiện
- [ ] Bản nháp slide thuyết trình

---

## Sprint 12 — Hoàn thiện & Nộp bài

**Mục tiêu Sprint:** Hoàn thiện mọi thứ, sẵn sàng nộp

| # | Công việc | Phụ trách | Trạng thái |
|:--|:---------|:----------|:-----------|
| 12.1 | Dọn code: xóa code thừa, thêm docstring | A + B | ☐ |
| 12.2 | Tạo `README.md` cho repository | A | ☐ |
| 12.3 | Xác minh tái lập: clone mới → chạy → kết quả giống | B | ☐ |
| 12.4 | Rà soát báo cáo (ngữ pháp, định dạng, trích dẫn) | A | ☐ |
| 12.5 | Hoàn thiện slide thuyết trình | B | ☐ |
| 12.6 | Tập thuyết trình (2–3 lần) | A + B | ☐ |
| 12.7 | Nộp báo cáo + code + thuyết trình | A + B | ☐ |

**Sản phẩm:**
- [ ] Báo cáo cuối cùng (đã nộp)
- [ ] GitHub repository sạch kèm README
- [ ] Slide thuyết trình (bản cuối)
- [ ] Đã tập thuyết trình

---

## Theo dõi Tốc độ (Velocity Tracking)

| Sprint | Công việc Dự kiến | Hoàn thành | Tốc độ | Ghi chú |
|:-------|:-----------------|:-----------|:-------|:--------|
| 0 | 5 | | | |
| 1 | 6 | | | |
| 2 | 9 | | | |
| 3 | 8 | | | |
| 4 | 8 | | | |
| 5 | 7 | | | |
| 6 | 7 | | | |
| 7 | 7 | | | |
| 8 | 8 | | | |
| 9 | 7 | | | |
| 10 | 7 | | | |
| 11 | 8 | | | |
| 12 | 7 | | | |

---

## Tiêu chí Hoàn thành (Toàn cục)

Một công việc được coi là **Hoàn thành** khi:
1. Code chạy được, không lỗi
2. Kết quả được ghi lại (chỉ số, hình)
3. Code được push lên GitHub
4. Đồng đội đã review (nếu là code quan trọng)

## Mẫu Hồi cứu (Dùng mỗi cuối Sprint)

| Câu hỏi | Trả lời |
|:---------|:--------|
| Tuần này làm tốt nhất điều gì? | |
| Tuần này gặp khó khăn gì? | |
| Tuần tới cần thay đổi gì? | |
| Có blocker nào cần giải quyết? | |
