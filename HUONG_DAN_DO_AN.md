# Đồ án phân loại sự hiện diện của bệnh tim

## Chạy dự án

Chạy lệnh từ thư mục `DoAn` với Python 3.10 (môi trường đã kiểm chứng). PowerShell:

```powershell
python -m pip install -r requirements.txt
python benchmark.py
python -m streamlit run app.py
```

Sau khi clone, chạy `python benchmark.py` để tạo biểu đồ trong `results/charts/`, mô hình trong `results/models/` và các bảng kết quả trong `results/`. Kho mã giữ `Data/heart.csv` và `results/label_verification.json` để bước huấn luyện có đủ đầu vào. Sau khi mở ứng dụng, truy cập địa chỉ Streamlit in ra, thường là `http://localhost:8501`. Nếu cổng bận, thêm `--server.port 8502`.

## Cấu trúc và sản phẩm

- `Data/heart.csv`: dữ liệu gốc giữ nguyên. Chỉ tệp này tham gia nghiên cứu.
- `heart_pipeline.py`: quy tắc dữ liệu, tiền xử lý, CV, tìm tham số, đánh giá, lưu mô hình.
- `benchmark.py`: đầu vào huấn luyện cả ba thuật toán. Các lệnh `Dtree_heart.py`, `Bayes_heart.py`, `RF_heart.py` chuyển sang cùng quy trình chung, tránh sinh kết quả không nhất quán.
- `app.py`: hai tab nhập liệu và đánh giá, tải Pipeline, không huấn luyện khi nhập.
- `results/`: kết quả mới duy nhất mà ứng dụng và Slide 8–13 sử dụng.
- `results/models/*.joblib`: ba Pipeline, bao gồm đổi mã thiếu, thứ tự cột, imputer, encoder, scaler khi cần và bộ phân loại.
- `results/metrics.csv`, `cross_validation.csv`, `experiment.json`: Test, CV trung bình/SD và cấu hình baseline/tối ưu.
- `results/*_search.csv`: điểm từng fold của toàn bộ ứng viên GridSearchCV.
- `results/data_audit.json`, `label_verification.json`: kiểm toán dữ liệu và bằng chứng nhãn.
- `results/charts/`: EDA, ba confusion matrix, so sánh và permutation importance độ phân giải 200 dpi.
- `results/train.csv`, `test.csv`, `heart_clean.csv`: dữ liệu đã xử lý với `source_row` là chỉ số dòng gốc, đếm từ 0 (không tính header).
- `results/demo_samples.json`: hai mẫu Test dùng khi thuyết trình.
- `TaiLieu/DoAn_DuDoanBenhTim_HoanChinh.pptx`: bài PowerPoint cập nhật Slide 8–13, có Speaker Notes.
- `TaiLieu/LOI_THUYET_TRINH_SLIDE_08_13.md`: lời nói, câu hỏi và chuyển ý.
- `backups/original_20261105/`: bản sao các mã gốc, CSV và PowerPoint trước sửa.
- `tools/test_project.py`: kiểm thử tích hợp. `results/test_report.json` lưu kết quả.
- `.build/`: mã dựng, bản xem trước, dữ liệu đối chiếu nguồn và hồ sơ kiểm tra PowerPoint. Đây là tài nguyên kỹ thuật, không phải dữ liệu nghiên cứu bổ sung.

Các thư mục `metrics/`, `models/`, `Bayes/`, `Dtree/`, `RF/` và tài liệu Markdown cũ được giữ để đối chiếu lịch sử. Chúng có thể chứa số liệu hoặc nhãn trước sửa, không dùng trong kết quả cuối. `Data/cleveland.csv` giữ nguyên, chương trình huấn luyện và demo không đọc tệp này.

## Dữ liệu và nhãn đã xác minh

Heart có 1.025 dòng, 14 cột, 13 biến đầu vào, 723 dòng trùng hoàn toàn và 302 dòng duy nhất. Không có nhóm đặc trưng giống nhau nhưng khác nhãn, không có NaN/chuỗi thiếu trong tệp gốc. Có 4 dòng duy nhất `ca=4`, 2 dòng `thal=0`, chuyển thành thiếu rồi điền trong Pipeline. Giữ các giá trị cực trị khả dĩ như cholesterol 564 mg/dL, không loại theo nhãn.

**Nhãn gốc bị đảo so với quy ước nghiên cứu:** `target=0` là có bệnh, `target=1` là không bệnh. Dùng `y=1-target` để chuẩn hóa `1=có bệnh`, `0=không bệnh`.

Đã ghép một-một cả 302 bản ghi Heart bằng 8 biến không đổi với dữ liệu gốc công khai UCI. Cả 302/302 nhãn chuẩn hóa khớp `num > 0`. Dữ liệu tham chiếu chỉ dùng xác minh ngữ nghĩa, tuyệt đối không dùng huấn luyện, CV, Test hoặc tuyên bố kiểm định độc lập. Hash, khóa ghép và bảng ánh xạ lưu trong `results/label_verification.json`.

Nguồn chính thức:

- [Tài liệu thuộc tính UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)
- [Bản ghi nguồn UCI để đối chiếu](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)

`python tools/verify_labels.py` chạy lại đối chiếu. Hai tệp nguồn đã lưu trong `.build/provenance/` để tái lập ngoại tuyến. Lần tải hiện tại qua HTTP chính thức do chứng chỉ HTTPS nguồn UCI hết hạn; nguồn metadata còn được kiểm tra qua công cụ web. Không tắt kiểm tra TLS trong mã.

| Cột Heart | Ý nghĩa đúng trong phiên bản này |
|---|---|
| cp | 0 không triệu chứng; 1 đau thắt ngực không điển hình; 2 đau ngực không do thắt ngực; 3 điển hình |
| restecg | 0 phì đại thất trái; 1 bình thường; 2 bất thường ST-T |
| slope | 0 xuống; 1 ngang; 2 lên |
| thal | 0 thiếu/không xác định; 1 khiếm khuyết cố định; 2 bình thường; 3 hồi phục |
| ca | 0–3 mạch; 4 ngoài phạm vi hợp lệ, coi là thiếu |

Chú ý: không diễn giải `ca=4` là bốn mạch. Một số mã 4 không trùng hoàn toàn với vị trí thiếu ở nguồn gốc, nên đây là chính sách xử lý mã không hợp lệ, không khẳng định đã phục hồi giá trị thật.

## Quy trình thực nghiệm và kết quả

Loại trùng trước chia phân tầng 70/30, seed 42: Train 211 (115 không bệnh, 96 có bệnh), Test 91 (49 không bệnh, 42 có bệnh). Không có dòng đặc trưng giống nhau giữa hai tập. Năm biến số điền median, ba biến nhị phân điền mode, năm biến danh mục điền mode và One-Hot. Gaussian NB thêm StandardScaler cho biến số. Tất cả phép fit nằm trong Pipeline trên Train hoặc fold Train.

Chọn trước tiêu chí **F1 trung bình CV trên Train**. GridSearchCV thử 32 cấu hình DT, 48 RF, 6 Gaussian NB với Stratified 5-Fold (shuffle, seed 42). Test chỉ đánh giá sau khi cố định mô hình. Các điểm CV sau tìm kiếm dùng lại fold, có thể lạc quan; không phải nested CV.

| Mô hình tối ưu | Accuracy Test | Precision bệnh | Recall bệnh | F1 bệnh | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0,758 | 0,778 | 0,667 | 0,718 | 0,743 |
| Random Forest | 0,813 | 0,805 | 0,786 | 0,795 | 0,885 |
| Gaussian NB | 0,813 | 0,838 | 0,738 | 0,785 | 0,906 |

RF được chọn vì CV F1 0,831 ± 0,058, không phải vì mặc định rừng tốt hơn. DT đạt 0,725 ± 0,032, NB đạt 0,808 ± 0,049. Test RF: TN=41, FP=8, FN=9, TP=33. NB có Precision và AUC cao hơn RF, nhưng Recall thấp hơn. Không kết luận khác biệt có ý nghĩa thống kê từ một lần chia nhỏ.

Permutation importance hoán vị từng **cột gốc** qua Pipeline trên Test, 30 lần, đo giảm F1. Ba cột đứng đầu là `ca`, `cp`, `thal`. CSV có SD, biểu đồ ứng dụng có thanh sai số; biểu đồ slide trình bày trung bình cho gọn. Phân tích sau đánh giá, không dùng chọn lại biến hay mô hình. Không suy ra nhân quả y khoa.

## Demo và kiểm thử

Chọn RF và mẫu có bệnh, dòng nguồn 331: nhãn thật 1, dự đoán 1, điểm lớp bệnh 71,854%. Mẫu không bệnh, dòng 158: nhãn thật 0, dự đoán 0, điểm 43,635%. Ảnh trong `results/demo_streamlit.png` và `demo_detail.png` chụp ứng dụng thật. Không xem hai ví dụ này là bằng chứng độ chính xác chung.

```powershell
python tools/test_project.py
```

Kiểm thử kiểm tra số dòng, nhãn và hash dữ liệu gốc, Train/Test không trùng, median imputer chỉ từ Train, tải lại mô hình, tính lại đủ chỉ số, mã thiếu, thứ tự cột, cả ba mô hình trên UI, hai mẫu và đổi đầu vào. `results/browser_test.json` ghi kiểm thử trình duyệt và `results/ppt_preservation.json` ghi các phần PowerPoint được giữ nguyên. Ứng dụng đọc điểm xác suất theo vị trí lớp 1 trong `classes_`, không gán cứng phần tử thứ hai.

## Giới hạn và lưu ý bài trình chiếu

File PowerPoint gốc có 15 slide nhưng Slide 2–14 là khung trống. Chỉ điền Slide 8–13 theo phạm vi được yêu cầu. Slide 1–7, 14–15 giữ nguyên, kể cả thông tin mẫu ở Slide 15. Vì vậy, nhóm còn cần hoàn thiện phần ngoài phạm vi này trước khi thuyết trình toàn bộ bài. Bản gốc `TaiLieu/DoAN_TTNT.pptx` không bị ghi đè.

PowerPoint đã nhập lại bằng Artifact Tool, dựng đủ 15 ảnh xem trước trong `results/slide_previews/` và qua kiểm tra cấu trúc gói (`results/ppt_integrity.json`). Đã xem trực quan các slide cập nhật và sửa chỗ chữ sát trang trí. Bốn biểu đồ có bảng Excel nhúng để sửa dữ liệu. Các giá trị vẽ biểu đồ làm tròn 6 chữ số thập phân, bảng slide làm tròn 3, CSV giữ đầy đủ độ chính xác.

Bộ finalizer đã kiểm tra bảng, biểu đồ, workbook và hình học trên bản sao bỏ riêng 3 liên kết metadata `customXml` vì bộ kiểm tra biểu đồ không chấp nhận đích hợp lệ nằm ngoài `ppt/`. Bản bàn giao khôi phục nguyên vẹn các liên kết metadata gốc, giữ nguyên mọi phần nội dung đã kiểm tra, rồi kiểm tra cấu trúc và nhập/dựng lại lần nữa. Biên nhận kỹ thuật nằm ở `.build/final_validation.json`. Chưa mở kiểm tra trực tiếp trong ứng dụng Microsoft PowerPoint; ảnh xem trước do công cụ dựng slide tạo, không phải ảnh chụp PowerPoint.

Dữ liệu ít, không có định danh bệnh nhân, chưa kiểm định ngoài, chưa hiệu chỉnh xác suất hay tối ưu ngưỡng. Gaussian NB dùng One-Hot với giả định độc lập có hạn chế. Xếp hạng importance và chỉ số một lần chia có độ bất định. Chỉ phân loại sự hiện diện của bệnh theo dữ liệu nghiên cứu, không dự báo nguy cơ bệnh trong tương lai, không thay thế thăm khám.
