# TÀI LIỆU BÁO CÁO & KỊCH BẢN THUYẾT TRÌNH ĐỒ ÁN MÔN HỌC
**MÔN HỌC:** TRÍ TUỆ NHÂN TẠO  
**ĐỀ TÀI:** ỨNG DỤNG CÁC MÔ HÌNH MÁY HỌC TRONG CẢNH BÁO NGUY CƠ BỆNH TIM: NGHIÊN CỨU THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐA BỘ DỮ LIỆU  

---

## CẤU TRÚC CHI TIẾT 10 SLIDES & LỜI THOẠI THUYẾT TRÌNH (SPEAKER NOTES)

---

### SLIDE 1: TRANG TIÊU ĐỀ (TITLE)

#### 1. Nội dung hiển thị trên Slide:
* **Tên đề tài:** ỨNG DỤNG CÁC MÔ HÌNH MÁY HỌC TRONG CẢNH BÁO NGUY CƠ BỆNH TIM: NGHIÊN CỨU THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐA BỘ DỮ LIỆU
* **Học phần:** Trí tuệ Nhân tạo (Artificial Intelligence)
* **Giảng viên hướng dẫn:** [Điền tên Giảng viên]
* **Nhóm sinh viên thực hiện:**
  * Thành viên 1: [Võ Văn Tỷ - DTH235811] 
  * Thành viên 2: [Lê Tuấn Anh - DTH235607] 
  * Thành viên 3: [Phạm Dương Lệ Duyên - DTH235634] 
  * Thành viên 4: [Nguyễn Phú Quý - DTH235]

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Kính thưa Thầy và toàn thể các bạn sinh viên. Hôm nay, nhóm chúng em xin phép được đại diện trình bày đề tài nghiên cứu đồ án môn học Trí tuệ Nhân tạo với chủ đề: **'Ứng dụng các mô hình máy học trong cảnh báo nguy cơ bệnh tim: Nghiên cứu thực nghiệm và đánh giá đa bộ dữ liệu'**. Đề tài của nhóm không chỉ dừng lại ở việc áp dụng một thuật toán đơn lẻ trên một tập dữ liệu đóng, mà hướng đến việc phân tích đối sánh chuyên sâu giữa ba mô hình học máy kinh điển là Decision Tree, Naive Bayes và Random Forest trên hai bộ dữ liệu y khoa độc lập, từ đó triển khai ứng dụng web cảnh báo sớm trong thực tế. Sau đây, em xin kính mời Thầy và các bạn cùng theo dõi phần trình bày chi tiết của nhóm."*

---

### SLIDE 2: ĐẶT VẤN ĐỀ & MỤC TIÊU NGHIÊN CỨU

#### 1. Nội dung hiển thị trên Slide:
* **Tính cấp thiết & Bối cảnh:**
  * Bệnh tim mạch (Cardiovascular Diseases - CVDs) là nguyên nhân gây tử vong hàng đầu trên toàn cầu theo Tổ chức Y tế Thế giới (WHO).
  * Chẩn đoán lâm sàng truyền thống đòi hỏi chi phí lớn và thời gian xét nghiệm chuyên sâu; việc sàng lọc sớm dựa trên các chỉ số cận lâm sàng định kỳ có ý nghĩa sống còn.
* **Mục tiêu nghiên cứu:**
  * **Mục tiêu 1:** Nghiên cứu và huấn luyện 3 thuật toán máy học: Decision Tree, Gaussian Naive Bayes và Random Forest.
  * **Mục tiêu 2 (Trọng tâm):** Đánh giá độ ổn định và tính tổng quát hóa của các thuật toán khi thực nghiệm trên 2 quy mô dữ liệu khác nhau (Tập 303 mẫu gốc vs Tập 1.025 mẫu mở rộng).
  * **Mục tiêu 3:** Xây dựng giải pháp Web Demo tương tác (Streamlit) hỗ trợ chuyển đổi linh hoạt mô hình theo bộ dữ liệu để ứng dụng vào cảnh báo sớm.

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Thưa Thầy, theo thống kê y tế, bệnh tim mạch là mối đe dọa hàng đầu đối với sức khỏe con người, cướp đi hàng triệu sinh mạng mỗi năm. Thách thức lớn nhất hiện nay là triệu chứng bệnh tim thường diễn biến âm thầm, khiến bệnh nhân chỉ phát hiện khi bệnh đã ở giai đoạn nặng. Việc ứng dụng Trí tuệ Nhân tạo nhằm phân tích các chỉ số sinh hóa và triệu chứng ban đầu để sàng lọc nguy cơ là một hướng tiếp cận vô cùng cấp thiết.*  
>  
> *Xuất phát từ thực tiễn đó, nhóm đặt ra ba mục tiêu rõ ràng: Thứ nhất, làm chủ và thực thi ba thuật toán cốt lõi. Thứ hai, trả lời câu hỏi khoa học quan trọng: **Liệu một mô hình hoạt động tốt trên bộ dữ liệu nhỏ có tiếp tục duy trì được độ chính xác và tính bền vững khi số lượng mẫu tăng lên gấp nhiều lần hay không?** Và thứ ba, hiện thực hóa các mô hình này thành một công cụ web trực quan, cho phép người dùng nhập thông tin và nhận kết quả chẩn đoán tức thì."*

---

### SLIDE 3: TỔNG QUAN 2 BỘ DỮ LIỆU THỰC NGHIỆM (DATASETS)

#### 1. Nội dung hiển thị trên Slide:
* **Bảng so sánh 2 nguồn dữ liệu:**

| Tiêu chí | Dataset 1 (Kaggle Heart Disease) | Dataset 2 (UCI Cleveland Gốc) |
| :--- | :--- | :--- |
| **Quy mô mẫu** | 1.025 dòng dữ liệu | 303 dòng dữ liệu |
| **Số lượng thuộc tính** | 13 biến đặc trưng + 1 nhãn mục tiêu | 13 biến đặc trưng + 1 nhãn mục tiêu |
| **Nguồn gốc** | Bộ dữ liệu mở rộng từ Kaggle | Bộ chuẩn kinh điển từ UCI ML Repository |
| **Đặc điểm dữ liệu** | Đã được chuẩn hóa nhãn nhị phân (0/1) | Dữ liệu gốc có chứa missing values (dấu `?`) và nhãn phân độ (0 đến 4) |
| **Ý nghĩa nghiên cứu** | Kiểm tra độ vững và khả năng học sâu của mô hình trên dữ liệu quy mô lớn | Thử nghiệm độ nhạy và tính ổn định trên tập dữ liệu y tế gốc hạn chế mẫu |

* **13 thuộc tính lâm sàng dùng chung:**
  * `age` (tuổi), `sex` (giới tính), `cp` (loại đau thắt ngực), `trestbps` (huyết áp nghỉ), `chol` (cholesterol), `fbs` (đường huyết đói), `restecg` (điện tâm đồ nghỉ), `thalach` (nhịp tim tối đa), `exang` (đau ngực khi gắng sức), `oldpeak` (độ chênh ST), `slope` (độ dốc ST), `ca` (số mạch máu lớn), `thal` (xét nghiệm thallium).
  * **Biến mục tiêu (`target`):** `0` - Bình thường (Không bệnh) | `1` - Có nguy cơ bệnh tim.

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Để đảm bảo tính khách quan và khoa học, nhóm quyết định không chỉ kiểm nghiệm trên một tập dữ liệu duy nhất mà sử dụng đồng thời hai bộ dữ liệu y khoa chuẩn mực.*  
>  
> *Bộ dữ liệu thứ nhất lấy từ Kaggle với quy mô 1.025 bản ghi, đại diện cho tập dữ liệu mở rộng giúp mô hình học được nhiều biến thể lâm sàng hơn. Bộ thứ hai là cơ sở dữ liệu gốc UCI Cleveland với 303 bệnh nhân, vốn là bộ dữ liệu chuẩn mực được trích dẫn nhiều nhất trong các nghiên cứu học thuật về tim mạch. Điểm đặc biệt là cả hai bộ đều cùng đo lường 13 chỉ số sinh học quan trọng như huyết áp, lượng cholesterol, điện tâm đồ và kết quả gắng sức. Việc giữ nguyên không gian đặc trưng giữa hai bộ giúp nhóm có được cơ sở công bằng để so sánh chéo hiệu năng thuật toán."*

---

### SLIDE 4: QUY TRÌNH PIPELINE & TIỀN XỬ LÝ DỮ LIỆU

#### 1. Nội dung hiển thị trên Slide:
* **Sơ đồ kiến trúc Pipeline:**
  $$\text{Dữ liệu thô} \xrightarrow{\text{Tiền xử lý & Làm sạch}} \text{Phân chia Train/Test (70/30)} \xrightarrow{\text{Huấn luyện 3 Mô hình}} \text{Đánh giá Đối sánh} \xrightarrow{\text{Xuất Artifacts (.pkl)}}$$
* **Các kỹ thuật tiền xử lý cốt lõi:**
  * **Xử lý dữ liệu khuyết thiếu (Missing Values):** Tập Cleveland gốc có 6 mẫu bị khuyết ở thuộc tính `ca` và `thal` (dạng `?`). Nhóm áp dụng phương pháp gán giá trị theo Mode (giá trị xuất hiện nhiều nhất) để bảo toàn cỡ mẫu 303.
  * **Chuẩn hóa nhãn phân loại (`target`):** Tập Cleveland gốc có 5 cấp độ bệnh (0 đến 4). Nhóm chuyển đổi về bài toán nhị phân chuẩn: nhãn `0` là khỏe mạnh, các mức từ `1 đến 4` quy về nhãn `1` (có bệnh tim).
  * **Mã hóa thuộc tính:** Sử dụng `LabelEncoder` cho các biến danh mục.
  * **Chiến lược phân chia dữ liệu:** Tách tập Train (70%) và Test (30%) với `random_state=42` cố định để đảm bảo mọi kết quả thực nghiệm đều có tính tái lập (reproducibility).

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Thưa Thầy, một mô hình máy học chỉ có thể đưa ra dự đoán chính xác khi đầu vào được làm sạch đúng quy chuẩn y tế. Pipeline của nhóm được thiết kế khép kín gồm 5 giai đoạn: từ thu nạp, làm sạch, phân chia tập mẫu, huấn luyện đồng thời và đóng gói mô hình.*  
>  
> *Đặc biệt, ở tập Cleveland gốc, dữ liệu thô tồn tại các giá trị khuyết ở thuộc tính mạch máu lớn và xét nghiệm máu thallium. Nhóm đã xử lý bằng phương pháp Mode Imputation nhằm tránh làm mất mát dữ liệu vốn đã rất nhỏ. Đồng thời, nhóm chuẩn hóa nhãn từ đa phân lớp thành bài toán phân loại nhị phân để thống nhất cấu trúc đầu ra với tập Kaggle. Mọi phép chia tập dữ liệu đều được cố định ngẫu nhiên để đảm bảo tính khách quan tuyệt đối."*

---

### SLIDE 5: CƠ SỞ LÝ THUYẾT CÁC THUẬT TOÁN MÁY HỌC

#### 1. Nội dung hiển thị trên Slide:
* **1. Cây quyết định (Decision Tree - Tiêu chuẩn Entropy):**
  * Xây dựng cấu trúc cây phân nhánh dựa trên độ giảm độ hỗn loạn thông tin (Information Gain).
  * **Ưu điểm:** Khả năng giải thích cao (White-box model), mô phỏng quy trình chẩn đoán loại suy của bác sĩ.
  * **Nhược điểm:** Dễ gặp hiện tượng học vẹt (Overfitting) trên cây đơn lẻ.
* **2. Naive Bayes (Gaussian Naive Bayes):**
  * Dựa trên định lý xác suất Bayes với giả định ngây thơ: Các thuộc tính hoàn toàn độc lập với nhau có điều kiện.
  * **Ưu điểm:** Tốc độ tính toán tức thì, hoạt động ổn định khi mẫu ít.
  * **Nhược điểm:** Giả định độc lập không hoàn toàn khớp với y khoa (ví dụ: huyết áp và cholesterol thường có tương quan).
* **3. Rừng ngẫu nhiên (Random Forest - 100 cây con):**
  * Áp dụng kỹ thuật học kết hợp (Ensemble Learning - Bagging). Tạo 100 cây quyết định độc lập trên các tập con mẫu và đặc trưng ngẫu nhiên, sau đó biểu quyết đa số.
  * **Ưu điểm:** Khắc phục triệt để overfitting của Decision Tree đơn lẻ, cho độ chính xác vượt trội và cung cấp thước đo tầm quan trọng đặc trưng (Feature Importance).

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Nhóm lựa chọn ba thuật toán đại diện cho ba trường phái học máy kinh điển:*  
> *Đầu tiên là Decision Tree – thuật toán có ưu điểm lớn nhất là tính minh bạch. Bác sĩ hoàn toàn có thể nhìn vào sơ đồ cây để hiểu vì sao hệ thống đưa ra phán đoán đó.*  
> *Thứ hai là Naive Bayes – mô hình dựa trên lý thuyết xác suất thống kê thuần túy, nổi tiếng với sự nhanh nhẹn và hiệu quả trên các tập dữ liệu nhỏ.*  
> *Và cuối cùng là Random Forest – giải pháp học kết hợp thông minh. Bằng cách kết hợp quyết định từ 100 cây con độc lập thông qua cơ chế biểu quyết đa số, Random Forest triệt tiêu sai số phương sai của từng cây đơn lẻ, mang lại độ tin cậy vượt trội trong các bài toán chẩn đoán có độ rủi ro cao."*

---

### SLIDE 6: KẾT QUẢ THỰC NGHIỆM TRÊN DATASET 1 (KAGGLE - 1.025 MẪU)

#### 1. Nội dung hiển thị trên Slide:
* **Bảng kết quả đối sánh (Tập kiểm thử 308 mẫu):**

| Thuật toán | Accuracy (%) | Precision Lớp 1 (Bệnh tim) | Recall Lớp 1 (Bệnh tim) | F1-Score Lớp 1 |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 81.17 % | 0.75 | 0.92 | 0.83 |
| **Naive Bayes** | 81.49 % | 0.76 | 0.89 | 0.82 |
| **Random Forest (100 cây)** | **88.96 %** | **0.84** | **0.95** | **0.89** |

* **Hình ảnh minh họa cần chèn vào Slide:**
  * Chèn ảnh: `metrics/Dataset_1_Kaggle/charts/2_so_sanh_hieu_nang.png` (Biểu đồ cột hiệu năng).
  * Chèn ảnh: `metrics/Dataset_1_Kaggle/charts/3_ma_tran_nham_lan.png` (Ma trận nhầm lẫn 3 mô hình).
* **Nhận xét then chốt:**
  * **Random Forest xuất sắc toàn diện:** Đạt độ chính xác cao nhất (**88.96%**), tăng gần 8% so với 2 thuật toán còn lại.
  * **Độ nhạy (Recall) đạt 95%:** Trong bài toán bệnh tim, bỏ sót ca bệnh (False Negative) là điều cấm kỵ. Random Forest chỉ bỏ sót 7 ca trong tổng số 149 bệnh nhân thực tế, chứng minh tính an toàn y khoa rất cao.

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Bước vào phần kết quả thực nghiệm trên tập Kaggle 1.025 mẫu, bảng số liệu trên màn hình đã chứng minh sự vượt trội rõ rệt của Random Forest. Mô hình đạt độ chính xác ấn tượng là **88.96%** so với mức xấp xỉ 81% của Decision Tree và Naive Bayes.*  
>  
> *Tuy nhiên, chỉ số mà nhóm chúng em tự hào nhất không phải là Accuracy mà chính là **Recall lớp bệnh tim đạt tới 0.95 (95%)**. Trong y tế, chẩn đoán nhầm người khỏe thành người bệnh chỉ gây tốn thêm xét nghiệm, nhưng nếu bỏ sót một người bệnh tim (False Negative) thì bệnh nhân có thể nguy hiểm đến tính mạng vì không được điều trị kịp thời. Với việc bắt trúng 95% bệnh nhân thực tế, Random Forest chứng minh đây là mô hình sàng lọc y tế vô cùng tin cậy."*

---

### SLIDE 7: KẾT QUẢ THỰC NGHIỆM TRÊN DATASET 2 (UCI CLEVELAND - 303 MẪU)

#### 1. Nội dung hiển thị trên Slide:
* **Bảng kết quả đối sánh (Tập kiểm thử 91 mẫu):**

| Thuật toán | Accuracy (%) | Precision Lớp 1 (Bệnh tim) | Recall Lớp 1 (Bệnh tim) | F1-Score Lớp 1 |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 78.02 % | 0.86 | 0.63 | 0.73 |
| **Naive Bayes** | 79.12 % | 0.77 | 0.81 | 0.79 |
| **Random Forest (100 cây)** | **86.81 %** | **0.86** | **0.86** | **0.86** |

* **Hình ảnh minh họa cần chèn vào Slide:**
  * Chèn ảnh: `metrics/Dataset_2_Cleveland/charts/2_so_sanh_hieu_nang.png`.
  * Chèn ảnh: `metrics/Dataset_2_Cleveland/charts/3_ma_tran_nham_lan.png`.
* **Nhận xét then chốt:**
  * Khi cỡ mẫu giảm từ 1.025 xuống 303 dòng, **Decision Tree đơn lẻ bộc lộ điểm yếu lớn nhất**: Recall sụt giảm nghiêm trọng xuống chỉ còn **63%** (bỏ sót gần 40% người có bệnh).
  * **Random Forest vẫn giữ vững vị thế số 1:** Đạt Accuracy **86.81%** và Recall cân bằng **86%**, chứng minh khả năng tổng quát hóa không bị phụ thuộc vào việc tập dữ liệu lớn hay nhỏ.

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Để kiểm tra xem Random Forest có thực sự mạnh hay chỉ do tập dữ liệu Kaggle lớn, nhóm đưa mô hình vào thử thách trên tập dữ liệu gốc Cleveland chỉ vỏn vẹn 303 bệnh nhân.*  
>  
> *Kết quả cho thấy một phát hiện rất giá trị: Khi dữ liệu ít, Decision Tree đơn lẻ bị hụt hơi nghiêm trọng, độ nhạy Recall tụt dốc xuống chỉ còn 63%, tức là cứ 10 người bị bệnh tim thì cây đơn lẻ bỏ sót gần 4 người. Trong khi đó, Random Forest vẫn xuất sắc duy trì độ chính xác ở mức **86.81%** và Recall đạt **86%**. Điều này khẳng định cơ chế Ensemble Learning giúp Random Forest có khả năng chống chịu nhiễu và tổng quát hóa cực kỳ bền bỉ ngay cả trong điều kiện thiếu thốn dữ liệu y tế."*

---

### SLIDE 8: ĐÁNH GIÁ ĐỐI SÁNH CHÉO (CROSS-BENCHMARK) & THUỘC TÍNH QUAN TRỌNG

#### 1. Nội dung hiển thị trên Slide:
* **1. Tổng hợp Đối sánh chéo (Cross-Dataset Benchmark):**
  * Chèn ảnh: `metrics/Cross_Benchmark/bieu_do_so_sanh_2_bo_du_lieu.png` (Biểu đồ so sánh cột đôi giữa Kaggle vs Cleveland).
  * **Quy luật:** Random Forest luôn đạt đỉnh hiệu năng trên mọi tập dữ liệu ($\approx 87\% - 89\%$). Khi dữ liệu tăng, hiệu năng các mô hình đều tăng theo một tỷ lệ tịnh tiến đồng đều.
* **2. Phân tích Độ quan trọng của các thuộc tính (Feature Importance):**
  * Chèn ảnh: `metrics/Dataset_1_Kaggle/charts/4_do_quan_trong_dac_trung.png`.
  * **Top 5 thuộc tính quyết định nhiều nhất đến nguy cơ bệnh tim:**
    1. **`cp` (Loại đau thắt ngực):** Chiếm **17.01%** trọng số quyết định.
    2. **`thal` (Tình trạng tan máu bẩm sinh Thalassemia):** Chiếm **14.69%**.
    3. **`ca` (Số mạch máu lớn cản quang):** Chiếm **13.05%**.
    4. **`oldpeak` (Độ chênh ST khi gắng sức):** Chiếm **12.66%**.
    5. **`thalach` (Nhịp tim tối đa đạt được):** Chiếm **11.85%**.
  * $\rightarrow$ **Top 5 chỉ số này chiếm tới gần 70%** toàn bộ quyết định của mô hình; các chỉ số như `fbs` (đường huyết lúc đói) có ảnh hưởng không đáng kể ($< 0.5\%$).

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Trên Slide số 8 là biểu đồ đối sánh chéo đa tập dữ liệu – đây là bức tranh toàn cảnh của đồ án. Nhìn vào biểu đồ cột, chúng ta thấy một đường đồng dạng: Dù là trên tập dữ liệu mở rộng hay tập dữ liệu y tế gốc, thứ tự hiệu năng luôn là **Random Forest dẫn đầu, kế đến là Naive Bayes và cuối cùng là Decision Tree**.*  
>  
> *Đồng thời, để trả lời câu hỏi cốt lõi: 'Yếu tố lâm sàng nào quyết định nhiều nhất đến bệnh tim?', nhóm đã trích xuất Feature Importance từ Random Forest. Kết quả cho thấy hơn 69% quyết định chẩn đoán phụ thuộc vào 5 chỉ số hàng đầu: đặc biệt là triệu chứng đau ngực `cp` (17%), xét nghiệm Thallium (14.7%) và tình trạng tắc nghẽn mạch máu `ca` (13%). Kết quả học máy này hoàn toàn trùng khớp với thực hành lâm sàng trong tim mạch học hiện đại."*

---

### SLIDE 9: ỨNG DỤNG WEB DEMO ĐA BỘ DỮ LIỆU (STREAMLIT DEMO)

#### 1. Nội dung hiển thị trên Slide:
* **Hình ảnh minh họa:** Chèn ảnh chụp màn hình giao diện ứng dụng web Streamlit gồm 2 phần:
  * Ảnh 1: Thanh Sidebar lựa chọn Dataset (Dataset 1 Kaggle vs Dataset 2 Cleveland) và Form nhập 13 chỉ số.
  * Ảnh 2: Thẻ kết quả chẩn đoán cảnh báo đỏ/xanh kèm xác suất (%) và Tab báo cáo biểu đồ đối sánh.
* **Các tính năng nổi bật của ứng dụng:**
  * **Kiến trúc đa mô hình (Multi-model Engine):** Người dùng có thể linh hoạt chọn mô hình huấn luyện từ Dataset 1 hoặc Dataset 2 trên thanh công cụ Sidebar.
  * **Giao diện chuẩn y khoa:** Các ô lựa chọn có chú thích lâm sàng bằng tiếng Việt rõ ràng (thay vì nhập số thô `0, 1, 2, 3`).
  * **Cảnh báo tức thì & Định lượng rủi ro:** Đưa ra kết luận có nguy cơ hay bình thường cùng độ tin cậy phần trăm và lời khuyên y tế.
  * **Dashboard tích hợp:** Tích hợp sẵn tab xem toàn bộ bảng số liệu Excel và biểu đồ ma trận nhầm lẫn ngay trên web.

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Để đưa kết quả nghiên cứu từ mô hình toán học vào đời sống, nhóm đã triển khai ứng dụng web demo bằng framework Streamlit.*  
>  
> *Điểm sáng tạo của ứng dụng là ở thanh Sidebar bên trái: Thầy và các bạn có thể tùy chọn áp dụng mô hình được huấn luyện từ tập Kaggle hay tập Cleveland. Form nhập liệu được thiết kế trực quan với 13 trường dữ liệu được giải thích cặn kẽ theo thuật ngữ y tế. Khi nhấn nút dự đoán, mô hình sẽ phản hồi tức thì với tỷ lệ xác suất nguy cơ rõ ràng, kèm theo cảnh báo màu đỏ nếu phát hiện dấu hiệu bệnh tim. Ứng dụng cũng tích hợp sẵn dashboard đối sánh để người dùng có thể tra cứu minh chứng khoa học ngay trên giao diện."*

---

### SLIDE 10: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

#### 1. Nội dung hiển thị trên Slide:
* **Kết luận đạt được:**
  * Hoàn thành trọn vẹn việc khảo sát, thực nghiệm và đối sánh 3 mô hình học máy trên cả 2 tập dữ liệu tim mạch độc lập.
  * Khẳng định tính ưu việt vượt trội của thuật toán **Random Forest** (Accuracy $\approx 87\% - 89\%$, Recall $\ge 86\% - 95\%$) trong việc giải quyết bài toán chẩn đoán có rủi ro cao.
  * Xác định chính xác nhóm 5 thuộc tính y khoa then chốt chi phối nguy cơ bệnh tim mạch.
  * Xây dựng thành công ứng dụng Web Demo tương tác, sẵn sàng ứng dụng thực tế.
* **Hướng phát triển tương lai:**
  * Thử nghiệm thêm các thuật toán Gradient Boosting hiện đại (XGBoost, LightGBM, CatBoost).
  * Tích hợp kỹ thuật K-Fold Cross Validation lặp lại và cân bằng dữ liệu (SMOTE).
  * Kết nối cơ sở dữ liệu bệnh viện thực tế tại Việt Nam để nâng cao tính ứng dụng địa phương hóa.
* **Lời cảm ơn & Q&A:** Nhóm xin chân thành cảm ơn Thầy và các bạn đã lắng nghe. Mời Thầy và hội đồng đặt câu hỏi.

#### 2. Kịch bản thuyết trình (Văn xuôi):
> *"Kính thưa Thầy và các bạn, qua quá trình nghiên cứu đồ án, nhóm đã hoàn thành đầy đủ tất cả các mục tiêu đề ra. Nghiên cứu đã chứng minh một cách khoa học rằng: Random Forest chính là thuật toán tối ưu nhất cho bài toán sàng lọc nguy cơ bệnh tim nhờ độ chính xác cao và khả năng bảo toàn độ nhạy Recall xuất sắc trên cả hai quy mô dữ liệu.*  
>  
> *Trong tương lai, nhóm dự định sẽ tiếp tục mở rộng đề tài với các thuật toán Boosting nâng cao như XGBoost và thử nghiệm trên các bộ dữ liệu bệnh nhân thực tế tại các bệnh viện Việt Nam.  
>  
> Nhóm chúng em xin gửi lời cảm ơn chân thành đến Thầy đã tận tình hướng dẫn trong suốt môn học Trí tuệ Nhân tạo. Sau đây, nhóm xin kính mời Thầy và các bạn cùng đặt câu hỏi để nhóm có thể làm rõ thêm các nội dung của đề tài. Nhóm xin trân trọng cảm ơn!"*

---

## BẢNG TỔNG HỢP DANH MỤC TÀI NGUYÊN HÌNH ẢNH CHÈN VÀO SLIDES

| Slide | Tên biểu đồ / Minh họa | Đường dẫn file tương ứng trong dự án |
| :---: | :--- | :--- |
| **Slide 6** | Cột so sánh hiệu năng (Kaggle) | `metrics/Dataset_1_Kaggle/charts/2_so_sanh_hieu_nang.png` |
| **Slide 6** | Ma trận nhầm lẫn 3 mô hình (Kaggle) | `metrics/Dataset_1_Kaggle/charts/3_ma_tran_nham_lan.png` |
| **Slide 7** | Cột so sánh hiệu năng (Cleveland) | `metrics/Dataset_2_Cleveland/charts/2_so_sanh_hieu_nang.png` |
| **Slide 7** | Ma trận nhầm lẫn 3 mô hình (Cleveland) | `metrics/Dataset_2_Cleveland/charts/3_ma_tran_nham_lan.png` |
| **Slide 8** | Đối sánh chéo đa tập dữ liệu | `metrics/Cross_Benchmark/bieu_do_so_sanh_2_bo_du_lieu.png` |
| **Slide 8** | Biểu đồ Feature Importance | `metrics/Dataset_1_Kaggle/charts/4_do_quan_trong_dac_trung.png` |
| **Slide 9** | Giao diện Web Streamlit | Chụp trực tiếp từ trình duyệt tại `http://localhost:8501` |
