import sys
import io
import os

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 1. Khai báo các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Nạp thuật toán Gaussian Naive Bayes và các hàm đánh giá
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# Tạo thư mục Bayes để lưu các biểu đồ, file csv, ảnh
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Bayes")
os.makedirs(output_dir, exist_ok=True)

# 3. Đọc dữ liệu từ file CSV (Kaggle Heart Disease - 1025 mẫu)
csv_path_kaggle = os.path.join(current_dir, "Data", "heart.csv")
if not os.path.exists(csv_path_kaggle):
    csv_path_kaggle = 'D:/Năm 4/Trí tuệ nhân tạo-20260820T130944Z-1-001/DoAn/Data/heart.csv'

data = pd.read_csv(csv_path_kaggle)
print("=" * 60)
print("HUẤN LUYỆN MÔ HÌNH GAUSSIAN NAIVE BAYES")
print("=" * 60)
print("Kích thước dữ liệu (Kaggle):", data.shape)
print(data.head())

# 4. Xử lý các cột dạng chữ nếu có
le = preprocessing.LabelEncoder()
for column_name in data.columns:
    if data[column_name].dtype == object:
        data[column_name] = le.fit_transform(data[column_name])

# 5. Tách thuộc tính (X) và nhãn mục tiêu (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1].astype(int)

# 6. Chia tập Train và Test (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))

# 7. Huấn luyện mô hình Gaussian Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# 8. Dự đoán và đánh giá
y_pred_nb = nb_clf.predict(X_test)
y_prob_nb = nb_clf.predict_proba(X_test)[:, 1]

print("\nKết quả dự đoán thử trên tập test (20 mẫu đầu):")
print("Predicted labels: ", y_pred_nb[:20])
print("Ground truth    : ", y_test[:20].values)

print("\nAccuracy of Naive Bayes (Kaggle): %.2f %%" % (100 * accuracy_score(y_test, y_pred_nb)))
print("\nClassification Report (Naive Bayes - Kaggle):\n", classification_report(y_test, y_pred_nb))

# ==============================================================================
# 9. THỰC NGHIỆM TRÊN BỘ DỮ LIỆU UCI CLEVELAND & XUẤT KẾT QUẢ VÀO THƯ MỤC Bayes/
# ==============================================================================
COL_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

csv_path_cleveland = os.path.join(current_dir, "Data", "cleveland.csv")
results_list = [{
    "Bộ dữ liệu": "Kaggle Heart Disease (1025 mẫu)",
    "Mô hình": "Gaussian Naive Bayes",
    "Accuracy (%)": round(100 * accuracy_score(y_test, y_pred_nb), 2),
    "Precision Lớp 1": round(precision_score(y_test, y_pred_nb), 2),
    "Recall Lớp 1": round(recall_score(y_test, y_pred_nb), 2),
    "F1-Score Lớp 1": round(f1_score(y_test, y_pred_nb), 2)
}]

nb_clf_clv = None
X_test_clv, y_test_clv, y_pred_clv = None, None, None

if os.path.exists(csv_path_cleveland):
    data_clv = pd.read_csv(csv_path_cleveland, header=None, names=COL_NAMES, na_values='?')
    for col in ['ca', 'thal']:
        data_clv[col] = data_clv[col].fillna(data_clv[col].mode()[0])
    data_clv['target'] = (data_clv['target'] > 0).astype(int)
    X_clv = data_clv.iloc[:, :-1]
    y_clv = data_clv.iloc[:, -1].astype(int)
    X_train_clv, X_test_clv, y_train_clv, y_test_clv = train_test_split(X_clv, y_clv, test_size=0.3, random_state=42)
    nb_clf_clv = GaussianNB()
    nb_clf_clv.fit(X_train_clv, y_train_clv)
    y_pred_clv = nb_clf_clv.predict(X_test_clv)
    print("\nAccuracy of Naive Bayes (Cleveland): %.2f %%" % (100 * accuracy_score(y_test_clv, y_pred_clv)))
    print("\nClassification Report (Naive Bayes - Cleveland):\n", classification_report(y_test_clv, y_pred_clv))
    results_list.append({
        "Bộ dữ liệu": "UCI Cleveland (303 mẫu)",
        "Mô hình": "Gaussian Naive Bayes",
        "Accuracy (%)": round(100 * accuracy_score(y_test_clv, y_pred_clv), 2),
        "Precision Lớp 1": round(precision_score(y_test_clv, y_pred_clv), 2),
        "Recall Lớp 1": round(recall_score(y_test_clv, y_pred_clv), 2),
        "F1-Score Lớp 1": round(f1_score(y_test_clv, y_pred_clv), 2)
    })

# Xuất bảng kết quả tổng hợp ra CSV & Excel trong thư mục Bayes/
df_results = pd.DataFrame(results_list)
df_results.to_csv(os.path.join(output_dir, "ket_qua_bayes.csv"), index=False, encoding="utf-8-sig")
df_results.to_excel(os.path.join(output_dir, "ket_qua_bayes.xlsx"), index=False)

# Xuất báo cáo chi tiết từng lớp (Classification Report)
report_dict = classification_report(y_test, y_pred_nb, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose().round(2)
df_report.to_csv(os.path.join(output_dir, "bao_cao_chi_tiet_bayes.csv"), encoding="utf-8-sig")

# Xuất bảng tham số thống kê Gaussian (Mean & Variance của từng thuộc tính theo lớp 0 và lớp 1)
df_stats = pd.DataFrame({
    "Thuộc tính": X.columns,
    "Trung bình Lớp 0 (Bình thường)": nb_clf.theta_[0].round(3),
    "Trung bình Lớp 1 (Bệnh tim)": nb_clf.theta_[1].round(3),
    "Phương sai Lớp 0": nb_clf.var_[0].round(3),
    "Phương sai Lớp 1": nb_clf.var_[1].round(3)
})
df_stats.to_csv(os.path.join(output_dir, "thong_ke_gaussian_bayes.csv"), index=False, encoding="utf-8-sig")
df_stats.to_excel(os.path.join(output_dir, "thong_ke_gaussian_bayes.xlsx"), index=False)

# Vẽ và lưu các biểu đồ vào thư mục Bayes/
# 1. Ma trận nhầm lẫn (Confusion Matrix)
if y_pred_clv is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nb), display_labels=['Bình thường', 'Bệnh tim']).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title("Naive Bayes - Kaggle (1025 mẫu)", fontweight='bold')
    ConfusionMatrixDisplay(confusion_matrix(y_test_clv, y_pred_clv), display_labels=['Bình thường', 'Bệnh tim']).plot(ax=axes[1], cmap='Blues', colorbar=False)
    axes[1].set_title("Naive Bayes - Cleveland (303 mẫu)", fontweight='bold')
    plt.suptitle("Ma trận nhầm lẫn (Confusion Matrix) - Gaussian Naive Bayes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_confusion_matrix_bayes.png"), bbox_inches='tight')
    plt.close()

# 2. Biểu đồ cột so sánh hiệu năng Naive Bayes
plt.figure(figsize=(8, 5), dpi=300)
x_pos = np.arange(len(df_results["Bộ dữ liệu"]))
width = 0.25
plt.bar(x_pos - width, df_results["Accuracy (%)"], width, label="Accuracy (%)", color="#3498db")
plt.bar(x_pos, df_results["Recall Lớp 1"] * 100, width, label="Recall Lớp 1 (%)", color="#e74c3c")
plt.bar(x_pos + width, df_results["F1-Score Lớp 1"] * 100, width, label="F1-Score Lớp 1 (%)", color="#2ecc71")
plt.xticks(x_pos, df_results["Bộ dữ liệu"], fontweight='bold')
plt.ylabel("Tỷ lệ (%)")
plt.ylim(0, 115)
plt.title("Hiệu năng mô hình Gaussian Naive Bayes", fontsize=13, fontweight='bold')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2_bieu_do_hieu_nang_bayes.png"), bbox_inches='tight')
plt.close()

# 3. Biểu đồ phân phối xác suất dự đoán của Naive Bayes
plt.figure(figsize=(9, 5), dpi=300)
plt.hist(y_prob_nb[y_test == 0], bins=20, alpha=0.65, label="Thực tế: Bình thường (0)", color="#2ecc71", edgecolor="black")
plt.hist(y_prob_nb[y_test == 1], bins=20, alpha=0.65, label="Thực tế: Bệnh tim (1)", color="#e74c3c", edgecolor="black")
plt.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Ngưỡng phân loại (0.5)")
plt.title("Phân bố xác suất dự đoán bệnh tim (Gaussian Naive Bayes)", fontsize=13, fontweight='bold')
plt.xlabel("Xác suất dự đoán mắc bệnh tim P(y=1|X)")
plt.ylabel("Số lượng bệnh nhân (Tập Test)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3_phan_bo_xac_suat_bayes.png"), bbox_inches='tight')
plt.close()

print(f"\n--> Đã lưu toàn bộ bảng CSV, Excel và biểu đồ của Naive Bayes vào thư mục: {output_dir}")

