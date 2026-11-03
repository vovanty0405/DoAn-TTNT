import sys
import io
import os

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Nạp các thuật toán và hàm đánh giá từ scikit-learn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score

# ==========================================
# 1. ĐỌC DỮ LIỆU VÀ TIỀN XỬ LÝ
# ==========================================
print("=" * 50)
print("1. NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 50)

# Xác định đường dẫn file dữ liệu (hỗ trợ cả đường dẫn tương đối và tuyệt đối)
current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(current_dir, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

csv_path = os.path.join(current_dir, "Data", "heart.csv")

if not os.path.exists(csv_path):
    # Dự phòng đường dẫn tuyệt đối nếu chạy ở thư mục khác
    csv_path = "../Data/heart.csv"

data = pd.read_csv(csv_path)
print("Kích thước dữ liệu:", data.shape)
print(data.head())

# Xử lý các cột dạng chữ nếu có bằng LabelEncoder
le = preprocessing.LabelEncoder()
for column_name in data.columns:
    if data[column_name].dtype == object:
        data[column_name] = le.fit_transform(data[column_name])

# Tách thuộc tính (X) và nhãn mục tiêu (y) - cột 'target' nằm ở cuối cùng
X = data.iloc[:, :-1]
y = data.iloc[:, -1].astype(int)

# Chia tập Train và Test (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Số mẫu tập Train: %d" % len(y_train))
print("Số mẫu tập Test : %d" % len(y_test))

# ==========================================
# 2. HUẤN LUYỆN MÔ HÌNH CÂY QUYẾT ĐỊNH (DECISION TREE)
# ==========================================
print("\n" + "=" * 50)
print("2. HUẤN LUYỆN DECISION TREE (CRITERION: ENTROPY)")
print("=" * 50)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy of Decision Tree: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred))

# ==========================================
# 3. XUẤT HÌNH ẢNH CÂY QUYẾT ĐỊNH
# ==========================================
print("\n--- Đang xuất hình ảnh cây quyết định ---")
plt.figure(figsize=(24, 12), dpi=300)
plot_tree(
    clf, 
    feature_names=list(X.columns), 
    class_names=['BinhThuong (0)', 'BenhTim (1)'], 
    filled=True, 
    rounded=True, 
    fontsize=9
)
plt.title("Sơ đồ Cây Quyết định dự đoán Bệnh Tim (Criterion: Entropy)", fontsize=16, fontweight='bold')
output_img = os.path.join(metrics_dir, "cay_quyet_dinh.png")
plt.savefig(output_img, bbox_inches='tight')
plt.close()
print(f"--> Đã lưu thành công file ảnh tại: {output_img}")

# ==========================================
# 4. HUẤN LUYỆN MÔ HÌNH NAIVE BAYES (GAUSSIAN NB)
# ==========================================
print("\n" + "=" * 50)
print("3. HUẤN LUYỆN NAIVE BAYES (GAUSSIAN)")
print("=" * 50)

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred_nb = nb_clf.predict(X_test)

print("Accuracy of Naive Bayes: %.2f %%" % (100 * accuracy_score(y_test, y_pred_nb)))
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, y_pred_nb))

# ==========================================
# 5. HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
# ==========================================
print("\n" + "=" * 50)
print("4. HUẤN LUYỆN RANDOM FOREST (100 CÂY)")
print("=" * 50)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

print("Accuracy of Random Forest: %.2f %%" % (100 * accuracy_score(y_test, y_pred_rf)))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# ==========================================
# 6. TỔNG HỢP BẢNG SO SÁNH HIỆU NĂNG CHO SLIDE / BÁO CÁO
# ==========================================
comparison_data = {
    "Mô hình": ["Decision Tree", "Naive Bayes", "Random Forest"],
    "Accuracy (%)": [
        round(100 * accuracy_score(y_test, y_pred), 2),
        round(100 * accuracy_score(y_test, y_pred_nb), 2),
        round(100 * accuracy_score(y_test, y_pred_rf), 2)
    ],
    "Recall Lớp 1 (Bệnh tim)": [
        round(recall_score(y_test, y_pred), 2),
        round(recall_score(y_test, y_pred_nb), 2),
        round(recall_score(y_test, y_pred_rf), 2)
    ],
    "F1-Score Lớp 1": [
        round(f1_score(y_test, y_pred), 2),
        round(f1_score(y_test, y_pred_nb), 2),
        round(f1_score(y_test, y_pred_rf), 2)
    ]
}

df_compare = pd.DataFrame(comparison_data)
print("\n" + "=" * 50)
print("BẢNG TỔNG KẾT ĐỐI SÁNH KẾT QUẢ:")
print("=" * 50)
print(df_compare.to_string(index=False))

# ==========================================
# 7. PHÂN TÍCH ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (FEATURE IMPORTANCE)
# ==========================================
print("\n" + "=" * 50)
print("7. ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (RANDOM FOREST)")
print("=" * 50)

# Trích xuất độ quan trọng của các đặc trưng từ Random Forest
feature_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=True)

# In danh sách xếp hạng từ cao xuống thấp để tiện đưa vào báo cáo
ranked_features = feature_importances.sort_values(ascending=False)
print("Bảng xếp hạng mức độ ảnh hưởng đến nguy cơ bệnh tim:")
for rank, (feat, imp) in enumerate(ranked_features.items(), start=1):
    print(f" {rank:2d}. {feat:<12}: {imp*100:6.2f} %")

# Vẽ biểu đồ thanh ngang
plt.figure(figsize=(10, 6), dpi=300)
sorted_importances.plot(kind='barh', color='#3498db', edgecolor='black')
plt.title("Mức độ quan trọng của các đặc trưng (Random Forest Feature Importance)", fontsize=14, fontweight='bold')
plt.xlabel("Mức độ quan trọng (Importance Score)", fontsize=11)
plt.ylabel("Thuộc tính (Features)", fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

importance_img = os.path.join(metrics_dir, "feature_importance.png")
plt.savefig(importance_img, bbox_inches='tight')
plt.close()
print(f"\n--> Đã lưu thành công biểu đồ tại: {importance_img}")

import joblib

# 2. Đóng gói mô hình ra file .pkl để làm Web Demo
model_data = {
    'model': rf_clf,
    'feature_names': list(X.columns)
}
joblib.dump(model_data, 'heart_disease_rf_model.pkl')
print("--> Đã đóng gói mô hình thành công vào file: heart_disease_rf_model.pkl")