import sys
import io
import os

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    recall_score, 
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Thiết lập thư mục gốc và các thư mục kết quả
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
metrics_dir = os.path.join(current_dir, "metrics")
cross_benchmark_dir = os.path.join(metrics_dir, "Cross_Benchmark")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(cross_benchmark_dir, exist_ok=True)

COL_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


def load_and_preprocess_dataset(csv_path, is_cleveland=False):
    """
    Nạp và tiền xử lý dữ liệu chuẩn hóa cho từng bộ dữ liệu.
    """
    if is_cleveland:
        # File cleveland.csv gốc từ UCI không có header và có giá trị '?'
        data = pd.read_csv(csv_path, header=None, names=COL_NAMES, na_values='?')
        # Điền các giá trị thiếu bằng giá trị xuất hiện nhiều nhất (mode)
        for col in ['ca', 'thal']:
            data[col] = data[col].fillna(data[col].mode()[0])
        # Chuyển nhãn mục tiêu: 0 là bình thường, >0 (1,2,3,4) là mắc bệnh tim
        data['target'] = (data['target'] > 0).astype(int)
    else:
        # File heart.csv có header sẵn
        data = pd.read_csv(csv_path)
    
    # Mã hóa các cột dạng chữ nếu có
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = le.fit_transform(data[col].astype(str))
            
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)
    return data, X, y


def train_and_benchmark(dataset_id, dataset_title, csv_path, is_cleveland=False):
    """
    Huấn luyện, đánh giá 3 mô hình và xuất báo cáo/biểu đồ cho một bộ dữ liệu cụ thể.
    """
    print("\n" + "=" * 65)
    print(f"TIẾN HÀNH BENCHMARK: {dataset_title}")
    print("=" * 65)
    
    # Tạo thư mục riêng cho từng Dataset
    ds_metrics_dir = os.path.join(metrics_dir, dataset_id)
    ds_charts_dir = os.path.join(ds_metrics_dir, "charts")
    os.makedirs(ds_metrics_dir, exist_ok=True)
    os.makedirs(ds_charts_dir, exist_ok=True)

    data, X, y = load_and_preprocess_dataset(csv_path, is_cleveland)
    print(f"-> Kích thước tập dữ liệu: {data.shape[0]} dòng, {data.shape[1]} cột")
    print(f"-> Phân bố nhãn: 0 (Bình thường): {(y == 0).sum()} mẫu | 1 (Bệnh tim): {(y == 1).sum()} mẫu")

    # Chia tập Train/Test (70/30) cố định random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"-> Phân chia: Train = {len(y_train)} mẫu | Test = {len(y_test)} mẫu")

    # 1. Decision Tree
    clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    clf_dt.fit(X_train, y_train)
    y_pred_dt = clf_dt.predict(X_test)

    # 2. Naive Bayes
    clf_nb = GaussianNB()
    clf_nb.fit(X_train, y_train)
    y_pred_nb = clf_nb.predict(X_test)

    # 3. Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)

    # Tổng hợp bảng so sánh hiệu năng
    df_compare = pd.DataFrame({
        "Bộ dữ liệu": dataset_title,
        "Mô hình": ["Decision Tree", "Naive Bayes", "Random Forest"],
        "Accuracy (%)": [
            round(100 * accuracy_score(y_test, y_pred_dt), 2),
            round(100 * accuracy_score(y_test, y_pred_nb), 2),
            round(100 * accuracy_score(y_test, y_pred_rf), 2)
        ],
        "Recall Lớp 1 (Bệnh tim)": [
            round(recall_score(y_test, y_pred_dt), 2),
            round(recall_score(y_test, y_pred_nb), 2),
            round(recall_score(y_test, y_pred_rf), 2)
        ],
        "F1-Score Lớp 1": [
            round(f1_score(y_test, y_pred_dt), 2),
            round(f1_score(y_test, y_pred_nb), 2),
            round(f1_score(y_test, y_pred_rf), 2)
        ]
    })

    print("\nKết quả thực nghiệm:")
    print(df_compare.to_string(index=False))

    # Xuất file Excel và CSV bảng so sánh
    df_compare.to_csv(os.path.join(ds_metrics_dir, "bang_so_sanh_mo_hinh.csv"), index=False, encoding="utf-8-sig")

    # Phân tích độ quan trọng đặc trưng (Feature Importance từ Random Forest)
    feat_imp = pd.Series(clf_rf.feature_importances_, index=X.columns)
    ranked_imp = feat_imp.sort_values(ascending=False)
    df_importance = pd.DataFrame({
        "Thứ hạng": list(range(1, len(ranked_imp) + 1)),
        "Thuộc tính": ranked_imp.index,
        "Mức độ quan trọng (%)": (ranked_imp.values * 100).round(2)
    })
    df_importance.to_csv(os.path.join(ds_metrics_dir, "do_quan_trong_dac_trung.csv"), index=False, encoding="utf-8-sig")

    try:
        df_compare.to_excel(os.path.join(ds_metrics_dir, "bang_so_sanh_mo_hinh.xlsx"), index=False)
        df_importance.to_excel(os.path.join(ds_metrics_dir, "do_quan_trong_dac_trung.xlsx"), index=False)
    except ModuleNotFoundError:
        pass

    # ==========================
    # VẼ VÀ LƯU CÁC BIỂU ĐỒ
    # ==========================
    # 1. Biểu đồ Cây Quyết định
    plt.figure(figsize=(22, 11), dpi=300)
    plot_tree(clf_dt, feature_names=list(X.columns), class_names=['Bình thường', 'Bệnh tim'], filled=True, rounded=True, fontsize=9)
    plt.title(f"Sơ đồ Cây Quyết định - {dataset_title}", fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(ds_charts_dir, "1_cay_quyet_dinh.png"), bbox_inches='tight')
    plt.close()

    # 2. Biểu đồ cột so sánh hiệu năng
    plt.figure(figsize=(9, 5), dpi=300)
    x_pos = np.arange(len(df_compare["Mô hình"]))
    bar_width = 0.25
    plt.bar(x_pos - bar_width, df_compare["Accuracy (%)"], bar_width, label="Accuracy (%)", color="#3498db")
    plt.bar(x_pos, df_compare["Recall Lớp 1 (Bệnh tim)"] * 100, bar_width, label="Recall Lớp 1 (%)", color="#e74c3c")
    plt.bar(x_pos + bar_width, df_compare["F1-Score Lớp 1"] * 100, bar_width, label="F1-Score Lớp 1 (%)", color="#2ecc71")
    plt.title(f"So sánh Hiệu năng Mô hình - {dataset_title}", fontsize=13, fontweight='bold')
    plt.xticks(x_pos, df_compare["Mô hình"], fontweight='bold')
    plt.ylim(0, 115)
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ds_charts_dir, "2_so_sanh_hieu_nang.png"), bbox_inches='tight')
    plt.close()

    # 3. Ma trận nhầm lẫn
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    models_preds = [("Decision Tree", y_pred_dt), ("Naive Bayes", y_pred_nb), ("Random Forest", y_pred_rf)]
    for idx, (m_name, pred) in enumerate(models_preds):
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bình thường', 'Bệnh tim'])
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f"{m_name}", fontsize=11, fontweight='bold')
    plt.suptitle(f"Confusion Matrix - {dataset_title}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ds_charts_dir, "3_ma_tran_nham_lan.png"), bbox_inches='tight')
    plt.close()

    # 4. Feature Importance
    plt.figure(figsize=(9, 5.5), dpi=300)
    feat_imp.sort_values(ascending=True).plot(kind='barh', color='#3498db', edgecolor='black')
    plt.title(f"Mức độ quan trọng đặc trưng (Random Forest) - {dataset_title}", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(ds_charts_dir, "4_do_quan_trong_dac_trung.png"), bbox_inches='tight')
    plt.close()

    # Lưu mô hình Random Forest tốt nhất của Dataset này vào thư mục models/
    model_filename = f"heart_rf_model_{dataset_id.lower()}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump({
        'model': clf_rf,
        'feature_names': list(X.columns),
        'dataset_title': dataset_title,
        'accuracy': df_compare.loc[df_compare["Mô hình"] == "Random Forest", "Accuracy (%)"].values[0],
        'recall': df_compare.loc[df_compare["Mô hình"] == "Random Forest", "Recall Lớp 1 (Bệnh tim)"].values[0]
    }, model_path)
    print(f"--> Đã đóng gói mô hình tại: {model_path}")

    return df_compare


def create_cross_benchmark_summary(df_ds1, df_ds2):
    """
    Tổng hợp bảng đối sánh và vẽ biểu đồ so sánh chéo giữa cả 2 bộ dữ liệu.
    """
    print("\n" + "=" * 65)
    print("TỔNG HỢP SO SÁNH CHÉO ĐA TẬP DỮ LIỆU (CROSS-DATASET BENCHMARK)")
    print("=" * 65)

    df_combined = pd.concat([df_ds1, df_ds2], ignore_index=True)
    print(df_combined.to_string(index=False))

    # Xuất file Excel & CSV tổng hợp
    excel_all = os.path.join(cross_benchmark_dir, "bang_tong_hop_cross_benchmark.xlsx")
    csv_all = os.path.join(cross_benchmark_dir, "bang_tong_hop_cross_benchmark.csv")
    df_combined.to_csv(csv_all, index=False, encoding="utf-8-sig")
    try:
        df_combined.to_excel(excel_all, index=False)
    except ModuleNotFoundError:
        pass

    # Vẽ biểu đồ so sánh chéo Accuracy & Recall của 2 bộ dữ liệu
    models = ["Decision Tree", "Naive Bayes", "Random Forest"]
    acc_ds1 = df_ds1["Accuracy (%)"].values
    acc_ds2 = df_ds2["Accuracy (%)"].values

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6), dpi=300)
    rects1 = plt.bar(x - width/2, acc_ds1, width, label='Dataset 1: Kaggle (1025 mẫu)', color='#3498db', edgecolor='black')
    rects2 = plt.bar(x + width/2, acc_ds2, width, label='Dataset 2: UCI Cleveland (303 mẫu)', color='#e67e22', edgecolor='black')

    plt.ylabel('Độ chính xác - Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('ĐỐI SÁNH ĐỘ CHÍNH XÁC CỦA CÁC THUẬT TOÁN TRÊN 2 BỘ DỮ LIỆU', fontsize=13, fontweight='bold')
    plt.xticks(x, models, fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            plt.annotate(f'{h:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_cross = os.path.join(cross_benchmark_dir, "bieu_do_so_sanh_2_bo_du_lieu.png")
    plt.savefig(chart_cross, bbox_inches='tight')
    plt.close()

    print(f"\n--> Đã xuất bảng tổng hợp tại: {excel_all}")
    print(f"--> Đã xuất biểu đồ đối sánh chéo tại: {chart_cross}")


if __name__ == "__main__":
    path_ds1 = os.path.join(current_dir, "Data", "heart.csv")
    path_ds2 = os.path.join(current_dir, "Data", "cleveland.csv")

    if not os.path.exists(path_ds1):
        print(f"Lỗi: Không tìm thấy file {path_ds1}")
        sys.exit(1)
    if not os.path.exists(path_ds2):
        print(f"Lỗi: Không tìm thấy file {path_ds2}")
        sys.exit(1)

    # 1. Huấn luyện & Đánh giá Dataset 1 (Kaggle Heart)
    res_ds1 = train_and_benchmark(
        dataset_id="Dataset_1_Kaggle",
        dataset_title="Kaggle Heart Disease (1025 mẫu)",
        csv_path=path_ds1,
        is_cleveland=False
    )

    # 2. Huấn luyện & Đánh giá Dataset 2 (UCI Cleveland)
    res_ds2 = train_and_benchmark(
        dataset_id="Dataset_2_Cleveland",
        dataset_title="UCI Cleveland (303 mẫu)",
        csv_path=path_ds2,
        is_cleveland=True
    )

    # 3. Tạo báo cáo & Biểu đồ đối sánh chéo giữa 2 bộ dữ liệu
    create_cross_benchmark_summary(res_ds1, res_ds2)

    # Cập nhật file mô hình mặc định cho web app trong thư mục models/
    default_model_src = os.path.join(models_dir, "heart_rf_model_dataset_1_kaggle.pkl")
    default_model_dst = os.path.join(models_dir, "heart_disease_rf_model.pkl")
    if os.path.exists(default_model_src):
        import shutil
        shutil.copy(default_model_src, default_model_dst)
        print("--> Đã đồng bộ mô hình mặc định cho web demo trong thư mục models/!")
