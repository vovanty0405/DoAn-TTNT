import os
# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import joblib

# Thiết lập cấu hình trang giao diện
st.set_page_config(
    page_title="Chẩn đoán Bệnh Tim AI - Đa tập dữ liệu", 
    page_icon="🫀", 
    layout="wide"
)

# Đường dẫn tài nguyên
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
metrics_dir = os.path.join(current_dir, 'metrics')
cross_benchmark_dir = os.path.join(metrics_dir, 'Cross_Benchmark')

# Cấu hình danh mục các bộ dữ liệu
DATASETS_CONFIG = {
    "ds1": {
        "title": "Dataset 1: Kaggle Heart Disease (1,025 mẫu)",
        "file_name": "heart_rf_model_dataset_1_kaggle.pkl",
        "folder": "Dataset_1_Kaggle",
        "description": "Tập dữ liệu chuẩn hóa từ Kaggle với 1,025 bệnh nhân (Train: 717, Test: 308).",
        "acc": "88.96%",
        "recall": "95.0%"
    },
    "ds2": {
        "title": "Dataset 2: UCI Cleveland (303 mẫu)",
        "file_name": "heart_rf_model_dataset_2_cleveland.pkl",
        "folder": "Dataset_2_Cleveland",
        "description": "Tập dữ liệu y khoa nguyên bản từ UCI Machine Learning Repository với 303 bệnh nhân.",
        "acc": "86.81%",
        "recall": "86.0%"
    }
}

# ==========================================
# THANH ĐIỀU HƯỚNG BÊN TRÁI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/865/865969.png", width=70)
    st.markdown("### ⚙️ Cấu hình Hệ thống")
    
    selected_ds_key = st.selectbox(
        "Chọn Bộ dữ liệu huấn luyện:",
        options=["ds1", "ds2"],
        format_func=lambda k: DATASETS_CONFIG[k]["title"]
    )
    
    selected_config = DATASETS_CONFIG[selected_ds_key]
    
    st.info(f"**Nguồn dữ liệu:** {selected_config['description']}")
    st.markdown("#### 🏆 Hiệu năng Random Forest:")
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Độ chính xác", selected_config["acc"])
    col_sb2.metric("Độ nhạy (Recall)", selected_config["recall"])
    
    st.write("---")
    st.caption("Khóa luận / Đồ án Trí tuệ nhân tạo\nThuật toán: Decision Tree, Naive Bayes, Random Forest")

# ==========================================
# NẠP MÔ HÌNH THEO BỘ DỮ LIỆU ĐƯỢC CHỌN (TỪ THƯ MỤC models/)
# ==========================================
model_path = os.path.join(models_dir, selected_config["file_name"])

@st.cache_resource
def load_trained_model(path):
    if not os.path.exists(path):
        # Dự phòng mô hình mặc định nếu file chưa kịp sinh
        fallback_path = os.path.join(models_dir, "heart_disease_rf_model.pkl")
        if os.path.exists(fallback_path):
            return joblib.load(fallback_path)
        st.error(f"Không tìm thấy file mô hình tại `{path}`! Hãy chạy file `benchmark.py` trước.")
        st.stop()
    return joblib.load(path)

artifacts = load_trained_model(model_path)
model = artifacts['model']
feature_names = artifacts['feature_names']

# ==========================================
# TIÊU ĐỀ CHÍNH
# ==========================================
st.markdown(
    "<p style='text-align: center; font-size: 28px; font-weight: bold; color: #e74c3c;'>🫀 ỨNG DỤNG CÁC MÔ HÌNH MÁY HỌC TRONG CẢNH BÁO NGUY CƠ BỆNH TIM</p>"
    "<p style='text-align: center; font-size: 15px; color: #555;'>"
    "Nghiên cứu, thực nghiệm đối sánh các mô hình máy học (Decision Tree, Naive Bayes, Random Forest)<br>"
    "và ứng dụng mô hình tối ưu vào hệ thống cảnh báo sớm trên nhiều bộ dữ liệu độc lập.</p>", 
    unsafe_allow_html=True
)
st.write("---")

# Phân chia tab chức năng: 1. Dự đoán bệnh nhân | 2. Báo cáo & Đánh giá mô hình
tab_predict, tab_analytics = st.tabs(["🩺 Chẩn đoán bệnh nhân", "📊 Báo cáo & Benchmark Đa bộ dữ liệu"])

# ==========================================
# TAB 1: CHẨN ĐOÁN BỆNH NHÂN
# ==========================================
with tab_predict:
    st.subheader(f"Nhập thông tin chỉ số lâm sàng (Áp dụng: {selected_config['title']})")
    st.info(f"ℹ️ **Ghi chú:** Tính năng dự đoán trực tiếp đang sử dụng mô hình tối ưu nhất qua thực nghiệm (**Random Forest** - Độ chính xác: {selected_config['acc']}, Recall: {selected_config['recall']}) từ **{selected_config['title']}**.")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Thông tin chung & Huyết động")
            age = st.number_input("1. Tuổi (age)", min_value=18, max_value=100, value=52, help="Độ tuổi bệnh nhân")
            sex = st.selectbox(
                "2. Giới tính (sex)", 
                options=[1, 0], 
                format_func=lambda x: "Nam" if x == 1 else "Nữ"
            )
            cp = st.selectbox(
                "3. Loại đau ngực (cp)", 
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 - Đau thắt ngực điển hình (Typical Angina)",
                    1: "1 - Đau thắt ngực không điển hình (Atypical Angina)",
                    2: "2 - Đau không do tim (Non-anginal Pain)",
                    3: "3 - Không triệu chứng (Asymptomatic)"
                }[x]
            )
            trestbps = st.number_input("4. Huyết áp lúc nghỉ ngơi (trestbps - mmHg)", min_value=80, max_value=220, value=125)
            chol = st.number_input("5. Cholesterol trong huyết thanh (chol - mg/dl)", min_value=100, max_value=600, value=212)
            fbs = st.selectbox(
                "6. Đường huyết lúc đói > 120 mg/dl? (fbs)", 
                options=[0, 1], 
                format_func=lambda x: "Không (<= 120 mg/dl)" if x == 0 else "Có (> 120 mg/dl)"
            )
            restecg = st.selectbox(
                "7. Điện tâm đồ lúc nghỉ (restecg)", 
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 - Bình thường",
                    1: "1 - Bất thường sóng ST-T",
                    2: "2 - Phì đại thất trái"
                }[x]
            )

        with col2:
            st.markdown("#### 🏃 Chỉ số gắng sức & Chẩn đoán hình ảnh")
            thalach = st.number_input("8. Nhịp tim tối đa đạt được (thalach)", min_value=60, max_value=220, value=168)
            exang = st.selectbox(
                "9. Đau thắt ngực khi vận động gắng sức? (exang)", 
                options=[0, 1], 
                format_func=lambda x: "Không" if x == 0 else "Có"
            )
            oldpeak = st.number_input("10. Độ chênh xuống ST khi gắng sức (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
            slope = st.selectbox(
                "11. Độ dốc đoạn ST đỉnh lúc gắng sức (slope)", 
                options=[0, 1, 2],
                format_func=lambda x: {
                    0: "0 - Dốc xuống (Downsloping)",
                    1: "1 - Phẳng (Flat)",
                    2: "2 - Dốc lên (Upsloping)"
                }[x]
            )
            ca = st.selectbox(
                "12. Số lượng mạch máu lớn cản quang (ca)", 
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: f"{x} mạch máu"
            )
            thal = st.selectbox(
                "13. Kết quả xét nghiệm Thallium (thal)", 
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 - Chưa rõ",
                    1: "1 - Khuyết tật cố định (Fixed Defect)",
                    2: "2 - Bình thường (Normal)",
                    3: "3 - Khuyết tật có thể hồi phục (Reversible Defect)"
                }[x]
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔍 TIẾN HÀNH DỰ ĐOÁN NGUY CƠ", use_container_width=True)

    if submit:
        input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        prediction = model.predict(input_df)[0]
        proba_benh = model.predict_proba(input_df)[0][1]
        proba_khoe = model.predict_proba(input_df)[0][0]

        st.markdown("### 📋 KẾT QUẢ CHẨN ĐOÁN:")
        res_col1, res_col2 = st.columns([1, 1])

        if prediction == 1:
            with res_col1:
                st.error("### ⚠️ NGUY CƠ CAO MẮC BỆNH TIM")
                st.markdown(f"- Mô hình sử dụng: **{selected_config['title']}**")
                st.markdown(f"- Xác suất dự đoán bệnh tim: **{proba_benh * 100:.1f}%**")
                st.markdown("- **Khuyến nghị:** Cần sớm đến cơ sở y tế chuyên khoa tim mạch để được làm các xét nghiệm chuyên sâu và can thiệp kịp thời.")
            with res_col2:
                st.metric(label="Độ tin cậy mắc bệnh tim", value=f"{proba_benh * 100:.1f}%", delta="Cảnh báo", delta_color="inverse")
        else:
            with res_col1:
                st.success("### ✅ TÌNH TRẠNG BÌNH THƯỜNG / NGUY CƠ THẤP")
                st.markdown(f"- Mô hình sử dụng: **{selected_config['title']}**")
                st.markdown(f"- Xác suất an toàn: **{proba_khoe * 100:.1f}%**")
                st.markdown("- **Khuyến nghị:** Duy trì lối sống lành mạnh, tập thể dục điều độ và khám định kỳ 6 tháng/lần.")
            with res_col2:
                st.metric(label="Độ an toàn sức khỏe", value=f"{proba_khoe * 100:.1f}%", delta="Bình thường")

# ==========================================
# TAB 2: BÁO CÁO & BENCHMARK ĐA BỘ DỮ LIỆU
# ==========================================
with tab_analytics:
    st.subheader("📈 Kết quả Thực nghiệm & Đối sánh Đa Tập dữ liệu")
    
    # 1. So sánh chéo đa tập dữ liệu
    st.markdown("### 1. Đối sánh chéo giữa các Bộ dữ liệu (Cross-Dataset Benchmark)")
    csv_cross_path = os.path.join(cross_benchmark_dir, "bang_tong_hop_cross_benchmark.csv")
    chart_cross_path = os.path.join(cross_benchmark_dir, "bieu_do_so_sanh_2_bo_du_lieu.png")
    
    if os.path.exists(csv_cross_path):
        df_cross = pd.read_csv(csv_cross_path)
        st.dataframe(df_cross, use_container_width=True)
        
    if os.path.exists(chart_cross_path):
        st.image(chart_cross_path, caption="Biểu đồ đối sánh độ chính xác của 3 thuật toán trên cả 2 bộ dữ liệu", use_container_width=True)
        st.success("💡 **Nhận xét khoa học:** Thuật toán **Random Forest** đạt độ chính xác cao nhất và ổn định vượt trội trên cả 2 bộ dữ liệu nghiên cứu độc lập (Kaggle: **88.96%**, UCI Cleveland: **86.81%**) so với Decision Tree và Naive Bayes.")

    st.write("---")
    
    # 2. Chi tiết theo từng bộ dữ liệu
    st.markdown(f"### 2. Chi tiết kết quả của `{selected_config['title']}`")
    ds_folder = os.path.join(metrics_dir, selected_config["folder"])
    ds_charts_folder = os.path.join(ds_folder, "charts")
    
    ds_compare_csv = os.path.join(ds_folder, "bang_so_sanh_mo_hinh.csv")
    if os.path.exists(ds_compare_csv):
        df_curr = pd.read_csv(ds_compare_csv)
        st.dataframe(df_curr, use_container_width=True)
        
    c1, c2 = st.columns(2)
    img_so_sanh = os.path.join(ds_charts_folder, "2_so_sanh_hieu_nang.png")
    img_imp = os.path.join(ds_charts_folder, "4_do_quan_trong_dac_trung.png")
    
    if os.path.exists(img_so_sanh):
        with c1:
            st.image(img_so_sanh, caption=f"Hiệu năng các mô hình - {selected_config['title']}", use_container_width=True)
    if os.path.exists(img_imp):
        with c2:
            st.image(img_imp, caption=f"Độ quan trọng đặc trưng - {selected_config['title']}", use_container_width=True)
            
    img_cm = os.path.join(ds_charts_folder, "3_ma_tran_nham_lan.png")
    if os.path.exists(img_cm):
        st.image(img_cm, caption=f"Ma trận nhầm lẫn (Confusion Matrix) - {selected_config['title']}", use_container_width=True)

    img_tree = os.path.join(ds_charts_folder, "1_cay_quyet_dinh.png")
    if os.path.exists(img_tree):
        with st.expander(f"🔍 Xem chi tiết Sơ đồ Cây Quyết định ({selected_config['title']})"):
            st.image(img_tree, caption=f"Sơ đồ cây quyết định - {selected_config['title']}", use_container_width=True)
