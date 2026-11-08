"""Streamlit demo loads persisted Heart pipelines; never fits at runtime."""
import json
import joblib
import pandas as pd
import streamlit as st
from heart_pipeline import OUT, FEATURES, CODES, LABELS, normalize_input, disease_score

st.set_page_config(page_title='Phân loại bệnh tim – Heart',page_icon='🫀',layout='wide')

@st.cache_resource
def load_artifacts(version):
    experiment=json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
    models={name:joblib.load(OUT/'models'/(name.replace(' ','_')+'.joblib')) for name in experiment['models']}
    return experiment,models

if not (OUT/'experiment.json').exists():
    st.error('Chưa có kết quả. Chạy: python benchmark.py')
    st.stop()
experiment,models=load_artifacts((OUT/"experiment.json").stat().st_mtime_ns)
audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
samples=json.loads((OUT/'demo_samples.json').read_text(encoding='utf-8'))
metrics=pd.read_csv(OUT/'metrics.csv')
st.title('Phân loại sự hiện diện của bệnh tim')
st.caption('Heart Disease Dataset • Decision Tree, Random Forest và Gaussian Naive Bayes')
st.info('Ứng dụng phục vụ mục đích học tập và nghiên cứu, không thay thế việc thăm khám hoặc chẩn đoán của bác sĩ.')
with st.sidebar:
    st.header('Thực nghiệm Heart')
    st.write(f"{audit['raw_rows']} bản ghi gốc → {audit['unique_rows']} dòng duy nhất")
    st.write(f"Train: {audit['train_rows']} · Test: {audit['test_rows']}")
    st.write('Mô hình chọn bằng F1 trung bình CV trên Train:')
    st.write(experiment['selected_model'])
    st.caption('Nhãn chuẩn hóa: 0 = không bệnh, 1 = có bệnh. Điểm mô hình chưa được hiệu chỉnh thành xác suất lâm sàng.')
tab1,tab2=st.tabs(['Dự đoán bệnh tim','Đánh giá mô hình'])
with tab1:
    model_name=st.selectbox('Mô hình',list(models),index=list(models).index(experiment['selected_model']))
    demo_choice=st.selectbox('Bản ghi Test để demo',['Có bệnh theo nhãn thật','Không có bệnh theo nhãn thật'])
    sample=samples[0 if demo_choice.startswith('Có bệnh') else 1]
    st.caption(f"Mẫu từ Test, dòng nguồn {sample['source_row']} (đếm từ 0). Nhãn thật: {LABELS[sample['true_label']]}. Có thể sửa các giá trị bên dưới.")
    labels={'age':'Tuổi (năm)','trestbps':'Huyết áp nghỉ (mmHg)','chol':'Cholesterol (mg/dL)','thalach':'Nhịp tim tối đa (nhịp/phút)','oldpeak':'ST chênh xuống (oldpeak)',
        'sex':'Giới tính','cp':'Loại đau ngực','fbs':'Đường huyết lúc đói','restecg':'Điện tâm đồ nghỉ','exang':'Đau ngực khi gắng sức','slope':'Độ dốc ST','ca':'Số mạch lớn','thal':'Kết quả thal'}
    limits={'age':(1.,120.),'trestbps':(1.,300.),'chol':(1.,1000.),'thalach':(1.,250.),'oldpeak':(0.,10.)}
    with st.form('patient'):
        columns=st.columns(3); values={}
        for i,col in enumerate(FEATURES):
            with columns[i%3]:
                val=sample['features'][col]
                if col in CODES:
                    options=list(CODES[col]);code=(4 if col=='ca' else 0) if val is None else int(val)
                    values[col]=st.selectbox(labels[col],options,index=options.index(code),format_func=lambda v,c=col:f'{v} – {CODES[c][v]}',key=f'{sample["source_row"]}_{col}')
                else:
                    values[col]=st.number_input(labels[col],min_value=limits[col][0],max_value=limits[col][1],value=float(val),step=.1 if col=='oldpeak' else 1.,key=f'{sample["source_row"]}_{col}')
        submitted=st.form_submit_button('Phân loại')
    if submitted:
        frame=normalize_input(pd.DataFrame([values],columns=FEATURES));model=models[model_name]
        pred=int(model.predict(frame)[0]);score=float(disease_score(model,frame)[0])
        st.subheader('Kết quả phân loại của mô hình')
        st.success('Thuộc nhóm '+('có bệnh' if pred else 'không có bệnh')+' theo dữ liệu nghiên cứu')
        st.metric('Điểm xác suất do mô hình ước lượng cho lớp có bệnh',f'{score:.1%}')
        st.caption('Điểm này không phải xác suất mắc bệnh tim thực tế hay nguy cơ trong tương lai. Kết quả có thể sai ở cả hai nhóm.')
with tab2:
    st.subheader('Kết quả trên tập Test độc lập')
    st.dataframe(metrics.style.format({c:'{:.3f}' for c in ['accuracy','precision','recall','f1','roc_auc']}),hide_index=True,width='stretch')
    st.image(str(OUT/'charts/performance.png'),width=750)
    selected=experiment['selected_model']
    st.write('Phân tích mô hình được chọn: '+selected)
    left,right=st.columns(2)
    left.image(str(OUT/'charts'/('cm_'+selected.replace(' ','_')+'.png')),width='stretch')
    right.image(str(OUT/'charts/importance.png'),width='stretch')
    st.caption('Permutation importance trên Test, mức giảm F1 sau hoán vị 30 lần. Đây là phân tích sau đánh giá, không dùng để chọn đặc trưng hay huấn luyện lại. Không thể suy ra quan hệ nhân quả.')
    st.subheader('Stratified 5-Fold CV chỉ trên Train')
    st.dataframe(pd.read_csv(OUT/'cross_validation.csv'),hide_index=True)
    st.caption('Điểm CV sau tối ưu dùng lại các fold tìm kiếm nên có thể lạc quan. Test gồm 91 bản ghi duy nhất, chưa có kiểm định ngoài hoặc hiệu chỉnh xác suất.')
    st.json(audit,expanded=False)
