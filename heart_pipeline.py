"""Quy trình chính để huấn luyện, chọn và đánh giá ba mô hình phân loại bệnh tim."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

# 1. Đường dẫn, thứ tự 13 đặc trưng và ý nghĩa các mã trong bộ Heart.
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
FEATURES = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal',
]
NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
BINARY = ['sex', 'fbs', 'exang']
CATEGORICAL = ['cp', 'restecg', 'slope', 'ca', 'thal']
LABELS = {0: 'Không có bệnh', 1: 'Có bệnh'}
CODES = {
    'sex': {0: 'Nữ', 1: 'Nam'},
    'fbs': {0: '≤ 120 mg/dL', 1: '> 120 mg/dL'},
    'exang': {0: 'Không', 1: 'Có'},
    'cp': {
        0: 'Không triệu chứng',
        1: 'Đau thắt ngực không điển hình',
        2: 'Đau ngực không do thắt ngực',
        3: 'Đau thắt ngực điển hình',
    },
    'restecg': {0: 'Phì đại thất trái', 1: 'Bình thường', 2: 'Bất thường ST-T'},
    'slope': {0: 'Dốc xuống', 1: 'Ngang', 2: 'Dốc lên'},
    'ca': {
        0: '0 mạch',
        1: '1 mạch',
        2: '2 mạch',
        3: '3 mạch',
        4: 'Thiếu / không xác định',
    },
    'thal': {
        0: 'Thiếu / không xác định',
        1: 'Khiếm khuyết cố định',
        2: 'Bình thường',
        3: 'Khiếm khuyết hồi phục',
    },
}

# 2. Đọc và kiểm tra dữ liệu.


def write_json(path, value):
    """Ghi kết quả dạng JSON, giữ tiếng Việt và hỗ trợ số từ NumPy."""
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if hasattr(x, 'item') else str(x),
        ),
        encoding='utf-8',
    )


def load_heart():
    """Trả về dữ liệu gốc, dữ liệu đã làm sạch và thống kê kiểm tra."""
    raw = pd.read_csv(ROOT / 'Data/heart.csv', na_values=['?', '', ' ', '-9', '-9.0'])
    if list(raw.columns) != FEATURES + ['target']:
        raise ValueError('Unexpected schema')
    data = raw.drop_duplicates().copy()
    # Chỉ đảo nhãn khi bằng chứng đối chiếu khớp đúng tệp CSV hiện tại.
    proof = json.loads((OUT / 'label_verification.json').read_text(encoding='utf-8'))
    dataset_hash = hashlib.sha256((ROOT / 'Data/heart.csv').read_bytes()).hexdigest()
    if proof['heart_sha256'] != dataset_hash or proof['label_agreement'] != len(data):
        raise ValueError('Run tools/verify_labels.py for this dataset before training')
    if not set(data.target.unique()) <= {0, 1}:
        raise ValueError('Unexpected raw labels')
    data['target'] = 1 - data.target
    # Cùng 13 đặc trưng nhưng khác nhãn là dữ liệu cần xem lại.
    conflicts = int(data.groupby(FEATURES, dropna=False).target.nunique().gt(1).sum())
    if conflicts:
        raise ValueError('Conflicting labels require manual review')
    for (col, codes) in CODES.items():
        if not set(data[col].dropna().unique()) <= set(codes):
            raise ValueError('Unexpected code ' + col)
    audit = {
        'raw_rows': len(raw),
        'columns': len(raw.columns),
        'duplicates': int(raw.duplicated().sum()),
        'unique_rows': len(data),
        'conflicting_groups': conflicts,
        'raw_target_counts': raw.target.value_counts().to_dict(),
        'target_counts': data.target.value_counts().to_dict(),
        'dtypes': raw.dtypes.astype(str).to_dict(),
        'raw_missing': raw.isna().sum().to_dict(),
        'special_missing_unique': {'ca=4': int(data.ca.eq(4).sum()), 'thal=0': int(data.thal.eq(0).sum())},
        'ranges': raw.agg(['min', 'max']).to_dict(),
        'label_rule': 'normalized_target = 1 - raw_target',
        'outlier_policy': 'Retain plausible extremes including chol=564; no outcome-driven deletion.',
    }
    # Ghi số mã thiếu vào audit trước khi chuyển chúng thành NaN.
    data['ca'] = data.ca.replace(4, np.nan)
    data['thal'] = data.thal.replace(0, np.nan)
    return (raw, data, audit)


# 3. Tiền xử lý dùng chung cho huấn luyện và dự đoán.


def pipeline(model, scale=False):
    """Ghép các bước đổi mã thiếu, tiền xử lý và phân loại."""
    # scale=True chỉ dùng cho Gaussian NB trong thí nghiệm này.
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    pre = ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('binary', SimpleImputer(strategy='most_frequent'), BINARY),
        ('category', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([('codes', SpecialMissing()), ('preprocess', pre), ('model', model)])


class SpecialMissing(TransformerMixin, BaseEstimator):
    """Đưa quy tắc xử lý mã thiếu vào pipeline được lưu cùng mô hình."""

    def fit(self, X, y=None):
        """Ghi tên và số cột đầu vào. Bước này không ước lượng tham số."""
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        """Áp dụng cùng quy tắc cho dữ liệu huấn luyện hoặc dữ liệu mới."""
        return normalize_input(X)

    def get_feature_names_out(self, input_features=None):
        """Trả về danh sách tên cột theo thứ tự sau khi sắp xếp."""
        return np.array(FEATURES, dtype=object)


def normalize_input(frame):
    """Sắp đúng thứ tự cột và đổi mã thiếu trên bản sao dữ liệu."""
    frame = frame.loc[:, FEATURES].copy()
    frame['ca'] = frame.ca.replace(4, np.nan)
    frame['thal'] = frame.thal.replace(0, np.nan)
    return frame


# 4. Tính điểm dự đoán và các chỉ số đánh giá.


def disease_score(model, frame):
    """Lấy điểm của lớp 1 theo thứ tự lớp mà mô hình đã học."""
    return model.predict_proba(normalize_input(frame))[:, list(model.classes_).index(1)]


def evaluate(model, X, y):
    """So sánh dự đoán với nhãn thật và trả về điểm số cùng ma trận nhầm lẫn."""
    pred = model.predict(X)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, disease_score(model, X))),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def train():
    """Chạy toàn bộ thí nghiệm từ dữ liệu gốc đến kết quả trong results/."""
    # Bước 1. Đọc dữ liệu và chia phân tầng để giữ tỷ lệ nhãn giữa hai tập.
    OUT.mkdir(exist_ok=True)
    _, data, audit = load_heart()
    X, y = data[FEATURES], data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # Kiểm tra hai tập không trùng chỉ số dòng hoặc bộ đặc trưng.
    assert not set(X_train.index) & set(X_test.index)
    assert not set(map(tuple, X_train.fillna(-999).to_numpy())) & set(map(tuple, X_test.fillna(-999).to_numpy()))
    audit.update(
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_target_counts=y_train.value_counts().to_dict(),
        test_target_counts=y_test.value_counts().to_dict(),
    )
    for (filename, df) in [
        ('heart_clean', data),
        ('train', data.loc[X_train.index]),
        ('test', data.loc[X_test.index]),
    ]:
        df.to_csv(OUT / (filename + '.csv'), index_label='source_row')
    write_json(OUT / 'data_audit.json', audit)
    # Bước 2. Khai báo cách chia 5 fold, cấu hình ban đầu và lưới tham số.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scoring = {m: m for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
    configs = {
        'Decision Tree': (pipeline(DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)), {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [2, 4, 6, None],
            'model__min_samples_split': [2, 6],
            'model__min_samples_leaf': [1, 4],
        }),
        'Random Forest': (pipeline(RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1)), {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, None],
            'model__min_samples_split': [2, 6],
            'model__min_samples_leaf': [1, 3],
            'model__max_features': ['sqrt', 0.7],
        }),
        'Gaussian NB': (pipeline(GaussianNB(), True), {'model__var_smoothing': [1e-11, 1e-09, 1e-07, 1e-05, 0.001, 0.1]}),
    }
    details = {}
    fitted = {}
    cv_rows = []
    baselines = {}
    # Bước 3. Đánh giá baseline và tìm cấu hình tốt nhất trên Train.
    for name, (base, grid) in configs.items():
        scores = cross_validate(base, X_train, y_train, cv=cv, scoring=scoring)
        base.fit(X_train, y_train)
        baselines[name] = base
        search = GridSearchCV(
            base,
            grid,
            scoring=scoring,
            refit='f1',
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        # refit='f1' chọn cấu hình theo F1 CV rồi fit lại trên toàn bộ Train.
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        details[name] = {
            'best_params': search.best_params_,
            'baseline_params': base.named_steps['model'].get_params(),
            'search_candidates': len(search.cv_results_['params']),
        }
        for variant in ['baseline', 'tuned']:
            row = {'model': name, 'variant': variant}
            for metric in scoring:
                # Các khóa test_* ở đây là điểm trên fold xác thực của Train.
                if variant == 'baseline':
                    values = scores['test_' + metric]
                else:
                    values = np.array([
                        search.cv_results_[f'split{i}_test_{metric}'][search.best_index_]
                        for i in range(5)
                    ])
                row[metric + '_mean'] = float(values.mean())
                row[metric + '_std'] = float(values.std())
            cv_rows.append(row)
        pd.DataFrame(search.cv_results_).to_csv(
            OUT / (name.replace(' ', '_') + '_search.csv'),
            index=False,
        )
        print(name, search.best_score_, search.best_params_, flush=True)
    # Bước 4. Chọn mô hình chính theo F1 CV, trước khi tính điểm trên Test.
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(OUT / 'cross_validation.csv', index=False)
    selected = cv_table[cv_table.variant.eq('tuned')].sort_values(
        ['f1_mean', 'model'],
        ascending=[False, True],
    ).iloc[0]['model']
    # Bước 5. Đánh giá baseline và cấu hình đã chọn trên Test giữ riêng.
    rows = []
    (OUT / 'models').mkdir(exist_ok=True)
    for (name, model) in fitted.items():
        result = evaluate(model, X_test, y_test)
        details[name]['test'] = result
        details[name]['baseline_test'] = evaluate(baselines[name], X_test, y_test)
        rows.append({
            'model': name,
            **{k: v for (k, v) in result.items() if k != 'confusion_matrix'},
        })
        file = OUT / 'models' / (name.replace(' ', '_') + '.joblib')
        joblib.dump(model, file)
        # Xác nhận pipeline đã lưu cho cùng dự đoán khi tải lại.
        loaded = joblib.load(file)
        assert np.array_equal(model.predict(X_test), loaded.predict(X_test))
        assert np.allclose(disease_score(model, X_test), disease_score(loaded, X_test))
    pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False)
    # Bước 6. Hoán vị từng biến gốc trên Test để đo mức giảm F1.
    imp = permutation_importance(
        fitted[selected],
        X_test,
        y_test,
        scoring='f1',
        n_repeats=30,
        random_state=42,
        n_jobs=1,
    )
    pd.DataFrame({
        'feature': FEATURES,
        'mean': imp.importances_mean,
        'std': imp.importances_std,
    }).sort_values(
        'mean',
        ascending=False,
    ).to_csv(
        OUT / 'permutation_importance.csv',
        index=False,
    )
    # Bước 7. Lấy một mẫu mỗi lớp cho ứng dụng, ghi cấu hình và vẽ biểu đồ.
    demo = []
    for label in [1, 0]:
        idx = y_test[y_test.eq(label)].index[0]
        frame = X_test.loc[[idx]]
        demo.append({
            'source_row': int(idx),
            'true_label': label,
            'features': {k: None if pd.isna(v) else float(v) for (k, v) in frame.iloc[0].items()},
            'prediction': int(fitted[selected].predict(frame)[0]),
            'disease_score': float(disease_score(fitted[selected], frame)[0]),
        })
    write_json(OUT / 'demo_samples.json', demo)
    write_json(
        OUT / 'experiment.json',
        {
            'selected_model': selected,
            'selection_rule': 'Highest mean F1 in Stratified 5-fold CV on Train; alphabetical tie-break. Default classification threshold.',
            'cv_caveat': 'Tuned CV reuses search folds and may be optimistic. Test is independent. No nested CV.',
            'random_state': 42,
            'models': details,
            'dataset_sha256': hashlib.sha256((ROOT / 'Data/heart.csv').read_bytes()).hexdigest(),
            'versions': {
                'python': platform.python_version(),
                'sklearn': sklearn.__version__,
                'pandas': pd.__version__,
                'numpy': np.__version__,
            },
        },
    )
    from tools.make_charts import make_charts
    make_charts(data, audit, details, selected)
    print('Selected:', selected, flush=True)


if __name__ == '__main__':
    train()
