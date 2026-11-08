# 1. Khai báo thư viện và thuật toán
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools.make_charts import save_model_charts


if __name__ == '__main__':
    # 2. Đọc dữ liệu và chuẩn hóa nhãn của bộ Heart
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / 'Data' / 'heart.csv')
    data = data.drop_duplicates().copy()
    data['target'] = 1 - data['target']
    data['ca'] = data['ca'].replace(4, np.nan)
    data['thal'] = data['thal'].replace(0, np.nan)
    print('Dataset preview:')
    print(data.head())

    # 3. Tách nhãn, đặc trưng và chia Train/Test
    y = data['target']
    X = data.drop(columns='target')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print('Training size: %d' % len(y_train))
    print('Test size    : %d' % len(y_test))

    # 4. Học cách điền thiếu và mã hóa từ Train
    numeric = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    binary = ['sex', 'fbs', 'exang']
    categorical = ['cp', 'restecg', 'slope', 'ca', 'thal']
    numeric_transform = SimpleImputer(strategy='median')
    numeric_transform = make_pipeline(numeric_transform, StandardScaler())
    category_transform = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    )
    preprocess = ColumnTransformer([
        ('numeric', numeric_transform, numeric),
        ('binary', SimpleImputer(strategy='most_frequent'), binary),
        ('category', category_transform, categorical),
    ])
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    # 5. Khai báo mô hình, huấn luyện và dự đoán
    gnb = GaussianNB(var_smoothing=0.1)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    # 6. In dự đoán, nhãn thật và kết quả
    print('Print results for test data points:')
    print('Predicted labels: ', y_pred[:20])
    print('Ground truth    : ', np.array(y_test.iloc[:20]))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of Gaussian Naive Bayes: %.2f %%' % (100 * accuracy))
    print('Classification Report:\n{}'.format(
        classification_report(y_test, y_pred, zero_division=0)
    ))

    # 7. Lưu ma trận nhầm lẫn và biểu đồ hiệu năng
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    disease = report['1']
    roc_auc = roc_auc_score(y_test, gnb.predict_proba(X_test)[:, 1])
    metrics = {
        'accuracy': accuracy,
        'precision': disease['precision'],
        'recall': disease['recall'],
        'f1': disease['f1-score'],
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }
    save_model_charts('Gaussian Naive Bayes', metrics, root / 'Bayes')
    print('ROC-AUC: %.3f' % roc_auc)
    print('Charts: Bayes/')
