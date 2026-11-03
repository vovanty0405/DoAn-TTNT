# 1. Khai báo các thư viện cần thiết
import pandas as pd
import numpy as np

# 2. Nạp thuật toán và các hàm đánh giá
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# 3. Đọc dữ liệu từ file CSV
# (Đổi lại đường dẫn phù hợp với máy của bạn)
data = pd.read_csv('D:/Năm 4/Trí tuệ nhân tạo-20260820T130944Z-1-001/DoAn/Data/heart.csv') 
print("Kích thước dữ liệu:", data.shape)
print(data.head())

# 4. Xử lý các cột dạng chữ nếu có (theo code của thầy)
le = preprocessing.LabelEncoder()
for column_name in data.columns:
    if data[column_name].dtype == object:
        data[column_name] = le.fit_transform(data[column_name])

# 5. Tách thuộc tính (X) và nhãn mục tiêu (y)
# Cột 'target' nằm ở cuối cùng
X = data.iloc[:, :-1]
y = data.iloc[:, -1].astype(int)

# 6. Chia tập Train và Test (70% Train, 30% Test hoặc 80/20)
# Thêm random_state để kết quả cố định khi báo cáo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))

# 7. Huấn luyện Decision Tree với tiêu chuẩn entropy
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 8. Dự đoán và đánh giá
y_pred = clf.predict(X_test)

print("\nKết quả dự đoán thử trên tập test:")
print("Predicted labels: ", y_pred[:20])
print("Ground truth    : ", y_test[:20].values)

print("\nAccuracy of Decision tree: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
print('\nClassification Report:\n{}'.format(classification_report(y_test, y_pred)))