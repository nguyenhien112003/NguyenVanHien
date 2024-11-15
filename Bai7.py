import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

# Load và tiền xử lý dữ liệu IRIS
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Load và tiền xử lý dữ liệu CIFAR-10
(X_cifar10_train, y_cifar10_train), (X_cifar10_test, y_cifar10_test) = cifar10.load_data()
X_cifar10_train, X_cifar10_test = X_cifar10_train / 255.0, X_cifar10_test / 255.0  # Chuẩn hóa
y_cifar10_train_cat, y_cifar10_test_cat = to_categorical(y_cifar10_train), to_categorical(y_cifar10_test)

# Hàm đánh giá hiệu suất mô hình (chỉ áp dụng cho mô hình sklearn)
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, fit_time

# Đánh giá KNN và SVM trên IRIS
knn = KNeighborsClassifier(n_neighbors=3)
iris_knn_accuracy, iris_knn_time = evaluate_model(knn, X_iris_train, y_iris_train, X_iris_test, y_iris_test)

svm = SVC(kernel='linear')
iris_svm_accuracy, iris_svm_time = evaluate_model(svm, X_iris_train, y_iris_train, X_iris_test, y_iris_test)

# Đánh giá ANN trên IRIS
print("Huấn luyện ANN trên IRIS...")
ann_iris = Sequential([
    Dense(64, activation='relu', input_shape=(X_iris.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
ann_iris.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
ann_iris.fit(X_iris_train, y_iris_train, epochs=50, batch_size=5, verbose=0)
iris_ann_time = time.time() - start_time
iris_ann_accuracy = ann_iris.evaluate(X_iris_test, y_iris_test, verbose=0)[1]

# Đánh giá ANN trên CIFAR-10
print("Huấn luyện ANN trên CIFAR-10...")
ann_cifar = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
ann_cifar.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
ann_cifar.fit(X_cifar10_train, y_cifar10_train_cat, epochs=10, batch_size=32, verbose=0)
cifar_ann_time = time.time() - start_time
cifar_ann_accuracy = ann_cifar.evaluate(X_cifar10_test, y_cifar10_test_cat, verbose=0)[1]

# Hiển thị kết quả
# Hiển thị kết quả
print(f"IRIS KNN: Độ chính xác = {iris_knn_accuracy:.4f}, Thời gian = {iris_knn_time:.4f}s")
print(f"IRIS SVM: Độ chính xác = {iris_svm_accuracy:.4f}, Thời gian = {iris_svm_time:.4f}s")
print(f"CIFAR-10 ANN: Độ chính xác = {cifar_ann_accuracy:.4f}")
print(f"CIFAR-10 ANN: Độ chính xác = {cifar_ann_accuracy:.4f}, Thời gian = {cifar_ann_time:.4f}s")

