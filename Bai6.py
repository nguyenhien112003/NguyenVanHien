import cv2
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Đọc ảnh và kiểm tra
image1_path = 'D:/pyhton/nha1.jpg'
image2_path = 'D:/pyhton/nha2.jpg'

# Đọc ảnh và chuyển đổi sang ảnh mức xám
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None or image2 is None:
    print("Lỗi: Không thể tải một trong hai ảnh. Vui lòng kiểm tra đường dẫn.")
else:
    # Sử dụng ảnh đầu tiên để phân cụm
    image_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Chuẩn bị dữ liệu cho K-means và FCM
    data = image_gray.reshape(-1, 1)

    # Thực hiện phân cụm K-means
    k = 3  # Số cụm cần xác định
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    kmeans_labels = kmeans.labels_.reshape(image_gray.shape)

    # Thực hiện phân cụm FCM
    # Thiết lập số cụm và các thông số của FCM
    c = 3  # Số cụm
    fcm_data = data.T  # FCM cần dữ liệu ở dạng transpose (1 hàng, nhiều cột)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(fcm_data, c, 2, error=0.005, maxiter=1000, init=None)

    # Lấy chỉ số cụm dựa trên độ thành viên lớn nhất
    fcm_labels = np.argmax(u, axis=0).reshape(image_gray.shape)

    # Hiển thị kết quả
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Ảnh gốc')

    plt.subplot(1, 3, 2)
    plt.imshow(kmeans_labels, cmap='viridis')
    plt.title('K-means clustering')

    plt.subplot(1, 3, 3)
    plt.imshow(fcm_labels, cmap='viridis')
    plt.title('Fuzzy C-Means clustering')

    plt.show()
