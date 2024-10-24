import cv2
import numpy as np
from matplotlib import pyplot as plt

# Đọc ảnh tải lên ở chế độ xám (grayscale)
image_path = 'D:/pyhton/images.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Áp dụng toán tử Sobel theo hướng x và y
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel theo hướng x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel theo hướng y

# Kết hợp các gradient Sobel theo hướng x và y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Chuẩn hóa kết quả Sobel để hiển thị
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

# Áp dụng Laplacian of Gaussian (LoG)
log = cv2.GaussianBlur(image, (3, 3), 0)
laplacian = cv2.Laplacian(log, cv2.CV_64F, ksize=3)

# Chuẩn hóa kết quả Laplacian để hiển thị
laplacian_normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

# Vẽ biểu đồ kết quả
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Ảnh gốc')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Phát hiện biên bằng Sobel')
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Phát hiện biên bằng Laplacian Gaussian')
plt.imshow(laplacian_normalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
