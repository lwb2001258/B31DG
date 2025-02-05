import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取图像 (使用 OpenCV 读取，然后转换为 RGB 格式)
image = cv2.imread('girl.jpg')  # 替换为你的图像路径
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像的高度和宽度
h, w, _ = image.shape

# 生成原始坐标网格 (齐次坐标)
y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

# 将坐标转换为齐次坐标 (3xN 矩阵)
homogeneous_coords = np.stack([x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices).ravel()], axis=0)

# 定义平移变换矩阵 (dx=50, dy=30)
dx, dy = 50, 30
T = np.array([
    [1, 0, dx],
    [0, 1, dy],
    [0, 0, 1]
])

# 进行矩阵点乘 (应用变换)
transformed_coords = np.dot(T, homogeneous_coords)

# 提取变换后的 x' 和 y' 坐标
x_transformed = transformed_coords[0, :].reshape(h, w)
y_transformed = transformed_coords[1, :].reshape(h, w)

# 创建一个新的图像并填充平移后的像素
translated_image = np.zeros_like(image)

# 确保目标坐标在图像范围内
valid_x = (x_transformed >= 0) & (x_transformed < w)
valid_y = (y_transformed >= 0) & (y_transformed < h)
valid_indices = valid_x & valid_y

# 赋值像素
translated_image[y_transformed[valid_indices].astype(int), x_transformed[valid_indices].astype(int)] = image[y_indices[valid_indices], x_indices[valid_indices]]

# 显示原始图像和平移后的图像
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(translated_image)
ax[1].set_title("Translated Image")
ax[1].axis("off")

plt.show()
