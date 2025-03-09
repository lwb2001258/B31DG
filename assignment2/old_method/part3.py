import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1️⃣  读取 CIFAR-10 数据
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

# 2️⃣  将图像转换为灰度图
def convert_to_gray(images):
    return np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])

X_train_gray = convert_to_gray(X_train)
X_test_gray = convert_to_gray(X_test)

# 3️⃣  提取 SIFT 特征
sift = cv2.SIFT_create()

def extract_sift_features(images):
    descriptors_list = []
    for img in tqdm(images, desc="Extracting SIFT"):
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

train_descriptors = extract_sift_features(X_train_gray)
test_descriptors = extract_sift_features(X_test_gray)

# 4️⃣  合并所有描述子进行 KMeans 聚类
all_descriptors = np.vstack([desc for desc in train_descriptors if desc is not None])

# 5️⃣  训练不同 Codebook 大小的 KMeans 模型
codebook_sizes = [50, 100, 200]
results = {}

for k in codebook_sizes:
    print(f"Training KMeans for Codebook Size {k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    # 6️⃣  计算 BoVW 直方图
    def compute_bovw_histograms(descriptors_list, kmeans):
        histograms = []
        for descriptors in descriptors_list:
            if descriptors is None:
                histograms.append(np.zeros(kmeans.n_clusters))
                continue
            labels = kmeans.predict(descriptors)
            histogram, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(histogram)
        return np.array(histograms)

    X_train_bovw = compute_bovw_histograms(train_descriptors, kmeans)
    X_test_bovw = compute_bovw_histograms(test_descriptors, kmeans)

    # 7️⃣  使用 KNN 进行分类
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_bovw, y_train)
    y_pred = knn.predict(X_test_bovw)
    acc = accuracy_score(y_test, y_pred)

    print(f"Codebook Size {k}: KNN Accuracy = {acc:.4f}")
    results[k] = acc

# 8️⃣  绘制不同 Codebook 大小的分类准确率
plt.figure(figsize=(8, 5))
plt.plot(codebook_sizes, list(results.values()), marker='o', linestyle='-')
plt.xlabel("Codebook Size")
plt.ylabel("KNN Classification Accuracy")
plt.title("Effect of Codebook Size on Classification Performance")
plt.grid()
plt.show()
