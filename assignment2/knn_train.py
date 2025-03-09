import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1️⃣ Load CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Convert dataset to NumPy arrays
X_train = trainset.data
y_train = np.array(trainset.targets)
X_test = testset.data
y_test = np.array(testset.targets)

# 2️⃣ Convert images to grayscale
def convert_to_gray(images):
    return np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])

X_train_gray = convert_to_gray(X_train)
X_test_gray = convert_to_gray(X_test)

# 3️⃣ Extract SIFT features
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

# 4️⃣ Stack all descriptors for clustering
all_descriptors = np.vstack([desc for desc in train_descriptors if desc is not None])

# 5️⃣ Train KMeans with different codebook sizes
codebook_sizes = [50, 100, 150,200]
results = {}

for k in codebook_sizes:
    print(f"Training KMeans for Codebook Size {k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    # 6️⃣ Convert images into BoVW histograms
    def compute_bovw_histograms(descriptors_list, kmeans):
        histograms = []
        valid_indices = []  # Track which images have valid descriptors
        for idx, descriptors in enumerate(descriptors_list):
            if descriptors is None:
                histograms.append(np.zeros(kmeans.n_clusters))  # Fill empty images with zero histograms
            else:
                labels = kmeans.predict(descriptors)
                histogram, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
                histograms.append(histogram)
                valid_indices.append(idx)  # Store valid index
        return np.array(histograms), valid_indices

    X_train_bovw, valid_train_indices = compute_bovw_histograms(train_descriptors, kmeans)
    X_test_bovw, valid_test_indices = compute_bovw_histograms(test_descriptors, kmeans)

    # Filter labels to match histograms
    y_train_filtered = y_train[valid_train_indices]
    y_test_filtered = y_test[valid_test_indices]

    # 7️⃣ Train an SVM classifier
    print(f"Training SVM for Codebook Size {k}...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train_bovw, y_train_filtered)
    y_pred = svm.predict(X_test_bovw)
    acc = accuracy_score(y_test_filtered, y_pred)

    print(f"Codebook Size {k}: SVM Accuracy = {acc:.4f}")
    results[k] = acc

# 8️⃣ Plot accuracy for different codebook sizes
plt.figure(figsize=(8, 5))
plt.plot(codebook_sizes, list(results.values()), marker='o', linestyle='-')
plt.xlabel("Codebook Size")
plt.ylabel("SVM Classification Accuracy")
plt.title("Effect of Codebook Size on Classification Performance")
plt.grid()
plt.show()
