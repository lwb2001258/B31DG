import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load CIFAR-10 Dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Convert dataset to numpy
def dataset_to_numpy(dataset):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(data_loader))
    return images.permute(0, 2, 3, 1).numpy(), labels.numpy()

train_images, train_labels = dataset_to_numpy(trainset)
test_images, test_labels = dataset_to_numpy(testset)

# Convert images to grayscale
train_images_gray = np.array([cv2.cvtColor(img*255, cv2.COLOR_RGB2GRAY).astype(np.uint8)  for img in train_images])
test_images_gray = np.array([cv2.cvtColor(img*255, cv2.COLOR_RGB2GRAY).astype(np.uint8)  for img in test_images])
# 3️⃣ Extract SIFT features
sift = cv2.SIFT_create()
# Extract SIFT Features
# def extract_sift_features(images):
#     sift = cv2.SIFT_create()
#     descriptors_list = []
#     for img in images:
#         keypoints, descriptors = sift.detectAndCompute(img, None)
#         if descriptors is not None:
#             descriptors_list.append(descriptors)
#     return descriptors_list
#
# train_descriptors = extract_sift_features(train_images_gray)
# test_descriptors = extract_sift_features(test_images_gray)
#
# # Stack all descriptors for clustering
# all_descriptors = np.vstack(train_descriptors)
def extract_sift_features(images):
    descriptors_list = []
    for img in tqdm(images, desc="Extracting SIFT"):
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

train_descriptors = extract_sift_features(train_images_gray)
test_descriptors = extract_sift_features(test_images_gray)

# 4️⃣ Stack all descriptors for clustering
all_descriptors = np.vstack([desc for desc in train_descriptors if desc is not None])



# Function to generate codebook using K-Means
def generate_codebook(descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(descriptors)
    return kmeans

# Define codebook sizes to test
codebook_sizes = [50, 100, 200]
codebooks = {k: generate_codebook(all_descriptors, k) for k in codebook_sizes}
# Function to compute BoVW histograms
def compute_bovw_histograms(descriptors_list, kmeans):
    histograms = []
    valid_indices = []
    for idx,descriptors in enumerate(descriptors_list):
        if descriptors is not None:
            labels = kmeans.predict(descriptors)
            hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(hist)
            valid_indices.append(idx)
    return normalize(np.array(histograms), norm='l1'), valid_indices

# Generate BoVW histograms for training and testing
bovw_train= {k: compute_bovw_histograms(train_descriptors, codebooks[k]) for k in codebook_sizes}
bovw_test= {k: compute_bovw_histograms(test_descriptors, codebooks[k]) for k in codebook_sizes}
# Train SVM Classifier and Evaluate Performance
def train_svm_classifier(train_data, train_labels, test_data, test_labels):
    clf = SVC(kernel='linear')
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

# Evaluate performance for different codebook sizes
accuracy_results = {}
for k in codebook_sizes:
    accuracy_results[k] = train_svm_classifier(bovw_train[k], train_labels, bovw_test[k], test_labels)

# Print accuracy results
print("Classification Performance for Different Codebook Sizes:")
for k, acc in accuracy_results.items():
    print(f"Codebook Size {k}: Accuracy = {acc:.4f}")
