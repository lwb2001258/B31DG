import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import urllib.request
import zipfile
import tarfile
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

"""
1. Data Exploration and Preprocessing (10 marks)
1. Download the CIFAR-10 dataset and visualize a sample of images from each class to familiarize
yourself with the data. Include the samples in the report.
2. Preprocess the images by converting them to grayscale to simplify the feature extraction process.
"""

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Select one image per class
num_classes = len(class_names)
sample_images = np.zeros((num_classes, 32, 32, 3), dtype=np.uint8)

for i in range(num_classes):
    sample_idx = np.where(y_train.flatten() == i)[0][0]  # Get first occurrence of each class
    sample_images[i] = x_train[sample_idx]

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()

for i in range(num_classes):
    axes[i].imshow(sample_images[i])
    axes[i].set_title(class_names[i])
    axes[i].axis('off')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.savefig("cifar10_samples.png", dpi=300)  # Save the figure for the report
plt.show()
input("press enter to continue...")

# sample_images_gray = []
# for image in sample_images:
#     sample_images_gray.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
#     cv2.imshow("gray",cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
#     input("press enter to continue...")
sample_images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in sample_images])

# Plot the images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()

for i in range(num_classes):
    axes[i].imshow(sample_images_gray[i], cmap="gray")
    axes[i].set_title(class_names[i])
    axes[i].axis('off')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.savefig("cifar10_samples_gray.png", dpi=300)  # Save the figure for the report
plt.show()
input("press enter to continue...")


"""
2. Feature Extraction (15 marks)
1. Utilize feature detectors such as SIFT (Scale-Invariant Feature Transform) or SURF (Speeded-Up
Robust Features) to identify keypoints in the images.
2. Extract feature descriptors around the detected keypoints to capture the local image information.
"""

def extract_sift_features(images):
    descriptors_list = []
    for img in images: # Convert to uint8
        keypoints, descriptors = sift.detectAndCompute(img, None)
        descriptors_list.append(descriptors if descriptors is not None else np.array([]))
    return descriptors_list

def create_histograms(descriptors_list, kmeans_model):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors.size > 0:
            words = kmeans_model.predict(descriptors)
            histogram, _ = np.histogram(words, bins=range(101), density=True)
        else:
            histogram = np.zeros(100)  # Empty feature case
        histograms.append(histogram)
    return np.array(histograms)
sift = cv2.SIFT_create()

plt.figure(figsize=(10, 5))
for idx, img in enumerate(sample_images_gray):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # Draw keypoints on the image
    output_image = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.subplot(2, 5, idx + 1)
    plt.imshow(output_image, cmap='gray')
    plt.title(class_names[idx])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # Print some details
    print(f"Detected {len(keypoints)} keypoints.")
    print(f"Descriptor shape: {descriptors.shape}")
plt.tight_layout()
plt.show(block=False)
input("Press any key to continue...")

"""
3. Codebook Generation (20 marks)
1. Apply the k-means clustering algorithm to the extracted descriptors to form a codebook (visual
vocabulary).
2. Experiment with different sizes of codebooks (e.g., 50, 100, 200 visual words) to observe the effect
on classification performance.
3. Represent each image by a histogram that quantifies the occurrence of each visual word from the
codebook.
"""




codebook_sizes = [50, 100, 150,200]
results = {}
kmeans_models = []



for codebook_size in codebook_sizes:
    train_descriptors = extract_sift_features(x_train)
    test_descriptors = extract_sift_features(x_test)
    all_descriptors = np.vstack([desc for desc in train_descriptors if desc.size > 0])
    kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)
    kmeans_models.append(kmeans)
    X_train_bovw = create_histograms(train_descriptors, kmeans)
    X_test_bovw = create_histograms(test_descriptors, kmeans)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_bovw, y_train)
    y_pred = svm_classifier.predict(X_test_bovw)
    accuracy = accuracy_score(y_test, y_pred)
    results[codebook_size] =  format(accuracy, ".4f")
    print(f"Codebook size: {codebook_size}, sModel Accuracy: {accuracy * 100:.2f}%")
plt.figure(figsize=(8, 5))
plt.plot(codebook_sizes, list(results.values()), marker='o', linestyle='-')
plt.xlabel("Codebook Size")
plt.ylabel("SVM Classification Accuracy")
plt.title("Effect of Codebook Size on Classification Performance")
plt.grid()
plt.show()
input("Press any key to continue...")


# Function to compute the histogram of visual words for an image
def compute_histogram(image, kmeans, num_clusters):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(num_clusters)  # Return empty histogram if no descriptors

    # Assign each descriptor to the closest cluster center
    labels = kmeans.predict(descriptors)

    # Compute the histogram
    hist, _ = np.histogram(labels, bins=np.arange(num_clusters+1), density=True)
    return hist

plt.figure(figsize=(16, 9))
plt.title("Histogram of different type of codebook of each class of picture")
for idx,codebook_size in enumerate(codebook_sizes):
    for idy, img in enumerate(sample_images_gray):
        image_histograms = np.array(compute_histogram(img, kmeans_models[idx], codebook_size))

        print("Shape of image histograms:", image_histograms.shape)

        # Plot histograms for different codebook sizes
        plt.subplot(len(codebook_sizes), len(sample_images_gray), idx*10+idy+1)
        plt.bar(range(codebook_size), image_histograms)
        # plt.title(f"Histogram (Codebook Size: {clusters})")
        # plt.xlabel("Visual Word Index")
        # plt.ylabel("Frequency")


plt.tight_layout()
plt.show()
input("Press any key to continue...")






