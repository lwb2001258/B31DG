import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


"""
1. Data Exploration and Preprocessing (10 marks)
1. Download the CIFAR-10 dataset and visualize a sample of images from each class to familiarize
yourself with the data. Include the samples in the report.
2. Preprocess the images by converting them to grayscale to simplify the feature extraction process.
"""

# Define CIFAR-10 class labels
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Define transformation: Convert to tensor and normalize
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
class_samples = {}

# Loop through dataset to collect one sample per class
for img, label in trainset:
    print(img.shape)
    print(label)
    if label not in class_samples:
        class_samples[label] =img.permute(1, 2, 0).numpy()
    if len(class_samples) == 10:
        break

print(len(class_samples))
print(class_samples.keys())
images, labels = next(iter(class_samples.values())), next(iter(class_samples.keys()))
# Visualize sample images
plt.figure(figsize=(24, 12))
for i, (label, image) in enumerate(class_samples.items()):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(class_names[label])
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
plt.suptitle("Sample CIFAR-10 Images")
plt.tight_layout()
# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

# Convert images to grayscale
images_gray = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in class_samples.values()])

# Display grayscale samples
plt.figure(figsize=(24, 12))
for i, label in enumerate(class_samples.keys()):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_gray[i], cmap="gray")
    plt.title(class_names[label])
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
plt.suptitle("Grayscale CIFAR-10 Images")
plt.tight_layout()
# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()


"""
2. Feature Extraction (15 marks)
1. Utilize feature detectors such as SIFT (Scale-Invariant Feature Transform) or SURF (Speeded-Up
Robust Features) to identify keypoints in the images.
2. Extract feature descriptors around the detected keypoints to capture the local image information.
"""
# Display grayscale samples
plt.figure(figsize=(24, 12))
for i, label in enumerate(class_samples.keys()):

    sift = cv2.SIFT_create()

    # Process a single sample image
    sample_image = images_gray[i]  # Use first image in the batch

    # Detect keypoints and extract descriptors
    keypoints, descriptors = sift.detectAndCompute(sample_image, None)

    # Draw keypoints
    output_image = cv2.drawKeypoints(sample_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.subplot(2, 5, i + 1)
    plt.imshow(output_image, cmap="gray")
    plt.title(class_names[label])
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")

plt.suptitle(f"Feature Detection with SIFT")
# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()


"""
3. Codebook Generation
1. Apply the k-means clustering algorithm to the extracted descriptors to form a codebook (visual
vocabulary).
2. Experiment with different sizes of codebooks (e.g., 50, 100, 200 visual words) to observe the effect
on classification performance.
3. Represent each image by a histogram that quantifies the occurrence of each visual word from the
codebook.
"""

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=False)

# Extract images and labels
data_iter = iter(train_loader)
images, labels = next(data_iter)
images = images.permute(0, 2, 3, 1).numpy()  # Convert to (N, H, W, C)

# Convert to grayscale
images_gray2 = np.array([cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in images])


# Extract SIFT features
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

train_descriptors = extract_sift_features(images_gray2)

# Stack all descriptors for clustering
all_descriptors = np.vstack(train_descriptors)

# K-means clustering for codebook generation
def generate_codebook(descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(descriptors)
    return kmeans

# Experiment with different codebook sizes
codebook_sizes = [50, 100, 200]
bovw_results = {}  # Store BoVW histograms for each codebook size
codebooks = {k: generate_codebook(all_descriptors, k) for k in codebook_sizes}


# Compute BoVW histograms
def compute_bovw_histograms(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors is not None:
            labels = kmeans.predict(descriptors)
            hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(hist)
    return normalize(np.array(histograms), norm='l1')


for codebook_size in codebook_sizes:
    print(f"Generating Codebook with {codebook_size} Visual Words...")

    # Train K-Means on all descriptors
    all_descriptors = np.vstack(train_descriptors)  # Stack all extracted descriptors
    kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10)
    kmeans.fit(all_descriptors)

    # Compute BoVW histograms
    bovw_train = compute_bovw_histograms(train_descriptors, kmeans)
    bovw_results[codebook_size] = bovw_train  # Store results


bovw_histograms = {k: compute_bovw_histograms(train_descriptors, codebooks[k]) for k in codebook_sizes}
# bovw_histograms_file = open('bovw_histograms.json', 'w')
# bovw_histograms_string = json.dumps(bovw_histograms)
# bovw_histograms_file.write(bovw_histograms_string)

# for codebook_size in codebook_sizes:
#     print(f"Generating Codebook with {codebook_size} Visual Words...")
#
#     # Train K-Means on all descriptors
#     all_descriptors = np.vstack(train_descriptors)  # Stack all extracted descriptors
#     kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10)
#     kmeans.fit(all_descriptors)
#
#     # Compute BoVW histograms
#     bovw_train = compute_bovw_histograms(train_descriptors, kmeans)
# for k, hist in bovw_histograms.items():
#
#     plt.figure(figsize=(10, 4))
#     plt.bar(range(codebook_sizes[k]), bovw_histograms[k][0])
#     plt.xlabel("Visual Word Index")
#     plt.ylabel("Frequency")
#     plt.title("BoVW Histogram with 50 Visual Words")
#     plt.show(block=False)
#     # waiting for the use to press the enter key to close the plot window
#     input("close the plot window if in pycharm and Press Enter to continue ... ")
#     # close the plot window
#     plt.close()
# bovw_results_file = open('bovw_results.json', 'w')
# bovw_results_string = json.dumps(bovw_results)
# bovw_results_file.write(bovw_results_string)
num_samples = 10  # Number of histograms to visualize
sample_indices = np.random.choice(len(bovw_train), num_samples, replace=False)

fig, axes = plt.subplots(len(codebook_sizes), num_samples,figsize=(25,16))

for row, codebook_size in enumerate(codebook_sizes):
    for col, idx in enumerate(sample_indices):
        ax = axes[row, col]
        ax.bar(range(codebook_size), bovw_results[codebook_size][idx], color='blue', alpha=0.7)
        ax.set_title(f"{codebook_size}")
        ax.set_xlabel("Visual Words")
        ax.set_ylabel("Frequency")
        ax.set_ylim(0, 1)



plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
# plt.get_current_fig_manager().full_screen_toggle()
plt.show()