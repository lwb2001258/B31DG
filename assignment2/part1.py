import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
1. Data Exploration and Preprocessing (10 marks)
1. Download the CIFAR-10 dataset and visualize a sample of images from each class to familiarize
yourself with the data. Include the samples in the report.
2. Preprocess the images by converting them to grayscale to simplify the feature extraction process.
"""

# Define class names for CIFAR-10
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Download CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Create a dictionary to store one image per class
class_samples = {}

# Loop through dataset to collect one sample per class
for img, label in dataset:
    if label not in class_samples:
        class_samples[label] = img
    if len(class_samples) == 10:
        break

# Plot the sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, (label, img) in enumerate(class_samples.items()):
    ax = axes[i // 5, i % 5]
    img = img.numpy().transpose((1, 2, 0))  # Convert from Tensor format
    img = np.clip(img, 0, 1)  # Ensure values are within [0,1]
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(class_names[label])
    ax.axis("off")

plt.tight_layout()
# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

images, labels = class_samples.values(), class_samples.keys()

#Convert tensor images to numpy format
images_np = images.permute(0, 2, 3, 1).numpy()  # Convert (C, H, W) → (H, W, C)

# Convert images to grayscale
images_gray = np.array([cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in images_np])


# Display grayscale samples
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_gray[i], cmap="gray")
    plt.title(class_names[labels[i].item()])
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

#
# # # Load the image
# image_path = "girl.jpg"  # Replace with your image path
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# print(type(image))
# print(image.shape)
# sift = cv2.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(image, None)

#
# if image is None:
#     raise FileNotFoundError("Image not found. Check the path.")

plt.figure(figsize=(10, 5))
for idx, img in enumerate(images_gray):
    sift = cv2.SIFT_create()



    keypoints, descriptors = sift.detectAndCompute(img, None)
    # Draw keypoints on the image
    output_image = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.subplot(2, 5, idx + 1)
    plt.imshow(output_image, cmap='gray')
    plt.title(class_names[labels[idx].item()])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # Print some details
    print(f"Detected {len(keypoints)} keypoints.")
    print(f"Descriptor shape: {descriptors.shape}")  # (num_keypoints, 128 for SIFT, variable for SURF)

plt.tight_layout()
# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

#
#
#
#
#
#
