import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define CIFAR-10 class labels
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Define transformation: Convert to tensor and normalize
transform = transforms.Compose([transforms.ToTensor()])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)

# Get sample images and labels
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# print(images[0].shape)
# print(len(images))
#
# # # Create a dictionary to store one image per class
# class_samples = {}
#
# images_np = images.permute(0, 2, 3, 1).numpy()  # Convert (C, H, W) → (H, W, C)
# for index, label in enumerate(labels):
#     if label.item() not in class_samples.keys():
#         class_samples[label.item()] = images_np[index]
#     if len(class_samples) == 10:
#         break

# Create a dictionary to store one image per class
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
