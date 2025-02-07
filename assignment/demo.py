import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the grayscale image
image_path = "peppers.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define different intensity ranges
N_values = [255, 128, 64, 32, 16, 8]

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

# Process and display images
for i, N in enumerate(N_values):
    scaled_image = (image.astype(np.float32) * N / 255).astype(np.uint8)
    axes[i].imshow(scaled_image, cmap='gray', vmin=0, vmax=N)
    axes[i].set_title(f"N = {N}")
    axes[i].axis('off')
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
# Show the images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 15),squeeze=False)
fig.subplots_adjust(hspace=0.5, wspace=0.3)
axes = axes.ravel()

# Process and display images
for i, N in enumerate(N_values):
    scaled_image = (image.astype(np.float32) * N / 255).astype(np.uint8)
    #  plot the histogram
    axes[i].hist(scaled_image.ravel(), bins=N, range=[0, N], color='gray', alpha=0.7)
    axes[i].set_xlim([0, N])
    axes[i].set_title(f"Histogram for N = {N}", pad=10)
    axes[i].set_ylabel("Frequency",labelpad=10)
# Show the images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()
