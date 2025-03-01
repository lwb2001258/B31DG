import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
Part B: Image Analysis Using Peppers.png 
Q4: Intensity Range Adjustment and Histogram Equalization
"""

"""
1. Load and display the color image ‘peppers.png’ (build-in in MATLAB or
download it from online source for Python users) and examine and report the size of the
image (width, height, and number of channels).
"""


# Load the image (make sure 'peppers.png' is in the same directory or provide the correct path)
image = cv2.imread("peppers.png")
# display the image
cv2.imshow("Peppers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Get image dimensions: height, width, number of channels
height, width, channels = image.shape
# Report the size
print(f"Width: {width}, Height: {height}, Number of channels: {channels}")

"""
2. Convert the color image to grayscale and display the grayscale image in its full
intensity range ([0,255]).
"""

# Load the color image
image = cv2.imread("peppers.png")
# Convert BGR to RGB for correct display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display original and grayscale images
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Original color image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Color Image")
axes[0].axis("off")
axes[0].set_xticks([])
axes[0].set_yticks([])

# Grayscale image
axes[1].imshow(gray_image, cmap="gray", vmin=0, vmax=255)  # Ensure full intensity range
axes[1].set_title("Grayscale Image (0-255)")
axes[1].axis("off")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
# Show the images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()


"""
3.  Reduce the intensity range of the grayscale image to a lower range ([0,N]) for
values of N ranging from 255 to 8. Display the resulting images.
"""

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



"""
2. Apply histogram equalization to the grayscale image. Display the result and
compare it to the original grayscale image. Discuss the differences in brightness and
contrast.
"""


# Load the image and convert to grayscale
image = cv2.imread("peppers.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to the grayscale image
equalized_image = cv2.equalizeHist(gray_image)

# Create a figure with two subplots to compare original and equalized images
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Display the original grayscale image
axes[0].imshow(gray_image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title("Original Grayscale Image")
axes[0].axis('off')
axes[0].set_xticks([])
axes[0].set_yticks([])

# Display the equalized grayscale image
axes[1].imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
axes[1].set_title("Equalized Grayscale Image")
axes[1].axis('off')
axes[1].set_xticks([])
axes[1].set_yticks([])



# Show the images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()


# Optionally, print intensity range to compare
print(f"Original Grayscale image intensity range: [{gray_image.min()}, {gray_image.max()}]")
print(f"Equalized Grayscale image intensity range: [{equalized_image.min()}, {equalized_image.max()}]")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of original grayscale image
axes[0].hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
axes[0].set_title("Histogram of Original Grayscale Image")
axes[0].set_xlabel("Pixel Intensity")
axes[0].set_ylabel("Frequency")

# Histogram of equalized grayscale image
axes[1].hist(equalized_image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
axes[1].set_title("Histogram of Equalized Grayscale Image")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")

# Show the images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

