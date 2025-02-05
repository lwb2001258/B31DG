import cv2
import matplotlib.pyplot as plt

"""
Part A: Image Loading and Basic Processing
Q3: Smoothing Filters and Edge Detection
1. Apply a mean filter to the input image. Change the kernel size and observe its effect on the image.
"""

# Load the image
image = cv2.imread("girl.jpg")

# Convert from BGR to RGB for correct color display in Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply mean filters with different kernel sizes
kernel_sizes = [(3, 3), (7, 7), (15, 15)]
filtered_images = [cv2.blur(image_rgb, ksize) for ksize in kernel_sizes]

# Display original and filtered images
fig, axes = plt.subplots(1, 4, figsize=(30, 10))

# Show original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[0].set_xticks([])
axes[0].set_yticks([])


# Show filtered images
for i, (ksize, img) in enumerate(zip(kernel_sizes, filtered_images)):
    axes[i + 1].imshow(img)
    axes[i + 1].set_title(f"Mean Filter {ksize}")
    axes[i + 1].axis("off")
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])


# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

"""
From the plot, we can see the following:
 The mean filter is useful for blurring and noise reduction, but it removes fine details and blurs edges.
 Larger kernels create stronger smoothing effects but also cause more detail loss.
"""
""""
2. Apply a Gaussian filter to the input image. Experiment with different standard deviations (σ) and describe how changing σ influences the result


"""
# Load the image
image = cv2.imread("girl.jpg")

# Convert from BGR to RGB for correct color display in Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Gaussian blur with different standard deviations (σ)
sigma_values = [1, 3, 5, 10]
blurred_images = [cv2.GaussianBlur(image_rgb, (9, 9), sigma) for sigma in sigma_values]

# Display original and blurred images
fig, axes = plt.subplots(1, 5, figsize=(30, 10))

# Show original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[0].set_xticks([])
axes[0].set_yticks([])

# Show blurred images for different sigma values
for i, (sigma, img) in enumerate(zip(sigma_values, blurred_images)):
    axes[i + 1].imshow(img)
    axes[i + 1].set_title(f" σ={sigma}")
    axes[i + 1].axis("off")
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])

# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()
"""
From the plot we can see the following:
 Gaussian filtering smooths images while retaining more structure than a mean filter.
 Higher σ leads to stronger blurring.
 Choose σ carefully based on how much detail you want to preserve.
"""

"""
3. Apply a Canny edge detector to the grayscale image. Display the result and analyze the impact of different threshold values on the detected edges.
"""

# Load the image
image = cv2.imread("girl.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection with different threshold values
thresholds = [(50, 150), (100, 200), (150, 250)]
edges_list = [cv2.Canny(gray, t1, t2) for t1, t2 in thresholds]

# Display original and edge-detected images
fig, axes = plt.subplots(1, 4, figsize=(30, 10))

# Show original grayscale image
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Grayscale")
axes[0].axis("off")
axes[0].set_xticks([])
axes[0].set_yticks([])

# Show Canny edge-detected images with different thresholds
for i, (t, edges) in enumerate(zip(thresholds, edges_list)):
    axes[i + 1].imshow(edges, cmap="gray")
    axes[i + 1].set_title(f"(T1={t[0]}, T2={t[1]})")
    axes[i + 1].axis("off")
    axes[i + 1].set_xticks([])
    axes[i + 1].set_yticks([])

# Display all images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()

"""
For Threshold Values (T1, T2),lower threshold (T1) Determines weak edges, while higher threshold (T2): Defines strong edges.
From the plot, we can see the following:
Low thresholds → Preserve fine edges but introduce noise.
Moderate thresholds → A balanced result with clear object boundaries.
High thresholds → Best for sharp edges, but loses small details.

Adaptive Strategy: Experiment with different values based on your image’s noise level and desired level of detail.


"""



