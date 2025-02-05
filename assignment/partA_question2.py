import cv2
import matplotlib.pyplot as plt


"""
Part A: Image Loading and Basic Processing
Q2: Geometric Transformations

1. perform a translation on the image using a translation matrix with tx = 50 and ty=30. Display the translated image.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt




# Load the image
image = cv2.imread("girl.jpg")

# Convert from BGR to RGB for correct color display in Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width = image.shape[:2]

# Define the translation matrix (tx = 50, ty = 30)
tx, ty = 50, 30
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation using warpAffine
translated_image = cv2.warpAffine(image_rgb, translation_matrix, (width + tx, height + ty))

# Display the original and translated images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Show the original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")


# Show the translated image
axes[1].imshow(translated_image)
axes[1].set_title("Translated Image (Right 50px, Down 30px)")
axes[1].axis("off")

# Display both images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()



"""
2. Define a rotation matrix for rotating the image by an angle of 45 degrees. Rotate the image and display the result.
"""


# Load the image
image = cv2.imread("girl.jpg")

# Convert from BGR to RGB (for correct color display in Matplotlib)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width = image.shape[:2]

# Define the center of rotation (center of the image)
center = (width // 2, height // 2)

# Define the rotation matrix (angle = 45 degrees, scale = 1)
angle = 45
scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Compute the new bounding dimensions of the rotated image
cos_val = abs(rotation_matrix[0, 0])
sin_val = abs(rotation_matrix[0, 1])
new_width = int((height * sin_val) + (width * cos_val))
new_height = int((height * cos_val) + (width * sin_val))

# Adjust the rotation matrix to take into account the translation
rotation_matrix[0, 2] += (new_width / 2) - center[0]
rotation_matrix[1, 2] += (new_height / 2) - center[1]

# Apply the rotation using warpAffine
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (new_width, new_height))

# Display the original and rotated images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Show the original image
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Show the rotated image
axes[1].imshow(rotated_image)
axes[1].set_title("Rotated Image (45°)")
axes[1].axis("off")

# Display both images
plt.show(block=False)
# waiting for the use to press the enter key to close the plot window
input("close the plot window if in pycharm and Press Enter to continue ... ")
# close the plot window
plt.close()
