import cv2
import matplotlib.pyplot as plt


"""
Part A: Image Loading and Basic Processing
"""
"""
# Q1: Image Loading and Conversion
# 1.  Load an image using Python or MATLAB. Display the image and provide the  code.
"""

# Load the image
image = cv2.imread('girl.jpg')
# create a window which can be resized
# cv2.namedWindow('beautiful girl', cv2.WINDOW_NORMAL)
# # resize the window
# cv2.resizeWindow('beautiful girl', 225, 225)
# Display the image
cv2.imshow('beautiful girl', image)
# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
2. Convert the image to grayscale and HSL/HSV color spaces. Display both converted images.
"""
# For my understanding, I will try to do two type, one displaying of grayscale and HSL, another for grayscale and HSV.

# Load the image
# image = cv2.imread('girl.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to HSL color space
hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imshow("gray_image",gray_image)
cv2.imshow("hsl_image",hsl_image)
cv2.imshow("hsv_image",hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






# 3. Binarize the grayscale image using a threshold based on intensity or hue.  Display the binarized image and explain your threshold selection in python



# Load the grayscale image
# image = cv2.imread("girl.jpg", cv2.IMREAD_GRAYSCALE)
#
# Apply binary thresholding
threshold_value = 127
# Change this value based on brightness
_, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

cv2.imshow("gray_image",gray_image)
cv2.imshow("binary_image",binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

















