import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('imgg/line.jpg', cv2.IMREAD_GRAYSCALE)

# Define a custom line detection mask (e.g., a vertical line)
line_mask = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])

# Perform convolution to detect lines
result_image = np.zeros_like(image)
height, width = image.shape
mask_size = line_mask.shape[0]

for i in range(1, height - 1):
    for j in range(1, width - 1):
        tmp_window = image[i - 1:i + 2, j - 1:j + 2]
        convolution_result = np.sum(tmp_window * line_mask)
        result_image[i, j] = convolution_result

# Display the original and line-detected images
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(result_image, cmap='gray')
plt.title('Line-Detected Image')
plt.axis('off')
plt.show()
