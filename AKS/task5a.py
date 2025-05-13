import matplotlib.pyplot as plt
import numpy as np
import cv2

st_elemen = np.ones((3, 3), np.uint8) * 255
size = st_elemen[0].size

image = cv2.imread('imgg/noisy.tif', 0)
image = cv2.resize(image, (512, 512))

height, width = image.shape


def erosion_op(image, st_elemen):
    pad_h = size // 2
    padded_image = np.pad(image, pad_h, mode='constant')

    erosion_img = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            tmp_window = padded_image[i:i + size, j:j + size]
            if np.array_equal(np.bitwise_and(tmp_window, st_elemen), st_elemen):
                erosion_img[i, j] = 255
           

    return erosion_img


def dilation_op(image, st_element):
    pad_h = size // 2
    padded_image = np.pad(image, pad_h, mode='constant')

    dilation_img = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            tmp_window = padded_image[i:i + size, j:j + size]
            if np.any(np.bitwise_and(tmp_window, st_element)):
                dilation_img[i, j] = 255          
    return dilation_img



erosion_img = erosion_op(image, st_elemen)
dilation_img = dilation_op(erosion_img , st_elemen)
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('original image')
plt.axis('off')
# plt.subplot(222)
# plt.imshow(cv2.dilate(image,st_elemen),cmap='gray')
# plt.axis('off')
plt.subplot(223)
plt.imshow(erosion_img, cmap='gray')
plt.title('erosion image')
plt.axis('off')
plt.subplot(224)
plt.imshow(dilation_img, cmap='gray')
plt.title('dilation image')
plt.axis('off')
plt.tight_layout()
plt.show()
