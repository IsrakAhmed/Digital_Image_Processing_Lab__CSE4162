import matplotlib.pyplot as plt
import numpy as np
import cv2

st_elemen = np.ones((5,5), np.uint8) * 255
st_elemen2 = np.ones((3,3), np.uint8) * 255
size = st_elemen[0].size
size2 = st_elemen2[0].size


image = cv2.imread('imgg/ahnaf.tif', 0)
image = cv2.resize(image, (512, 512))

height, width = image.shape


def erosion_op(image, st_elemen,size):
    pad_h = size // 2
    padded_image = np.pad(image, pad_h, mode='constant')

    erosion_img = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            tmp_window = padded_image[i:i + size, j:j + size]
            if np.array_equal(np.bitwise_and(tmp_window, st_elemen), st_elemen):
                erosion_img[i, j] = 255
           

    return erosion_img

erosion_img =image -  erosion_op(image, st_elemen,size)
erosion_img2 =image -  erosion_op(image, st_elemen2,size2)

plt.subplot(121)
plt.imshow(erosion_img2, cmap='gray')
plt.title('original image')
plt.axis('off')

plt.subplot(122)
plt.imshow(erosion_img, cmap='gray')
plt.title('erosion image')
plt.axis('off')

plt.tight_layout()
plt.show()