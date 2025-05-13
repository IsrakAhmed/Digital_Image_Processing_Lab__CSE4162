# cv2.calcHist([img],[0],None,[256],[0,256])


import numpy as np
import matplotlib.pyplot as plt
import cv2
img=cv2.imread('imgg/images.jpg',0)
image=cv2.resize(img,(512,512))
print(np.max(image[:]))
height,width=image.shape

enhancement=250
left=150
right=255
tmp_image=np.copy(image)

for i in range(height):
    for j in range(width):
        if left<=image[i,j] and right>=image[i,j]:
            tmp_image[i,j]+=enhancement


# min = np.min(tmp_image[:])
print(np.max(tmp_image))
# normalized_img =(tmp_image - min)//(max-min)
# output_img = np.uint8(normalized_img*255)
# plt.subplot(2,1,1)
# plt.imshow(image,cmap='gray')
# plt.title('gray image')
# plt.axis('off')
# plt.subplot(2,1,2)
# plt.imshow(tmp_image, cmap='gray')
# plt.title('enhanced image')
# plt.axis('off')
# plt.tight_layout()
# plt.show()


# last check right