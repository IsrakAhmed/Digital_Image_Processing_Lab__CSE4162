import numpy as np
import cv2  # OpenCV is used for visualization purposes
import matplotlib.pyplot as plt
import math
def upsampling(image,scale):
    h,w=image.shape
    h=h*scale
    w=w*scale
    resized_image=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            resized_image[i,j]=image[i//scale, j//scale]
    return resized_image


def downsample(image, scale):
    h, w = image.shape
    h,w= h // scale, w // scale  #new height and new width
    resized_image = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            resized_image[i, j] = image[i * scale, j * scale]
    
    return resized_image


image = cv2.imread('imgg/a.png',0)
img=cv2.resize(image,(512,512))
height,width=img.shape

dim=3
plt.subplot(3,3,1)
plt.imshow(img,cmap='gray')
plt.title('512 X 512')
plt.axis('off')
# plt.tight_layout()
for k in range(1,7):
    img=downsample(img,2)
    height,width=img.shape		 
    plt.subplot(dim,dim,k+1)
    plt.imshow(img,cmap='gray')
    plt.title(f'{height} X {width}')
    plt.axis('off')

plt.tight_layout()
plt.show()

#   checked in 3.30 pm is alright

