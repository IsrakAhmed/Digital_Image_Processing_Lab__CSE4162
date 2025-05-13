import numpy as np
import cv2  # OpenCV is used for visualization purposes
import matplotlib.pyplot as plt
import math
# Load the image
image = cv2.imread('imgg/images.jpg',0)
img=cv2.resize(image,(512,512))
height,width=img.shape
level=8
dim=3
plt.subplot(dim,dim,1)
plt.imshow(img,cmap='gray')
plt.title('8')
plt.axis('off')
# tmp_img= np.zeros((height, width), dtype=np.double)

for i in range(1,level):
    tmp_img=img>>i    
    plt.subplot(dim,dim,i+1)
    plt.imshow(tmp_img,cmap='gray')
    plt.title(f'{level-i} bit')
    plt.axis('off')
    plt. subplots_adjust(wspace=0.2)

plt.tight_layout()
plt.show()

# last checked in 3.32 pm is alright