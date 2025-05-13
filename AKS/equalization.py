import numpy as np
import matplotlib.pyplot as plt
import cv2

image= cv2.imread('imgg/cameraman.jpg', 0)
image=cv2.resize(image,(512,512))

height,width=image.shape

def histogram_cal(image):
    histogram=[0]*256
    for height in image:
        for pixel in height:
            histogram[pixel]+=1
    return histogram

def histogram_equalization(image,histogram):
    
    cdf=np.cumsum(histogram)
    # shifts the entire CDF curve downwards
    cdf_normalized=((cdf-cdf.min())*255)/(512*512) 
    equalized_image=cdf_normalized[image]

    return equalized_image.astype(np.uint8)

histogram_origin=histogram_cal(image)
equalized_image=histogram_equalization(image,histogram_origin)
# equalized_image=cv2. equalizeHist(image)
equalized_hist=histogram_cal(equalized_image)


plt.subplot(2,2,1)
plt.imshow(image,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(equalized_image,cmap='gray')
plt.subplot(2,2,3)
plt.bar(range(256),histogram_origin)

plt.subplot(2,2,4)
plt.bar(range(256),equalized_hist,width=1)
plt.show()