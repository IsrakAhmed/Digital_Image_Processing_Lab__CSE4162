import matplotlib.pyplot as plt
import numpy as np
import cv2
load_image=cv2.imread('imgg/cameraman.jpg',0)
image=cv2.resize(load_image,(512,512))
height,width=image.shape

def histogram_cal(image):
    histogram=[0]*256
    for height in image:
        for pixel in height:
            histogram[pixel]+=1
    return histogram

def threshold_cal(histogram):
    threshold_point = np.sum(histogram) // 2
    threshold_intensity = 0

    while True:
        if (threshold_point <= 0):
            break
        threshold_point -= histogram[threshold_intensity]
        threshold_intensity += 1
    return threshold_intensity
histo=histogram_cal(image)
threshold=threshold_cal(histo)
output=(image>threshold).astype(np.uint8)*255
equal_histo=histogram_cal(output)


plt.subplot(3,1,1)
plt.imshow(image,cmap='gray')
plt.title('Original image')

plt.subplot(3,1,2)
plt.bar(range(256), histo, width=1.0)
plt.subplot(3,1,3)
plt.bar(range(256), equal_histo, width=1.0)


# max,min=np.max(output[:]), np.min(output[:])
# print(max,min)

# plt.subplot(3,1,3)
# plt.imshow(output,cmap='gray')
plt.show()