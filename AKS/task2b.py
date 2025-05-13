import numpy as np
import cv2
import matplotlib.pyplot as plt
image= cv2.imread('imgg/cameraman.jpg', 0)
final_img=cv2.resize(image,(512,512))

height,width=final_img.shape

gama=0.6
 
c=255/(np.log(1+255))


inv_log=np.exp(final_img / c) -1
power_img=(final_img**gama)


plt.subplot(1,3,1)
plt.imshow(final_img,cmap='gray')
plt.title('original image')
plt.subplot(1,3,2)
plt.imshow(inv_log,cmap='gray') 
plt.title('inverse log image')
plt.subplot(1,3,3) 
plt.imshow(power_img,cmap='gray')
plt.title('power image ')
plt.show()