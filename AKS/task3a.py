import numpy as np
import cv2
import matplotlib.pyplot as plt
image= cv2.imread('imgg/Characters Test Pattern 688x688.tif', 0)
image=cv2.resize(image,(512,512))
height,width=image.shape
origin=np.copy(image)

def addNoise(image):
    noise_image=np.copy(image)
    noise=0.2
    height,width=image.shape

    for h in range(height):
        for w in range(width):
            random_val=np.random.rand() # returns random value between 0 and 1
            if random_val<noise/2:
                noise_image[h,w]=0
            elif random_val<noise:
                noise_image[h,w]=255
    return noise_image


def padding_add(n,noise_image):
    
    mask = np.ones((n,n)) / (n*n)
    pad_height=n//2
    pad_width=n//2
    new_height=height+2*pad_height
    new_width=width+2*pad_width 
    pad_image=np.zeros((new_height,new_width))
    pad_image[pad_height:pad_height + height, pad_width:pad_width + width] = noise_image

    return pad_image,mask


def Average(noise_image,height,width):    
    n=3
    mask = np.ones((n,n))/ (n*n)
    spatial_image=np.zeros((height,width))  
    # pad_image,mask=padding_add(n,noise_image)
    pad_height=n//2
    pad_image=np.pad(noise_image,pad_height,mode='constant')

    for h in range(height):
        for w in range(width):
            tmp_window=pad_image[h:h+n,w:w+n]
            weight=np.sum(tmp_window*mask)
            spatial_image[h,w]=weight

    return spatial_image

def Median(noise_image,height,width):
    n=3
    mask = np.ones((n,n))
    spatial_image=np.zeros((height,width))  
    # pad_image,mask=padding_add(n,noise_image)
    pad_height=n//2
    pad_image=np.pad(noise_image,pad_height,mode='constant')
    for h in range(height):
        for w in range(width):
            tmp_window=pad_image[h:h+n,w:w+n]
            weight=np.median(tmp_window*mask)
            spatial_image[h,w]=weight

    return spatial_image

def PSNR(original,noisy):
   # representing a 64-bit floating-point number
   original=original.astype(np.float64)
   noisy=noisy.astype(np.float64)
   mse = np.mean((original - noisy) ** 2)
   max_pixel_value = 255
   psnr = 20 * np.log10(max_pixel_value/(np.sqrt(mse)))
   return psnr


noise_image=addNoise(origin)
psnr_val=PSNR(image,noise_image)
spatial_image=Average(noise_image,height,width)
spatial_image2=Median(noise_image,height,width)
med_original=PSNR(origin,spatial_image2)
avg_psnr=PSNR(origin,spatial_image)

plt.subplot(2,2,1)
plt.imshow(origin,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(noise_image,cmap='gray')
plt.title(f'psnr : {psnr_val:.2f}db')
plt.subplot(2,2,3)
plt.imshow(spatial_image,cmap='gray')
plt.title(f'avg - PSNR: {avg_psnr :.2f} dB')
plt.subplot(2,2,4)
plt.imshow(spatial_image2,cmap='gray')
plt.title(f'median - PSNR: {med_original :.2f} dB')
plt.tight_layout()
plt.show()



