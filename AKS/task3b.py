import numpy as np
import cv2
import matplotlib.pyplot as plt
image= cv2.imread('imgg/cameraman.jpg', 0)
image=cv2.resize(image,(512,512))
height=512
width=512
def PSNR(original,noisy):
   original=original.astype(np.float64)
   noisy=noisy.astype(np.float64)
   mse = np.mean((original - noisy) ** 2)
   max_pixel_value = 255.0
   psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
   return psnr

def addNoise(image):
    noise_image=image.copy()
    noise=0.02
    height,width=512,512

    for h in range(height):
        for w in range(width):
            random_val=np.random.rand()
            if random_val<noise/2:
                noise_image[h,w]=0
            elif random_val<noise:
                noise_image[h,w]=255
    return noise_image
noise_image=addNoise(image)

psnr_val=PSNR(image,noise_image)
plt.subplot(3,3,4)
plt.imshow(image,cmap='gray')

plt.title('original image')
plt.subplot(3,3,5)
plt.imshow(noise_image,cmap='gray')
plt.title(f'psnr : {psnr_val:.2f}db')
def padding_add(n,noise_image):
    mask = np.ones((n,n)) / (n*n*1.0)
    pad_height=n//2
    pad_width=n//2
    new_height=height+2*pad_height # 0 value added left and right
    new_width=width+2*pad_width  # 0 value added top and bottom
    pad_image=np.zeros((new_height,new_width))
    pad_image[pad_height:pad_height + height, pad_width:pad_width + width] = noise_image

    return pad_image,mask

def masking2(noise_image,height,width,n):
    spatial_image=np.zeros((height,width))

    pad_image,mask=padding_add(n,noise_image)

    # mask = np.ones((n,n), dtype=np.float32) / (n*n*1.0)
    # pad_height=n//2
    # pad_width=n//2    
    # pad_image=np.pad(noise_image,((pad_height,pad_height),(pad_width,pad_width)),mode='constant')

    for h in range(height):
        for w in range(width):
            tmp_window=pad_image[h:h+n,w:w+n]
            weight=np.sum(tmp_window*mask)
            spatial_image[h,w]=weight

    return spatial_image


c=0
for i in range(1,4):# i< 4
    n=(i+1)*2-1
    spatial_image=masking2(noise_image,height,width,n)
    med_original=PSNR(image,spatial_image)
    plt.subplot(3,3,i)
    plt.imshow(spatial_image,cmap='gray')
    plt.title(f'{n}X{n} psnr {med_original:.2f} dB')

plt.tight_layout()
plt.show()



