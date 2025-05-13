import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('imgg/Characters Test Pattern 688x688.tif', 0)
image = cv2.resize(img, (512, 512))

height, width = image.shape

mean = 10
stddev = 25

noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)

noisy_image = cv2.add(image, noise)

# Compute the 2D discrete Fourier transform (DFT) of the noisy image
F = np.fft.fftshift(np.fft.fft2(noisy_image))

# Define the Butterworth filter

D0 = 25  # Cutoff frequency
n = 2  # Filter order
def gaussian(F):
    M, N = F.shape
    Gaussian = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            
            D =np.sqrt( (u - M/2)**2 + (v - N/2)**2)
            
            Gaussian[u, v] = np.exp(-((D**2) / (2 * D0**2)))

    Gaussian_constant=Gaussian*F
    filter_image=np.abs(np.fft.ifft2(Gaussian_constant))
    # filtered_image = cv2.normalize(filter_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # filtered_image=filter_image/255
    return filter_image


def butterworth(F):
    M, N = F.shape 
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            H[u, v] = 1 / (1 + (D / D0)**(2 * n))
    filterd_image= F * H
    filtered_image = np.abs(np.fft.ifft2(filterd_image))
    # filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    return filtered_image


butter_filter=butterworth(F)
gaussian_filter=gaussian(F)
# Display the original noisy image and the filtered image
plt.subplot(2, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(2, 2, 3)
plt.imshow(butter_filter, cmap='gray')
plt.title('butter Filtered Image')

plt.subplot(2,2,4)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('gaussian Filtered Image')
plt.tight_layout()


plt.show()
