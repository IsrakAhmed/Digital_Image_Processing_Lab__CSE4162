from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image_path ='test.jpg'
rgb_image = Image.open(image_path)


image_array = np.array(rgb_image)
image_shape = image_array.shape
print("Image Shape: ", image_shape)


# red = image_array[:, :, 0]
# green = image_array[:, :, 1]
# blue = image_array[:, :, 2]


grayscale = np.zeros_like(image_array[:, :, 0]) #BGR
# grayscale=[:]

for i in range(image_array.shape[0]):# height
    for j in range(image_array.shape[1]):#width
        red = image_array[i, j, 0]
        green = image_array[i, j, 1]
        blue = image_array[i, j, 2]
       
        gray_value = int(0.29*red + 0.58*green + 0.11*blue)
        grayscale[i,j] = gray_value


plt.subplot(2,2,1)
plt.imshow(rgb_image)
plt.title('RGB')


plt.subplot(2,2,2)
plt.imshow(grayscale, cmap='gray')
plt.title('Grayscale')


plt.show()
