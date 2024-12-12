import cv2
import matplotlib.pyplot as plt

# Load the image
#image_path = input("Enter the path to the image: ")

image_path = "./img.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load the image. Please check the file path.")
else:
    # Downsample the image
    downsampled = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(image))))

    # Upsample the image
    upsampled = cv2.pyrUp(cv2.pyrUp(image))


    # Save the images
    #cv2.imwrite("original_image.jpg", image)
    #cv2.imwrite("downsampled_image.jpg", downsampled)
    #cv2.imwrite("upsampled_image.jpg", upsampled)
    

    # Display the original, downsampled, and upsampled images
    images = [image, downsampled, upsampled]
    #titles = ["Original Image", "Downsampled Image", "Upsampled Image"]

    titles = [
        f"Original Image\n[ {image.shape[1]} x {image.shape[0]} ]\n",
        f"Downsampled Image\n[ {downsampled.shape[1]} x {downsampled.shape[0]} ]\n",
        f"Upsampled Image\n[ {upsampled.shape[1]} x {upsampled.shape[0]} ]\n",
    ]

    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        #plt.title(titles[i])
        plt.title(titles[i], fontsize=16)
        plt.axis("off")
    plt.show()
