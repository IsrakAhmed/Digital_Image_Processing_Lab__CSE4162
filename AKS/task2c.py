import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
image= cv2.imread('imgg/images.jpg', 0)
final_img=cv2.resize(image,(512,512))

height,width=final_img.shape

msb_mask =224 
gray_image_msb =final_img & msb_mask


diff=abs(final_img-gray_image_msb)
cv2.imshow('difference image', diff)
cv2.imshow('original image',final_img)
cv2.imshow('msb image',gray_image_msb)
cv2.waitKey()


#   this code is checked and result alright