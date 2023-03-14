import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#Student: Veljko Turudic 18022, github:Archduke-Vexx

#Reading image.
img = cv.imread('coins.png')
cv.imshow("input", img)

#Converting image to HSV with saturation, so when applying threshold only mask will be on copper coin.
imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
imgSaturation = imgHSV[:,:, 1] # 3rd parameter: 0=Hue, 1=Saturation, 2=Value

#Applying threshold to get binary image(0/black and 255/white) with purpose of getting mask of copper coin.
#Automatic value threshold applied(OTSU Method)
_, imgBinary = cv.threshold(imgSaturation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Binary image", imgBinary)

#Manually configuring kernel to use morphologic operation opening to fill the holes in the coin and get rid of noises/taints in background
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
mask = cv.morphologyEx(imgBinary, cv.MORPH_OPEN, kernel)
cv.imshow("image with opening", mask)
cv.imwrite('mask.png', mask)

#For the reason that mask has only 1 channel and BGR(colored) image has 3 channels, in order to use them in same operation
#we have to make mask with 3 channels. Code is from stackoverflow.
mask =  np.expand_dims(mask, 2) 
mask = np.repeat(mask, 3, axis=2) # give the mask the same shape as your image

#To fetch copper coin from colored image, we use mask for that specific coin in same position and apply it on colored image with AND operation.
result = cv.bitwise_and(img, mask)
cv.imshow("result", result)
cv.imwrite('Fetched cooper coin.png', result)

cv.waitKey(0)
cv.destroyAllWindows()