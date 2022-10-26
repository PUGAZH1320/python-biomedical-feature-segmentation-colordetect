import cv2
from cv2 import destroyAllWindows
import numpy as np

 #image_path
img_path="4.jpg"

#read image
img_raw = cv2.imread(img_path)
cv2.waitKey(0)

#select ROI function
roi = cv2.selectROI(img_raw)

#print rectangle points of selected roi
print(roi)

#Crop selected roi from raw image
roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

#show cropped image
cv2.imwrite("ROI2B.png",roi_cropped)
cv2.imshow("ROI",roi_cropped)
cv2.waitKey(0)
img = cv2.imread('ROI2B.png', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = 600
height = 600
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite("ROI_B.png",resized)
cv2.waitKey(0)
destroyAllWindows()