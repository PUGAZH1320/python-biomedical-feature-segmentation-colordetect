
import cv2
  
# read the image file
img = cv2.imread('1.jpg', 1)
cv2.imshow('Original', img)
cv2.waitKey(0)

# change into grayscale
img = cv2.imread('1.jpg', 2)
cv2.imshow('Original', img)
cv2.waitKey(0)
  
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  
# converting to its binary form
bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()