import cv2
 
img = cv2.imread('ROI2M.png', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = 600
height = 600
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite("resized_imgMttr.png",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()