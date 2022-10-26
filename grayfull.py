import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('4.jpg')

cv2.imshow('original', image)
cv2.waitKey(0)
 
# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0) 


img = cv2.imread('4.jpg', 2)
#cv2.imshow('Original', img)
#cv2.waitKey(0)
  
ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  
# converting to its binary form
bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
####################################################################################################################################
#sharpening
sh_image = cv2.imread('4.jpg', flags=cv2.IMREAD_COLOR)

#cv2.imshow('AV CV- Winter Wonder', sh_image)
#cv2.waitKey()
#cv2.destroyAllWindows()

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
cv2.imshow('AV CV- Winter Wonder Sharpened', image_sharp)
cv2.waitKey(0)
######################################################################################################################
#edge

#histogram
# img = cv2.imread('4.jpg',0)

# hist,bins = np.histogram(img.flatten(),256,[0,256])

# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()


# cv2.waitKey(0)
# img = cv2.imread('4.jpg',0)
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)
# cv2.imshow('res',res)
# cv2.waitKey(0)

########################################################################################################
#segmentation
img = cv2.imread('4.jpg')
kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
s = cv2.filter2D(img, -1, kernel_sharpening)
hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
lower_p=np.array([155,25,0])
upper_p=np.array([179,255,255])
lower_r=np.array([0,50,50])
upper_r=np.array([10,255,255])
lower_o=np.array([1,190,200])
upper_o=np.array([18,255,255])
lower_y=np.array([20,100,100])
upper_y=np.array([30,255,255])
lower_g=np.array([36,0,0])
upper_g=np.array([86,255,255])
lower_b=np.array([35,140,60])
upper_b=np.array([255,255,180])
maskp = cv2.inRange(hsv,lower_p,upper_p)
masky = cv2.inRange(hsv,lower_y,upper_y)
maskg = cv2.inRange(hsv,lower_g,upper_g)
maskb = cv2.inRange(hsv,lower_b,upper_b)
maskr = cv2.inRange(hsv,lower_r,upper_r)
masko = cv2.inRange(hsv,lower_o,upper_o)
resulty = cv2.bitwise_and(s,s,mask=masky)
resultg = cv2.bitwise_and(s,s,mask=maskg)
resultb = cv2.bitwise_and(s,s,mask=maskb)
resultr = cv2.bitwise_and(s,s,mask=maskr)
resulto = cv2.bitwise_and(s,s,mask=masko)
resultp = cv2.bitwise_and(s,s,mask=maskp)
cv2.imshow('original',img)
# cv2.imshow('sharpened',s)
cv2.imshow('yellow',resulty)
cv2.imshow('green',resultg)
cv2.imshow('blue',resultb)
cv2.imshow('red',resultr)
cv2.imshow('orange',resulto)
# cv2.imshow('pink',resultp)
cv2.waitKey(0)
###################################################################################################################################
# Window shown waits for any key pressing event
 #region benignnnnnnnnn

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
cv2.imwrite("ROIB.png",roi_cropped)
cv2.waitKey(0)
cv2.imshow("ROI",roi_cropped)
cv2.waitKey(0)



################################################################################################
#Resize
img = cv2.imread('ROIB.png', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = 600
height = 600
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite("resized_imgB.png",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################################################################################################
#feature extraction
image = cv2.imread("resized_imgB.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the GLCM
import skimage.feature as feature

# Param:
# source image
# List of pixel pair distance offsets - here 1 in each direction
# List of pixel pair angles in radians
graycom = feature.greycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)

# Find the GLCM properties
contrast = feature.greycoprops(graycom, 'contrast')
dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
homogeneity = feature.greycoprops(graycom, 'homogeneity')
energy = feature.greycoprops(graycom, 'energy')
correlation = feature.greycoprops(graycom, 'correlation')
ASM = feature.greycoprops(graycom, 'ASM')

print("Contrast: {}".format(contrast))
print("Dissimilarity: {}".format(dissimilarity))
print("Homogeneity: {}".format(homogeneity))
print("Energy: {}".format(energy))
print("Correlation: {}".format(correlation))
print("ASM: {}".format(ASM))