import cv2
import numpy as np
cv2.filter2D(img, -1, kernel_sharpening)
hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
lower_pimg = cv2.imread('1.jpg')
kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
s==np.array([155,25,0])
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
cv2.imshow('sharpened',s)
cv2.imshow('yellow',resulty)
cv2.imshow('green',resultg)
cv2.imshow('blue',resultb)
cv2.imshow('red',resultr)
cv2.imshow('orange',resulto)
cv2.imshow('pink',resultp)
cv2.waitKey(0)
cv2.destroyAllWindows()