# OpenCV program to perform Edge detection in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
 
# np is an alias pointing to numpy library
import numpy as np
 
 
# capture frames from a camera
cap = cv2.imread('1.jpg', 1)
 
 
# loop runs if capturing has been initialized
while(1):
 
    # reads frames from a camera
    #
 
    # converting BGR to HSV
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
     
    # define range of red color in HSV
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
     
    # create a red HSV colour boundary and
    # threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)
 
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(cap,cap, mask= mask)
 
    # Display an original image
    cv2.imshow('Original',cap)
    # cv2.waitKey(0)
 
    # finds edges in the input image and
    # marks them in the output map edges
    edges = cv2.Canny(cap,100,200)
 
    # Display edges in a frame
    cv2.imshow('Edges',edges)
 
    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
 
 
# Close the window
cap.release()
 
# De-allocate any associated memory usage
cv2.destroyAllWindows()