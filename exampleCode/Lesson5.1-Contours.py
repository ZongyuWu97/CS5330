#ryan bockmon
#10/15/2024
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('../images/blocks1.jpg',0)
#cv2.imshow('image',img1)
#plt.hist(img1.ravel(), 256, [0,256])
#plt.show()

ret, binary = cv2.threshold(img1,230,255,cv2.THRESH_BINARY)
#print(ret)
#cv2.imshow('binary', binary)

kernel = np.ones((7,7),np.uint8)
#img_dilation = cv2.dilate(binary, kernel, iterations = 3)
#cv2.imshow('dilated', img_dilation)

#img_erosion = cv2.erode(binary, kernel, iterations = 4)
#cv2.imshow('eroded', img_erosion)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#print(contours[0])
#print(hierarchy)
cv2.drawContours(img1, contours, -1, (0,255,0), 3)
#-1 = draw all contours
#(0,255,0) -> color them green
#3 = line width
cv2.imshow("contours", img1)

#Bounding boxes
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    cv2.rectangle(img1, (x,y), (x+w,y+h), (0,0,255),3)
cv2.imshow("bouding boxes", img1)






































