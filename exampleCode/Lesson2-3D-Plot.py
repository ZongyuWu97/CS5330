#Ryan Bockmon
#9/24/2024
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/dog.jpeg',0)
#cv2.imshow('image',img)
#print(img)
#plt.plot(img)#plots a line of gray scale values for each row
#plt.imshow(img, cmap=plt.cm.gray)
#plt.show()
height, width = img.shape
x = np.linspace(0,width,width, dtype = int)#creats an array from 0 ->width
y = np.linspace(0,height,height, dtype = int)
X, Y = np.meshgrid(x,y)
#print(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
surf = ax.plot_surface(X,Y,img, cmap = plt.cm.gray)
plt.show()
