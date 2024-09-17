import cv2

img = cv2.imread('images/testimage1.png',
                 0)  # enter the local file path to where you uploaded the image to open it. Mine in in a folder called ‘images’
cv2.imshow('image', img)

cv2.waitKey(0)
