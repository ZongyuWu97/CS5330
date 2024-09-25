import cv2
import matplotlib.pyplot as plt
import numpy as np


# AL3

# img = cv2.imread("images/dog.jpeg", cv2.IMREAD_GRAYSCALE)

# img = 255 - img

# cv2.imshow("image", img)

# cv2.waitKey(0)

# AL4

flowers = cv2.imread("images/flowers.jpg")

equ1 = cv2.equalizeHist(flowers[:, :, 0])
equ2 = cv2.equalizeHist(flowers[:, :, 1])
equ3 = cv2.equalizeHist(flowers[:, :, 2])

cv2.imshow("Equalized Flower", np.dstack((equ1, equ2, equ3)))
cv2.waitKey(0)
