# Zongyu Wu
# 10/19/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img_gray = cv2.imread("blocks.jpg", 0)
img_color = cv2.imread("blocks.jpg")
# cv2.imshow("gray", img_gray)

# Get yellow object using HSV
hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([23, 150, 100])
upper_yellow = np.array([30, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# cv2.imshow("mask", mask)

# Get other object with threshold
ret, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("binary", binary)

# Combine two image
combine = mask.copy()
row, col = mask.shape
for r in range(row):
    for c in range(col):
        combine[r][c] = max(mask[r][c], binary[r][c])
# cv2.imshow("combine", combine)

# Dilation and Erode
kernel = np.ones((7, 7), np.uint8)
img_dilation = cv2.dilate(combine, kernel, iterations=1)
# cv2.imshow("dilated", img_dilation)
img_erosion = cv2.erode(img_dilation, kernel, iterations=2)
# cv2.imshow("eroded", img_erosion)


# Display contour
contours, hierarchy = cv2.findContours(
    img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
# -1 = draw all contours
# (0,255,0) -> color them green
# 3 = line width
cv2.drawContours(img_color, contours, -1, (0, 255, 0), 3)


# Find center
for i in contours:
    M = cv2.moments(i)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(img_color, (cx, cy), 7, (0, 0, 255), -1)
        cv2.putText(
            img_color,
            "center",
            (cx - 20, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
    print(f"x: {cx} y: {cy}")

cv2.imshow("contours", img_color)


cv2.waitKey(0)
