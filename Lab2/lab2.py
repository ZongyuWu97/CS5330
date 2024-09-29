# Zongyu Wu, 09/10/2024

import cv2
import matplotlib.pyplot as plot
import numpy as np

# Read color and gray image
flowers = cv2.imread("flowers.jpg")
gray = cv2.imread("flowers.jpg", cv2.IMREAD_GRAYSCALE)

# Get channels
r, g, b = flowers[:, :, 0], flowers[:, :, 1], flowers[:, :, 2]

# Take average
Ave = r / 3 + g / 3 + b / 3
NTSC = 0.299 * b + 0.587 * g + 0.114 * r

# Format average
Ave = Ave
NTSC = NTSC.astype(np.uint8)

# Show gray images
cv2.imshow("Compare", gray)
cv2.imshow("Average", Ave)
cv2.imshow("NTSC", NTSC)
cv2.waitKey(0)
