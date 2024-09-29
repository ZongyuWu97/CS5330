# Zongyu Wu
# 09/28/2024

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Part1
# Manual histogram equalization
def manual_equ(img):
    row, col = img.shape

    # Calculate pixel value count
    count = [0] * 256
    for r in range(row):
        for c in range(col):
            count[img[r][c]] += 1

    # Calculate CDF
    cdf = [count[0]]
    for num in count[1:]:
        cdf.append(cdf[-1] + num)

    # Apply mapping
    newImg = img.copy()
    for r in range(row):
        for c in range(col):
            v = img[r][c]
            newImg[r][c] = (cdf[v] - cdf[0]) / (row * col - cdf[0]) * 255
    return newImg


# Read img
img = cv2.imread("dark_image.jpg", 0)
me = cv2.imread("me.jpeg", 0)

# Get equalized for comparison
equalize = cv2.equalizeHist(img)


# Show images
cv2.imshow("equalizeHist", equalize)
cv2.imshow("Manual equalizeHist", manual_equ(img))
cv2.imshow("Another image I chose", me)
cv2.imshow("Another image I chose after equalization", manual_equ(me))


# Part2
# Thermal filter
def thermal(filename):
    # Get color image and gray image
    color = cv2.imread(filename)
    # gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    gray = (color[:, :, 0] / 3 + color[:, :, 1] / 3 + color[:, :, 2] / 3).astype(
        np.uint8
    )

    row, col, _ = color.shape
    total = row * col
    new_img = color.copy()

    # Calculate brightness count
    count = [0] * 256
    for r in range(row):
        for c in range(col):
            count[gray[r][c]] += 1

    # Calculate CDF
    cdf = [count[0]]
    for num in count[1:]:
        cdf.append(cdf[-1] + num)

    # Calculate average brightness
    avg = 0
    for i, num in enumerate(count):
        avg += num / total * i
    avg = np.uint8(avg)

    # Apply mapping
    for i in range(row):
        for j in range(col):
            v = gray[i][j]
            if v >= avg:
                r = (
                    np.uint8((cdf[v] - cdf[avg]) / (row * col - cdf[avg]) * (255 - 127))
                    + 127
                )
                g = 255 - np.uint8((cdf[v] - cdf[avg]) / (row * col - cdf[avg]) * 255)
                b = 127 - np.uint8((cdf[v] - cdf[avg]) / (row * col - cdf[avg]) * (127))
            else:
                r = np.uint8((cdf[v] - cdf[0]) / (cdf[avg] - cdf[0]) * 127)
                g = np.uint8((cdf[v] - cdf[0]) / (cdf[avg] - cdf[0]) * 255)
                b = 255 - np.uint8(
                    (cdf[v] - cdf[0]) / (cdf[avg] - cdf[0]) * (255 - 127)
                )
            new_img[i][j] = [b, g, r]
    return new_img


thermal_flower = thermal("flowers.jpg")
cv2.imshow("thermal flower", thermal_flower)

cv2.waitKey(0)
