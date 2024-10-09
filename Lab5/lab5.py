# Zongyu Wu
# 10/09/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt


kernel = list(zip(range(-5, 6), range(-5, 6)))


# lowpass filter
def lowpass(img):
    # Get the dimensions of the image
    newImg = img.copy()
    row, col = img.shape

    for r in range(row):
        for c in range(col):
            val = 0
            for x, y in kernel:
                if x == 0 and y == 0:
                    continue
                newR, newC = r + x, c + y
                if 0 <= newR < row and 0 <= newC < col:
                    val += img[newR][newC] / (len(kernel) - 1)
            newImg[r][c] = val

    return newImg.astype(np.uint8)


# highpass filter
def highpass(img):
    lp = lowpass(img)
    newImg = img.copy()
    row, col = img.shape
    for r in range(row):
        for c in range(col):
            if lp[r][c] > img[r][c]:
                newImg[r][c] = 127
            elif img[r][c] - lp[r][c] > 128:
                newImg[r][c] = 255
            else:
                newImg[r][c] = img[r][c] - lp[r][c] + 127
    return newImg


# threshold
def threshold(img, t=200):
    newImg = img.copy()
    row, col = img.shape
    for r in range(row):
        for c in range(col):
            if img[r][c] < t:
                newImg[r][c] = 0
    return newImg


# Import images
img = cv2.imread("fun.jpeg", cv2.IMREAD_GRAYSCALE)


# Create highpass and apply threshold
hp = highpass(img)
th = threshold(hp, 150)

# Display images
fig, axes = plt.subplots(1, 1)
axes.imshow(th, cmap="afmhot", vmin=0, vmax=np.amax(th))
plt.savefig("result")
plt.show()
