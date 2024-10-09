# Zongyu Wu
# 10/03/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Part 1
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


# Combine 2 image
def combine(img1, img2):
    newImg = img1.copy()
    row, col = img1.shape
    for r in range(row):
        for c in range(col):
            newImg[r][c] = img1[r][c] / 2 + img2[r][c] / 2
    return newImg.astype(np.uint8)


# def combine1(img1, img2):
#     newImg = img1.copy()
#     row, col = img1.shape
#     for r in range(row):
#         for c in range(col):
#             newImg[r][c] = max(img1[r][c], img2[r][c])
#     return newImg.astype(np.uint8)


# Import images
img1 = cv2.imread("koala.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("puppy.jpg", cv2.IMREAD_GRAYSCALE)
img1 = img1[: img2.shape[0], : img2.shape[1]]


# Create lowpass and highpass then combine 2 images
lp = lowpass(img1)
hp = highpass(img2)
cb = combine(lp, hp)


# Display images
fig, axes = plt.subplots(1, 3)
axes[0].imshow(lp, cmap="gray", vmin=0, vmax=255)
axes[1].imshow(hp, cmap="gray", vmin=0, vmax=255)
axes[2].imshow(cb, cmap="gray", vmin=0, vmax=255)

# test = img1.copy()
# row, col = img1.shape
# for r in range(row):
#     for c in range(col):
#         test[r][c] = 127


# axes[3].imshow(test, cmap="gray", vmin=0, vmax=255)
# axes[4].imshow(combine1(lp, hp), cmap="gray", vmin=0, vmax=255)


# # Adjust font size for axis scales
# for ax in axes.flatten():
#     ax.tick_params(axis="both", which="major", labelsize=8)
#     ax.tick_params(axis="both", which="minor", labelsize=8)

plt.savefig("result")
plt.show()
