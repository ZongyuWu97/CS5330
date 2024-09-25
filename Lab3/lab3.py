# Zongyu Wu
# 9/24/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Add noise to img by percent of pixels
def addNoise(img, percent):
    # Get the dimensions of the image
    img = img.copy()
    row, col = img.shape

    # Set number of pixels
    numPixels = row * col * percent
    viewed = set()
    i = 0
    while i < numPixels:
        # Pick a random y coordinate
        y = np.random.randint(0, row - 1)

        # Pick a random x coordinate
        x = np.random.randint(0, col - 1)

        if (x, y) in viewed:
            continue

        i += 1
        viewed.add((x, y))

        if np.random.randint(1):
            img[y][x] = 255
        else:
            img[y][x] = 0
    return img


# Kernel used in filter functions
kernel3 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
kernel5 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, 2),
    (0, -2),
    (0, 2),
    (1, -2),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
]


# filter function 3x3
def filter3(img):
    # Get the dimensions of the image
    newImg = img.copy()
    row, col = img.shape

    for r in range(row):
        for c in range(col):
            val = 0
            for x, y in kernel3:
                newR, newC = r + x, c + y
                if 0 <= newR < row and 0 <= newC < col:
                    val += img[newR][newC] / 8
            newImg[r][c] = val

    return newImg.astype(np.uint8)


# filter function 5x5
def filter5(img):
    # Get the dimensions of the image
    newImg = img.copy()
    row, col = img.shape

    for r in range(row):
        for c in range(col):
            val = 0
            for x, y in kernel5:
                newR, newC = r + x, c + y
                if 0 <= newR < row and 0 <= newC < col:
                    val += img[newR][newC] / 24
            newImg[r][c] = val

    return newImg.astype(np.uint8)


# Read original image
img = cv2.imread("dog.jpeg", cv2.IMREAD_GRAYSCALE)

# Add noise
percent = [0.01, 0.1, 0.5]
imgNoise = [addNoise(img, p) for p in percent]

# Filter on each image
imgFilter = []
for i in range(3):
    imgFilter.append([filter3(imgNoise[i]), filter5(imgNoise[i])])

# Display images
fig, axes = plt.subplots(3, 3)
for i in range(3):

    axes[0, i].imshow(imgNoise[i], cmap="gray")
    axes[0, i].set_title(f"{int(percent[i]*100)}% Noise", fontsize=8)

    axes[1, i].imshow(imgFilter[i][0], cmap="gray")
    axes[1, i].set_title(f"3x3 Filter on {int(percent[i]*100)}%", fontsize=8)

    axes[2, i].imshow(imgFilter[i][1], cmap="gray")
    axes[2, i].set_title(f"5x5 Filter on {int(percent[i]*100)}%", fontsize=8)

# Adjust font size for axis scales
for ax in axes.flatten():
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="both", which="minor", labelsize=8)

plt.show()
