# Zongyu Wu
# 10/15/2024

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Part 1
img = cv2.imread("dog.jpeg", 0)
img5 = cv2.GaussianBlur(img, (5, 5), 0)
img11 = cv2.GaussianBlur(img, (11, 11), 0)

# Get meshgrid
height, width = img.shape
x = np.linspace(0, width, width, dtype=int)  # creats an array from 0 ->width
y = np.linspace(0, height, height, dtype=int)
X, Y = np.meshgrid(x, y)

# Display
fig = plt.figure()
ax = fig.add_subplot(131, projection="3d")
surf = ax.plot_surface(X, Y, img, cmap=plt.cm.gray)
ax.set_title("Original Image", fontsize=8)

ax5 = fig.add_subplot(132, projection="3d")
surf5 = ax5.plot_surface(X, Y, img5, cmap=plt.cm.gray)
ax5.set_title("5x5 Gaussian Blur", fontsize=8)

ax11 = fig.add_subplot(133, projection="3d")
surf11 = ax11.plot_surface(X, Y, img11, cmap=plt.cm.gray)
ax11.set_title("11x11 Gaussian Blur", fontsize=8)

# What do you notice about the 3D graphs as the filter size increases?
# When the filter size increase, there are less pulses and tips in the 3D plot.


# Part 2

kernel_sobel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
kernel_sobel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# sobel filter
def sobel(img, kernel):
    # Get the dimensions of the image
    img = img.astype(np.float32)
    newImg = img.copy()
    row, col = img.shape

    # Get dimension of kernel
    k_row, k_col = len(kernel), len(kernel[0])
    anchor_row, anchor_col = k_row // 2, k_col // 2

    for r in range(row):
        for c in range(col):
            val = 0
            # Add up convolution for each element in the kernel
            for x in range(k_row):
                for y in range(k_col):
                    new_r, new_c = r + x - anchor_row, c + y - anchor_col
                    if 0 <= new_r < row and 0 <= new_c < col:
                        val += img[new_r][new_c] * kernel[x][y]
            newImg[r][c] = val

    return newImg


# Combine 2 image
def combine(img1, img2, cutoff):
    # Add and take square root
    img12 = cv2.multiply(img1, img1)
    img22 = cv2.multiply(img2, img2)
    magnitude = cv2.sqrt(img12 + img22)

    _, magnitude = cv2.threshold(magnitude, cutoff, 255, 3)
    return magnitude


# Apply sobel filter on original image
img_sobel_X = sobel(img, kernel_sobel_X)
img_sobel_Y = sobel(img, kernel_sobel_Y)
img_sobel_50 = combine(img_sobel_X, img_sobel_Y, 50)
img_sobel_150 = combine(img_sobel_X, img_sobel_Y, 150)
img_canny = cv2.Canny(img, 100, 100)

# Apply gaussian blur then sobel again
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
gaussian_sobel_X = sobel(gaussian, kernel_sobel_X)
gaussian_sobel_Y = sobel(gaussian, kernel_sobel_Y)
gaussian_sobel_50 = combine(gaussian_sobel_X, gaussian_sobel_Y, 50)
gaussian_sobel_150 = combine(gaussian_sobel_X, gaussian_sobel_Y, 150)
gaussian_canny = cv2.Canny(gaussian, 100, 100)

# Display
fig, axes = plt.subplots(2, 4)
axes[0][0].imshow(img, cmap="gray", vmin=0, vmax=255)
axes[0][0].set_title("Original", fontsize=8)
axes[0][1].imshow(img_sobel_50, cmap="gray", vmin=0, vmax=255)
axes[0][1].set_title("Sobel Cut_off 50", fontsize=8)
axes[0][2].imshow(img_sobel_150, cmap="gray", vmin=0, vmax=255)
axes[0][2].set_title("Sobel Cut_off 150", fontsize=8)
axes[0][3].imshow(img_canny, cmap="gray", vmin=0, vmax=255)
axes[0][3].set_title("Canny", fontsize=8)

axes[1][0].imshow(gaussian, cmap="gray", vmin=0, vmax=255)
axes[1][0].set_title("Blurred", fontsize=8)
axes[1][1].imshow(gaussian_sobel_50, cmap="gray", vmin=0, vmax=255)
axes[1][2].imshow(gaussian_sobel_150, cmap="gray", vmin=0, vmax=255)
axes[1][3].imshow(gaussian_canny, cmap="gray", vmin=0, vmax=255)
plt.show()


# – What did you notice when you went from a lower threshold value to a higher one?
# There are less bright pixels in the picture but the edges are clearer since some unrelated details are dropped.

# – What did you notice before and after applying a Gaussian Blur to the image?
# The edges are cleared in all edge detection images including the Canny one. Some details are also dropped.
