import cv2

# Zongyu Wu, 09/10/2024

# Load the images
image1 = cv2.imread("flowers.jpg")
image2 = cv2.imread("flowers.jpg", cv2.IMREAD_GRAYSCALE)

# Display the images
cv2.imshow("Original", image1)
cv2.imshow("Gray", image2)

# Press any key to close both images
cv2.waitKey(0)
