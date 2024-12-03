# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def deskew(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to create a binary image
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find the contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_contour = max(contours, key=cv2.contourArea)

    # Calculate the minimum area rectangle that encloses the largest contour
    rect = cv2.minAreaRect(max_contour)
    angle = rect[2]

    # Rotate the image to deskew it
    if angle < -45:
        angle += 90

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

# Load the image
image = cv2.imread('images/12.jpg')
# Get grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Both equalization and blurring improve the result
equalized = cv2.equalizeHist(gray) # Equalization
blurred = cv2.GaussianBlur(equalized, (51, 51), 0) # Blur the image
# Thresholding
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Find contours in the threshold image.
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area in decreasing order
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Check if there is at least one contour
if contours:
    # Get the largest contour (first one in the sorted list)
    largest_contour = sorted_contours[0]

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract bounding book area
    book = image[y:y+h, x:x+w]
    book_processed = thresh[y:y+h, x:x+w]


custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode and Page Segmentation Mode
extracted_text = pytesseract.image_to_string(book_processed, config=custom_config)

print(extracted_text)

cv2.imshow("processed image", book)
cv2.waitKey(0)
cv2.destroyAllWindows()
