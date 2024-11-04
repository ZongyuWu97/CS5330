# Zongyu Wu
# 11/03/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Rotate eye
def rotate_image(image, angle, center):
    # Ensure center is a tuple of floats
    center = (float(center[0]), float(center[1]))
    # Get the rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return rotated_image


vid = cv2.VideoCapture(0)  # should open your webcam

# Get pre trained classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    # Get current frame
    _, frame = vid.read()

    # Convert to gray
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face and eye
    face = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    eye = eye_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw bounding box
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    for ex, ey, ew, eh in eye:
        # Extract the eye region
        eye_roi = frame[ey : ey + eh, ex : ex + ew]

        # Rotate the eye region (for example, by 45 degrees)
        eye_center = (ew // 2, eh // 2)
        rotated_eye = rotate_image(eye_roi, 180, eye_center)

        # Replace the original eye region with the rotated one
        frame[ey : ey + eh, ex : ex + ew] = rotated_eye

        # Draw a rectangle around the rotated eye
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 4)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destoryAllWindows()
