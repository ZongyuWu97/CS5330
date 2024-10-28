# Zongyu Wu
# 10/25/2024

import cv2
import numpy as np
import matplotlib.pyplot as plt


vid = cv2.VideoCapture(0)  # should open your webcam
backSub = cv2.createBackgroundSubtractorMOG2()
# _, prev_frame = vid.read()
while True:
    # Get current frame
    _, frame = vid.read()

    # Compute difference from previous frame
    diff = backSub.apply(frame)
    # cv2.imshow("diff", diff)

    # Threshold for binary image
    _, binary = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(
    #     diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    # cv2.imshow("binary", binary)

    # Dilation and Erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    eroded = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Get contour from binary image
    contours, hierarchy = cv2.findContours(
        eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_contour_area = 1000  # Define your minimum area threshold
    max_object = 20
    cnts = sorted(contours, key=cv2.contourArea)
    # Compute bounding box from contour
    for c in cnts[len(cnts) - max_object :]:
        if cv2.contourArea(c) < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destoryAllWindows()
