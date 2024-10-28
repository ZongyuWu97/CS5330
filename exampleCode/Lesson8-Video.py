# Ryan Bockmon
# 10/22/2024
import cv2

vid = cv2.VideoCapture(0)  # should open your webcam
# cv2.imshow("frame",vid)
while True:
    _, frame = vid.read()
    cv2.imshow("video", frame)

    frame_canny = cv2.Canny(frame, 100, 100)
    cv2.imshow("video_edge", frame_canny)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destoryAllWindows()
