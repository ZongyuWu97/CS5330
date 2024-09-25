# Ryan Bockmon
# 9/24/2024
import cv2
import numpy as np
import matplotlib.pyplot as plt

Dark_img = np.random.randint(50, size=(500, 500), dtype=np.uint8)
# cv2.imshow("dark", Dark_img)
Light_img = np.random.randint(low=205, high=255, size=(500, 500), dtype=np.uint8)
# cv2.imshow("light", Light_img)
Dark_Light = np.concatenate((Dark_img, Light_img), axis=1)
cv2.imshow("before", Dark_Light)

# equalize = cv2.equalizeHist(Dark_Light)
# cv2.imshow("equalize", equalize)
# merged = Dark_img + Light_img
# cv2.imshow("test", merged)

adaptive_img = Dark_Light.copy()  # creats a copy of the orginal
# 1000 x 500
# top left = 0->200, 0 ->250
subimage_0 = Dark_Light[:, :500]
# cv2.imshow("sub0",subimage_0)
sub0_equalize = cv2.equalizeHist(subimage_0)

subimage_1 = Dark_Light[:, 500:]
# cv2.imshow("sub0",subimage_0)
sub1_equalize = cv2.equalizeHist(subimage_1)

# cv2.imshow("sub0_equal",sub0_equalize)
Dark_Light[:, :500] = sub0_equalize[:, :]
Dark_Light[:, 500:] = sub1_equalize[:, :]
cv2.imshow("after", Dark_Light)

cv2.waitKey(0)
