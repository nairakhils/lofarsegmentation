#0.792075_-1.05485004.fits
#4.422704155_8.465436349.fits

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('vlaimages/lobe/233.4584786_7.030561307.fits.jpg', 0)
#median = cv2.medianBlur(img, 1)
median = cv2.GaussianBlur(img, (1,1), sigmaX=0)

kernel = np.ones((4,4), np.uint8)

#ret, th = cv2.threshold(median, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, th = cv2.threshold(img, 0, 155, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
erosion = cv2.erode(img, kernel, iterations=1)
#new = cv2.medianBlur(erosion, 3)
#new = cv2.GaussianBlur(erosion, (7,7), sigmaX=5)
new1 = cv2.medianBlur(erosion, 3)
new = cv2.GaussianBlur(new1, (5,5), sigmaX=4)
cv2.imshow("Original Image", img)
#cv2.imshow("Opened Image", opening)
cv2.imshow("Eroded Image", erosion)
cv2.imshow("NEW Image", new)
#cv2.imshow("Thresholded Image", th)
cv2.waitKey(0)
cv2.destroyAllWindows()