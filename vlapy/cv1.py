import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('../vlaimages/lobe/233.4584786_7.030561307.fits.jpg', 0)
img = cv2.imread('../vlaimages/lobe/125.2203169_58.47717684.fits.jpg', 0)

eq_img = cv2.equalizeHist(img)

#plt.hist(eq_img.flat, bins=1000, range=(0,255))

kernel = np.ones((4,4), np.uint8)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img)
erosion = cv2.erode(cl_img, kernel, iterations=1)

ret1, th1 = cv2.threshold(cl_img, 150, 255, cv2.THRESH_BINARY)
ret2, th2 = cv2.threshold(cl_img, 150, 255, cv2.THRESH_BINARY)

plt.hist(cl_img.flat, bins=100, range=(0,100))
#plt.show()

cv2.imshow("Original Image", img)
#cv2.imshow("Equalized Image", cl_img)
#cv2.imshow("Eroded Image", erosion)
#cv2.imshow("Thresh Image1", th1)
cv2.imshow("Thresh Image2", th2)
cv2.waitKey(0)
cv2.destroyAllWindows()