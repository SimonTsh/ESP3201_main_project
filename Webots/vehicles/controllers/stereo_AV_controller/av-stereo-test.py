import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('left.jpg',0)
imgR = cv2.imread('right.jpg',0)
disp_webots = cv2.imread('disp.jpg',0)

# stereo = cv2.StereoSGBM_create(numDisparities=256, blockSize=11)
stereo = cv2.StereoSGBM_create(numDisparities=16*4, blockSize=(1+2*3))
# stereo.setPreFilterType(1)
disparity = stereo.compute(imgL,imgR)
disparity2 = disparity
disparity2[disparity2 < 0] = 0
disparity3 = (disparity2 / 4).astype(np.uint8)
disparity3 = cv2.convertScaleAbs(disparity2,alpha=(16 / stereo.getNumDisparities()))

f,a=plt.subplots(2,2)
a[0,0].imshow(disparity,'gray')
a[0,1].imshow(disparity2,'gray');
a[1,0].imshow(disparity3,'gray');
a[1,1].imshow(disp_webots,'gray')
plt.show()
