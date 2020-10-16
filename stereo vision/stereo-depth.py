import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('data/Piano-perfect/im0.png',0)
imgR = cv2.imread('data/Piano-perfect/im1.png',0)

# stereo = cv2.StereoSGBM_create(numDisparities=256, blockSize=11)
stereo = cv2.StereoBM_create(numDisparities=16*13, blockSize=(1+2*8))
# stereo.setPreFilterType(1)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
