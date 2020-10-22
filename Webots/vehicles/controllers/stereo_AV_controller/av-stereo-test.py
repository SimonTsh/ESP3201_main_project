import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('left.jpg',0)
imgR = cv2.imread('right.jpg',0)
disp_webots = cv2.imread('disp.jpg',0)

def projectPoints(image, disparity, f, b):
    """ Simplified point projection algorithm for parallel stereo setup. """
    height = image.shape[0]
    width = image.shape[1]
    zs = b * f * np.reciprocal(np.ma.masked_less_equal(disparity, 0).astype(np.float32))
    # Zu = f*X + u0*Z
    # X = Z*(u - u0) / f
    xs = np.multiply(zs, (np.linspace(0.0,width-1,width) - 0.5*width) / f)
    ys = np.multiply(zs, (np.linspace(0.0,height-1,height) - 0.5*height) / f)

    xx, yy = np.meshgrid(xs, ys)


# stereo = cv2.StereoSGBM_create(numDisparities=256, blockSize=11)
stereo = cv2.StereoSGBM_create(numDisparities=16*4, blockSize=(1+2*3))
# stereo.setPreFilterType(1)
disparity = stereo.compute(imgL,imgR)
disparity2 = disparity
disparity2[disparity2 < 0] = 0
disparity3 = (disparity2 / 4).astype(np.uint8)
disparity3 = cv2.convertScaleAbs(disparity2,alpha=(16 / stereo.getNumDisparities()))

b = 0.8
f = 439.3170532109885
projectPoints(imgL, disparity, f, b)

# f,a=plt.subplots(2,2)
# a[0,0].imshow(disparity,'gray')
# a[0,1].imshow(disparity2,'gray');
# a[1,0].imshow(disparity3,'gray');
# a[1,1].imshow(disp_webots,'gray')
# plt.show()
