import numpy as np
import cv2
from matplotlib import pyplot as plt

# Initialize STAR detector
orb = cv2.ORB_create(nfeatures=1000,scoreType=cv2.ORB_FAST_SCORE)

# img1 = cv2.imread('data/templeSparseRing/templeSR0001.png',0)
# img2 = cv2.imread('data/templeSparseRing/templeSR0002.png',0)
img1 = cv2.imread('data/Bicycle1-imperfect/im0.png',0)
img2 = cv2.imread('data/Bicycle1-imperfect/im1.png',0)

# find the keypoints with ORB
# keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# # draw only keypoints location,not size and orientation
# img_kp = cv2.drawKeypoints(img,kp,outImage = None, color=(0,255,0), \
# flags=cv2.DrawMatchesFlags_DEFAULT)
#
# plt.imshow(img_kp),plt.show()

# bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)
#
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], outImg = None,flags=2)
# plt.imshow(img3), plt.show()


# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params= dict(algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3), plt.show()
