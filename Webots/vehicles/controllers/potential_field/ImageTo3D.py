import cv2
import numpy as np
# import pandas as pd

# labelIds = {'ROAD':7, 'BUILDING':11, 'TRAF_LIGHT':19, 'TRAF_SIGN':20, 'PLANT':21, 'TERRAIN':22, 'SKY':23, 'CAR':26}
labelIds = {'ROAD':0, 'BUILDING':1, 'TRAF_LIGHT':2, 'TRAF_SIGN':3, 'PLANT':4, 'TERRAIN':5, 'SKY':6, 'CAR':7}
MAX_CAST_RANGE = 100.0
PROB_GAIN = 600 / 10      # for h=256
# PROB_GAIN = 600 / 2         # for h=512

def computeDepthMap(y_root=1.77, fov_h=1.0, w=2048, h=1024, normal=[0.0,1.0,0.0]):
    us = np.arange(w) - w // 2 + 0.5
    # vs = np.arange(h // 2 - 1) + 1   # ignore the horizon line (v = 0)
    vs = np.arange(h) - h // 2   # ignore the horizon line (v = 0)
    uu, vv = np.meshgrid(us, vs)
    alpha = 0.5 * w / np.tan(0.5 * fov_h)
    if normal[1] >= 0.9999:         # flat approximation
        ground_mask = vv > 0
        frac = y_root / vv * ground_mask
        x = -uu * frac
        z = alpha * frac
    else:
        ss = normal[0]*uu + normal[2]*alpha
        ground_mask = (vv > -ss)    # all points below horizon, i.e. "above" down-directed v axis
        sInv = y_root / (ss + vv) * ground_mask
        x = -np.sign(uu) * sInv * np.sqrt(uu**2 + ss**2)
        z = sInv * np.sqrt(alpha**2 + ss**2)

    # the ground mask casts invalid points to z = 0, so these points are never sampled
    invProbMap = (np.random.random_sample(uu.shape) * PROB_GAIN < z)

    return (x,z), invProbMap

depthMap_default = computeDepthMap(w=512, h=256)
# depthMap_default = computeDepthMap(w=1024, h=512)

def castTo3D(img, flat_approx=True, pitch=0.0, roll=0.0):
    xs = []
    zs = []
    labels = []

    (height, width) = img.shape
    (x,z), p = depthMap_default if flat_approx else computeDepthMap(w = width, h = height, normal=[-pitch, 1.0 - 0.5*(pitch**2 + roll**2), -roll])

    for label, idx in labelIds.items():
        # mask = ((img[(height//2+1):, :] == idx) & ((p < 1.0)))   # mask and down-sample by p
        mask = (img == idx) & p   # mask and down-sample by p
        numEntries = np.count_nonzero(mask)
        # print("entries:",numEntries)
        if numEntries > 0:
            xs.extend(x[mask].ravel())
            zs.extend(z[mask].ravel())
            labels.extend([label] * numEntries)

    xs = np.array(xs)
    zs = np.array(zs)
    labels = np.array(labels)
    tmp = (zs < MAX_CAST_RANGE)
    xs = xs[tmp]; zs = zs[tmp]; labels = labels[tmp]
    # print("px in cast3d", xs.shape)
    if xs.size == 0:
        print("WARNING: no pixels to re-project")
    # df = np.stack((xs,zs,labels), )
    # df = pd.DataFrame({'x': xs, 'y': ys, 'z': zs, 'label':labels})
    # df = df.drop(df[df.z > MAX_CAST_RANGE].index)
    return xs, zs, labels
