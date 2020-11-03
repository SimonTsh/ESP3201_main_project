import cv2
import numpy as np
from enum import IntEnum

vidcap = cv2.VideoCapture('final_seg.mp4')
success,image = vidcap.read()
frame_count = 0
time_increment = 3000
frame_increment = time_increment / 30

isMask = True


class SegClass(IntEnum):
    ROAD = 1
    BUILDING = 2
    FENCE = 3
    TRAFFIC_LIGHT = 4
    TRAFFIC_SIGN = 5
    PLANT = 6
    TERRAIN = 7
    SKY = 8
    CAR = 9

class SegID(IntEnum):
    ROAD = 7
    BUILDING = 11
    FENCE = 13
    TRAFFIC_LIGHT = 19
    TRAFFIC_SIGN = 20
    PLANT = 21
    TERRAIN = 22
    SKY = 23
    CAR = 26

def segImageToMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_v = img_hsv[:,:,0], img_hsv[:,:,2]

    img_out = np.zeros(img_h.shape, dtype=np.uint8)
    img_out[(img_h > 27) & (img_h < 33)] = SegClass.CAR
    img_out[(img_h > 57) & (img_h < 63)] = int(SegClass.PLANT)
    img_out[(img_h > 85) & (img_h < 95)] = SegClass.TRAFFIC_SIGN
    img_out[(img_h > 97) & (img_h < 103)] = SegClass.TRAFFIC_LIGHT
    img_out[(img_h > 117) & (img_h < 123)] = SegClass.ROAD
    img_out[(img_h > 147) & (img_h < 153)] = SegClass.TERRAIN
    img_out[(img_v > 220) & (img_h < 3)] = SegClass.BUILDING
    img_out[(img_v < 5) & (img_h < 3)] = SegClass.SKY
    return img_out


while success and frame_count < 2000:
    if frame_count % frame_increment == 0:
        if isMask:
            cv2.imwrite("frame%d_mask.png" % frame_count, segImageToMask(image))
        else:
            cv2.imwrite("frame%d_raw.png" % frame_count, image)

    success,image = vidcap.read()
    # print('Read a new frame: ', success)
    if frame_count % 10 == 0:
        print("Count: ", frame_count)
    frame_count += 1
