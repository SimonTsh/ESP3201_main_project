"""stereo_AV_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Camera, Display, Keyboard
from vehicle import Driver
import numpy as np
import cv2

# get the time step of the current world.
timestep = 50
# timestep = int(robot.getBasicTimeStep())

# create the Robot instance.
driver = Driver()
driver.setCruisingSpeed(25.0)
kb = Keyboard()
kb.enable(timestep)


# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
cameraL = driver.getCamera('cameraL')
cameraR = driver.getCamera('cameraR')
cameraL.enable(timestep)
cameraR.enable(timestep)
camera_width = int(cameraL.getWidth())
camera_height = int(cameraL.getHeight())
display = driver.getDisplay('display')
focal_length = cameraL.getWidth() / 2.0 / np.tan(cameraL.getFov() / 2.0)      # https://stackoverflow.com/questions/61555182/webot-camera-default-parameters-like-pixel-size-and-focus

orb = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_FAST_SCORE)
min_disparity = 1
num_disparity16 = 4
blocksize_2 = 4
stereo = cv2.StereoSGBM_create(numDisparities=16*num_disparity16, blockSize=(1+2*blocksize_2))
# SGBM seems to work well with 80-96 disparities & block size of 7-11
# stereo.setMinDisparities(min_disparity)

def kbStereoUpdateListener(matcher):
    key = kb.getKey()
    if (key != -1):
        num_disparity16 = int(matcher.getNumDisparities() / 16)
        blocksize_2 = int(matcher.getBlockSize() / 2)
        if (key == ord('W')):
            num_disparity16 += 1
        if (key == ord('S')):
            num_disparity16 = max(1, num_disparity16-1)
        if (key == ord('E')):
            blocksize_2 = min(255, blocksize_2+1)
        if (key == ord('D')):
            blocksize_2 = max(2, blocksize_2-1)
        # stereo.setMinDisparities(min_disparity)
        print("DISP_MIN: ",min_disparity)
        print("DISP_NUM: ",16*num_disparity16)
        print("BLOCK_SIZE: ", 1+2*blocksize_2)
        return cv2.StereoSGBM_create(numDisparities=16*num_disparity16, blockSize=(1+2*blocksize_2))
    else:
        return matcher

def projectPoints(image, disparity, f, b):
    height = image.shape[0]
    width = image.shape[1]
    zs = b * f * np.reciprocal(np.ma.masked_less_equal(disparity, 0).astype(np.float32))
    # Zu = f*X + u0*Z
    # X = Z*(u - u0) / f
    xs = np.multiply(zs, (np.linspace(0.0,width-1,width) - 0.5*width) / f)
    ys = np.multiply(zs, (np.linspace(0.0,height-1,height) - 0.5*height) / f)

# OpenCV functions
def getImageFromCamera(cam):
    img = cam.getImageArray()
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.flip(img, 1)
    return img


def kpMatching(img1_color, img2_color):
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

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

    img_out = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # print(img_out.shape)
    return img_out


# subpixel disparities? not allowed?
def computeDisparityMap(imgL_color, imgR_color):
    imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(imgL,imgR) + stereo.getMinDisparity()
    # print(disparity.dtype)
    # print("Disparity (raw): ",np.min(disparity), np.median(disparity), np.max(disparity))

    # temp? set unmatched/invalid pixels (i.e. negative/-16) to zero. instead of being treated as invalid, they're cast out to max distance...
    disparity[disparity<0] = 0

    # for display, returns uint8 version (original is int16, but bits > 255 don't seem to be used)
    return cv2.cvtColor(cv2.convertScaleAbs(disparity,alpha=(16 / stereo.getNumDisparities())).astype(np.uint8) , cv2.COLOR_GRAY2RGB)


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while driver.step() != -1:
    # listen for kb input
    stereo = kbStereoUpdateListener(stereo)

    # Read the sensors:
    # imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8)
    # imgR = np.asarray(cameraR.getImageArray(),dtype=np.uint8)
    imgL = getImageFromCamera(cameraL)
    imgR = getImageFromCamera(cameraR)

    # cameraL.saveImage("left.jpg",100)
    # cameraR.saveImage("right.jpg",100)
    # print(imgR.shape)
    print("focal length : ",cameraL.getFocalDistance())
    # Process sensor data here.

    # display processed image in window
    # proc_img = kpMatching(imgL, imgR)
    proc_img = computeDisparityMap(imgL, imgR)

    # proc_img_trans = proc_img

    proc_img_trans = cv2.rotate(cv2.flip(proc_img,1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    print("Disparity (proc): ",np.min(proc_img), np.median(proc_img), np.max(proc_img))

    # img_ref = display.imageNew(proc_img.tolist(), Display.RGB, 480, 240);
    img_ref = display.imageNew(proc_img_trans.tolist(), Display.RGB, 480, 240);
    display.imagePaste(img_ref, 0, 0);
    # display.imageSave(img_ref,"disp.jpg")

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
