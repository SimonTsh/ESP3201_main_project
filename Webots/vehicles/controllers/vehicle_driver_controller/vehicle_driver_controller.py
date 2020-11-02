"""vehicle_driver controller."""
# https://cyberbotics.com/doc/reference/camera

import math
from controller import Camera, Display
from vehicle import Driver
import numpy as np

# def apply_PID(position, targetPosition):
#     """Apply the PID controller and return the angle command."""
#     KP = 0.25
#     KI = 0.006
#     KD = 2
#     diff = position - targetPosition
#     if apply_PID.previousDiff is None:
#         apply_PID.previousDiff = diff
#     # anti-windup mechanism
#     if diff > 0 and apply_PID.previousDiff < 0:
#         apply_PID.integral = 0
#     if diff < 0 and apply_PID.previousDiff > 0:
#         apply_PID.integral = 0
#     apply_PID.integral += diff
#     # compute angle
#     angle = KP * diff + KI * apply_PID.integral + KD * (diff - apply_PID.previousDiff)
#     apply_PID.previousDiff = diff
#     return angle

# apply_PID.integral = 0
# apply_PID.previousDiff = None

# lanePositions = [-10.6, -6.875, -3.2]
# currentLane = 1


KP = 0.25
KI = 0.006
KD = 2
FILTER_SIZE = 3 # size of yellow line angle filter

# get the time step of the current world.
timestep = 50

# create the robot instance
driver = Driver()
driver.setSteeringAngle(0.1)
driver.setCruisingSpeed(25.0)

gps = driver.getGPS("gps")
gps.enable(10)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
cameraL = driver.getCamera('cameraL')
cameraR = driver.getCamera('cameraR')
cameraL.enable(timestep)
cameraR.enable(timestep)
camera_width = int(cameraL.getWidth())
camera_height = int(cameraL.getHeight())
camera_fov = cameraL.getFov()
display = driver.getDisplay('display')
focal_length = 0.5*cameraL.getWidth() / np.tan(camera_fov / 2.0)      # https://stackoverflow.com/questions/61555182/webot-camera-default-parameters-like-pixel-size-and-focus

def apply_PID(yellow_line_angle):
    old_value = 0.0
    integral = 0.0

    if (PID_need_reset):
        old_value = yellow_line_angle
        integral = 0.0
        PID_need_reset = false

    # anti-windup mechanism
    if (yellow_line_angle != old_value):
        integral = 0.0

    diff = yellow_line_angle - old_value

    # limit integral
    if (integral < 30 and integral > -30):
        integral += yellow_line_angle
    
    old_value = yellow_line_angle
    return KP * yellow_line_angle + KI * integral + KD * diff

def set_speed(kmh):
    if (kmh > 250.0):
        kmh = 250.0
    speed = kmh
    print('setting speed to %g km/h\n', speed)

def color_diff(a, b):
    i, diff = 0, 0
    while (i < 3):
        d = a[i] - b[i]
        print(d[i]) # not sure if comparing index is right
        diff += d[i] if d[i] > 0 else -d[i]
        i += 1
    return diff

def process_camera_image(image):
    num_pixels = camera_height * camera_width
    REF = [95, 187, 203] # yellow road in BGR format
    sumx = 0
    pixel_count = 0
    i = 0
    
    pixel = image[i]
    for x in range(num_pixels):
        # print(pixel) # taking index as pointer
        if (color_diff(pixel, REF) < 30):
            sumx += x % camera_width
            pixel_count += 1
        i += 4
        
    if (pixel_count == 0):
        return None
    
    return (sumx/pixel_count/camera_width - 0.5) * camera_fov
    
def filter_angle(new_value):
    first_call = True
    old_value = [0]*FILTER_SIZE
    
    if (first_call or new_value == None): # reset all the old values to 0.0
        first_call = False
        for i in FILTER_SIZE:
            old_value[i] = 0.0
    else:
        for i in (FILTER_SIZE-1):
            old_value[i] = old_value[i+1]
    
    if (new_value == None):
        return None
    else:
        old_value[FILTER_SIZE-1] = new_value
        sumx = 0.0
        for i in FILTER_SIZE:
            sumx += old_value[1]
        return sumx/FILTER_SIZE

def set_steering_angle(wheel_angle):
    if (wheel_angle - steering_angle > 0.1):
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle < 0.1):
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle

    if (wheel_angle > 0.5):
        wheel_angle = 0.5
    elif (wheel_angle < -0.5):
        wheel_angle = -0.5
    setSteeringAngle(wheel_angle) # what is wbu_driver_set_steering_angle in c?

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while driver.step() != -1:
    imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8) # take everything from the left camera first
    yellow_line_angle = filter_angle(process_camera_image(imgL))
    if (yellow_line_angle != None):
        line_following_steering = apply_PID(yellow_line_angle)
    set_steering_angle(line_following_steering)

    # position = gps.getValues()[0]
    # angle = 0.1 * math.cos(driver.getTime())
    # angle = max(min(apply_PID(position, lanePositions[currentLane]), 0.5), -0.5)
    # driver.setSteeringAngle(angle)

    # # Read the sensors:
    # imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8)
    # imgR = np.asarray(cameraR.getImageArray(),dtype=np.uint8)

    # cameraL.saveImage("left.jpg",100)
    # cameraR.saveImage("right.jpg",100)
    # print(imgR.shape)
    # print("Focal length: ",focal_length)

    pass