"""manual_logged_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Camera, Display, Keyboard
from vehicle import Driver
import pandas as pd
import numpy as np
import cv2

# get the time step of the current world.
timestep = 10
# timestep = int(robot.getBasicTimeStep())

# create the Robot instance.
driver = Driver()
kb = Keyboard()
kb.enable(timestep)

cameraL = driver.getCamera('cameraL')
cameraL.enable(timestep)
camera_width = int(cameraL.getWidth())
camera_height = int(cameraL.getHeight())

speed = 25.0
steering_angle = 0.0
manual_steering = 0
idx = 0

driver.setCruisingSpeed(speed)
df = pd.read_csv('drive_data.csv')

def getImageFromCamera(cam):
    img = cam.getImageArray()
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return cv2.flip(img, 1)
    return img

out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (camera_width,camera_height))

while driver.step() != -1:
    # df.loc[idx]
    driver.setCruisingSpeed(df['speed'].loc[idx])
    driver.setSteeringAngle(df['angle'].loc[idx])

    idx += 1

    # Process sensor data here.
    if idx % 3 == 0:
        imgL = getImageFromCamera(cameraL)
        out.write(imgL)

    if (kb.getKey() == ord('Q') or idx > 36300):
        print("Stopping video...")
        out.release()
        break
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

    # Enter here exit cleanup code.
