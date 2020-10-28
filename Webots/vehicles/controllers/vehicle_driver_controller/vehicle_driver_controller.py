# Copyright 1996-2020 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vehicle_driver controller."""

import math
from controller import Camera, Display
from vehicle import Driver
import numpy as np

def apply_PID(position, targetPosition):
    """Apply the PID controller and return the angle command."""
    P = 0.05
    I = 0.000015
    D = 25
    diff = position - targetPosition
    if apply_PID.previousDiff is None:
        apply_PID.previousDiff = diff
    # anti-windup mechanism
    if diff > 0 and apply_PID.previousDiff < 0:
        apply_PID.integral = 0
    if diff < 0 and apply_PID.previousDiff > 0:
        apply_PID.integral = 0
    apply_PID.integral += diff
    # compute angle
    angle = P * diff + I * apply_PID.integral + D * (diff - apply_PID.previousDiff)
    apply_PID.previousDiff = diff
    return angle

apply_PID.integral = 0
apply_PID.previousDiff = None

# lanePositions = [-10.6, -6.875, -3.2]
# currentLane = 1

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
display = driver.getDisplay('display')
focal_length = 0.5*cameraL.getWidth() / np.tan(cameraL.getFov() / 2.0)      # https://stackoverflow.com/questions/61555182/webot-camera-default-parameters-like-pixel-size-and-focus

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while driver.step() != -1:
    position = gps.getValues()[0]
    angle = 0.1 * math.cos(driver.getTime())
    # angle = max(min(apply_PID(position, lanePositions[currentLane]), 0.5), -0.5)
    driver.setSteeringAngle(angle)

    # # Read the sensors:
    # imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8)
    # imgR = np.asarray(cameraR.getImageArray(),dtype=np.uint8)

    # cameraL.saveImage("left.jpg",100)
    # cameraR.saveImage("right.jpg",100)
    # print(imgR.shape)
    # print("Focal length: ",focal_length)

    pass