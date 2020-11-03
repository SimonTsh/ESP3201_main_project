"""scnn_test_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Camera, Display, Keyboard
from vehicle import Driver
import numpy as np
# import cv2

import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

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
focal_length = 0.5*cameraL.getWidth() / np.tan(cameraL.getFov() / 2.0)      # https://stackoverflow.com/questions/61555182/webot-camera-default-parameters-like-pixel-size-and-focus

# Fast SCNN initializations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_fast_scnn('citys', pretrained=True, root='./weights', map_cpu=False).to(device)
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


print("Model setup done!")


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while driver.step() != -1:
    # Read the sensors:
    imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8)
    # imgR = np.asarray(cameraR.getImageArray(),dtype=np.uint8)

    # cameraL.saveImage("left.jpg",100)
    # cameraR.saveImage("right.jpg",100)
    # print(imgR.shape)
    # print("Focal length: ",focal_length)
    # Process sensor data here.

    # run fast SCNN and display processed image in window
    scl = 2048 // imgL.shape[0]
    # image = Image.open('./png/frankfurt_000001_058914_leftImg8bit.png')
    image = Image.fromarray(imgL).resize((1024, 2048))
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    proc_img_trans = get_color_pallete(pred, 'citys')[0:pred.shape[0]:scl, 0:pred.shape[1]:scl]

    img_ref = display.imageNew(proc_img_trans.tolist(), Display.RGB, 512, 256);
    display.imagePaste(img_ref, 0, 0);
    # display.imageSave(img_ref,"disp.jpg")

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
