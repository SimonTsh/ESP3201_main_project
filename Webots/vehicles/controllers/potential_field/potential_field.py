"""potential_field controller."""

# Webots
from controller import Camera, Display, Keyboard
from vehicle import Driver

# SCNN
import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

# misc and own code
from Potential import getHeadingInput
import numpy as np
import cv2

timestep = 10
count = 0
speed = 60.0
steering_angle = 0.0

driver = Driver()
driver.setCruisingSpeed(speed)
kb = Keyboard()
kb.enable(timestep)
display_nav = driver.getDisplay('display_NAV')
display_seg = driver.getDisplay('display_SEG')

imu = driver.getInertialUnit('imu')
imu.enable(timestep)
cameraL = driver.getCamera('cameraL')
cameraL.enable(timestep)
camera_width = int(cameraL.getWidth())
camera_height = int(cameraL.getHeight())

# Fast-SCNN initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_fast_scnn('citys', pretrained=True, root='./weights', map_cpu=False).to(device)
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
print("Model setup done!")


def set_speed(kmh):
    speed = min(kmh, 250.0);
    print("setting speed to ", speed, " km/h");
    driver.setCruisingSpeed(speed);
    return speed

def checkKeyboard(speed, steering_angle):
    key = kb.getKey()
    angle = 0.0
    if (key == ord('W')):
        speed = set_speed(speed + 0.8)
    elif (key == ord('S')):
        speed = set_speed(speed - 2.0)
    elif (key == ord('D')):
        angle += 0.2
    elif (key == ord('A')):
        angle -= 0.2
    driver.setSteeringAngle(angle)
    steering_angle = angle
    return (speed, steering_angle)

def clamp(value, max_value):
    return max(min(value, max_value), -max_value)

while driver.step() != -1:
    # (speed, steering_angle) = checkKeyboard(speed, steering_angle)

    if count % 10 == 0:
        imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8)

        # run fast SCNN and display processed image in window
        image = Image.fromarray(np.transpose(imgL, (1,0,2)))
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        # img_seg = cv2.resize(get_color_pallete(pred, 'citys'),(0,0),fx=0.5,fy=0.5)      # for h=512
        img_seg = get_color_pallete(pred, 'citys')        # for h=256
        # if np.count_nonzero(pred) < 10000:
        #     continue

        # compute heading input using potential field
        # print("get hdg input")
        TILT_COMPENSATION = False
        if TILT_COMPENSATION:
            # print("pitch: %.3f   roll: %.3f" % (angles[1], angles[0]))
            angles = imu.getRollPitchYaw()
            hdg, u_field = getHeadingInput(pred, angles[1], angles[0]) if not np.isnan(angles[1]) else getHeadingInput(pred)
        else:
            # print("no correction")
            hdg, u_field = getHeadingInput(pred)

        alpha = 0.0
        steering_angle = alpha * steering_angle + (1-alpha) * hdg
        driver.setSteeringAngle(clamp(hdg, 0.1))
        print("Steering angle: ", hdg)

        # draw display feeds
        u_field = (255 * (u_field - np.min(u_field)) / (np.max(u_field) - np.min(u_field))).astype('uint8')
        u_field[u_field == np.bincount(u_field.flatten()).argmax()] = 0
        navImg_L = u_field
        navImg_A = np.ones(u_field.shape,dtype='uint8') * 42
        navImg_B = np.ones(u_field.shape,dtype='uint8') * 211
        img_nav = cv2.rotate(cv2.cvtColor(np.dstack((navImg_L,navImg_A,navImg_B)), cv2.COLOR_LAB2RGB), cv2.ROTATE_180)
        # img_nav = cv2.cvtColor(np.dstack((navImg_L,navImg_A,navImg_B)), cv2.COLOR_LAB2RGB)

        display_nav.imagePaste(display_nav.imageNew(img_nav.tolist(), Display.RGB, img_nav.shape[0], img_nav.shape[1]), 0, 0)
        display_seg.imagePaste(display_seg.imageNew(img_seg.tolist(), Display.RGB, img_seg.shape[0], img_seg.shape[1]), 0, 0)

    count += 1
    pass
