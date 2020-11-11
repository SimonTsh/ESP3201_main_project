"""scnn_test_controller controller."""

from controller import Camera, Display, Keyboard
from vehicle import Driver
import numpy as np

import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

timestep = 10
count = 0

driver = Driver()
driver.setCruisingSpeed(25.0)
kb = Keyboard()
kb.enable(timestep)
display = driver.getDisplay('display')

cameraL = driver.getCamera('cameraL')
cameraL.enable(timestep)
# cameraR = driver.getCamera('cameraR')
# cameraR.enable(timestep)
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



while driver.step() != -1:
    if count % 10 == 0:
        imgL = np.asarray(cameraL.getImageArray(),dtype=np.uint8)
        # imgR = np.asarray(cameraR.getImageArray(),dtype=np.uint8)
        # cameraL.saveImage("left.jpg",100)
        # cameraR.saveImage("right.jpg",100)

        # run fast SCNN and display processed image in window
        image = Image.fromarray(np.transpose(imgL, (1,0,2)))
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        proc_img_trans = get_color_pallete(pred, 'citys')
        img_ref = display.imageNew(proc_img_trans.tolist(), Display.RGB, 1024, 512);
        display.imagePaste(img_ref, 0, 0);

    count += 1
    pass
