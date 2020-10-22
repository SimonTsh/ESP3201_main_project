import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_fast_scnn('citys', pretrained=True, root='./weights', map_cpu=False).to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# image = Image.open(args.input_pic).convert('RGB')

image = PIL.Image.fromarray(cameraL.getImageArray())
image = transform(image).unsqueeze(0).to(device)
# print('Finished loading model!')
model.eval()
with torch.no_grad():
    outputs = model(image)
pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
mask = get_color_pallete(pred, 'citys')

# mask.save(os.path.join(args.outdir, outname))
