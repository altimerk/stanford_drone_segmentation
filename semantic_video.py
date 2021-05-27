import cv2
import torch
from torchvision import transforms as T
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
model = torch.load('Unet-Mobilenet.pt')
model.eval()
import csv
colors = {}
with open('class_dict_seg.csv') as csvfile:
    code_reader = csv.reader(csvfile, delimiter=',')
    headers = next(code_reader, None)
    for i,row in enumerate(code_reader):
        colors[i] = ((int(row[1]),int(row[2]),int(row[3])))

def maskToCV(mask, colors):
  cv_mask = np.zeros((512, 768, 3), dtype=np.uint8)
  for label, color in colors.items():
    cv_mask[mask == label] = color
  return cv_mask

def inf(img):
  t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
  img = t(img)
  img_patches = img.unfold(1, 512, 284).unfold(2, 768, 654)
  img_patches = img_patches.contiguous().view(3, -1, 512, 768)
  img_patches = img_patches.permute(1, 0, 2, 3)
  with torch.no_grad():
    output = model(img_patches.to('cuda')).cpu()
  return img_patches, output
cap = cv2.VideoCapture('/home/ad/video.mov')
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)
    img_patches, output = inf(frame)
    masked = torch.argmax(output, dim=1)
    # cv2.imshow('mask',masked[0].numpy())

    cv_mask = maskToCV(masked[0], colors)
    cv2.imshow('mask',cv_mask)
    # Press Q on keyboard to  exit
    if cv2.waitKey(2) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()