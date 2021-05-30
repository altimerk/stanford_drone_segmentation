import cv2
import torch
from torchvision import transforms as T
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from ensemble_boxes import *
from effdet_old import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet_old.efficientdet import HeadNet
import gc
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



def inf(img):
  t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
  img = t(img)
  #     img_patches = img.unfold(1, 512, 284).unfold(2, 768, 654)
  #     img_patches  = img_patches.contiguous().view(3,-1, 512, 768)
  #     img_patches = img_patches.permute(1,0,2,3)

  with torch.no_grad():
    output = model(img.unsqueeze(0).cuda()).cpu()
  #         output = model(img_patches.to('cuda')).cpu()
  return output

def make_predictions(images, score_threshold=0.15):
    images = images.cuda().float()
    predictions = []
    with torch.no_grad():
        print("shape of input: ",images.shape)
        det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]
            scores = det[i].detach().cpu().numpy()[:,4]
            labels = det[i].detach().cpu().numpy()[:,5]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
                'labels': labels[indexes]
            })
    return [predictions],det

def run_wbf(predictions, image_size=512, iou_thr=0.15, skip_box_thr=0.15, weights=None):
    boxes = [(prediction['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction['scores'].tolist()  for prediction in predictions]
    labels = [prediction['labels'].tolist()  for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels


def maskToCV(mask, colors):
  cv_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
  for label, color in colors.items():
    cv_mask[mask == label] = color
  return cv_mask

def load_net(checkpoint_path):
  config = get_efficientdet_config('tf_efficientdet_d1')
  net = EfficientDet(config, pretrained_backbone=False)

  config.num_classes = 6
  config.image_size = 512
  net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

  checkpoint = torch.load(checkpoint_path)
  net.load_state_dict(checkpoint['model_state_dict'])

  del checkpoint
  gc.collect()

  net = DetBenchEval(net, config)
  net.eval()
  return net.cuda()


net = load_net('effdet1_loss_1_42_batch12_state_dict.pt')
label_colors = {
1: (255, 22, 96),
2:(0,255,0),
3: (0,0,255),
 4: (255,255,0),
5:(0,255,255),
6:(255,255,255)
}
t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
cap = cv2.VideoCapture('/mnt/r4/aliev/videos/bookstore/video1/video.mov')
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
frame_num = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret:
    frame_num=frame_num+1
    # Display the resulting frame
    print(frame_num)
    frame_size = max(1024,frame.shape[0])
    frame = frame[:frame_size, :frame_size]
    frame = cv2.resize(frame, (512, 512))
    img = frame
    # img = cv2.resize(img,(512,512))
    image = t(frame).unsqueeze(0)


    predictions,det = make_predictions(image)



    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = inf(img)
    masked = torch.argmax(output, dim=1)
    cv_mask = maskToCV(masked[0], colors)


    boxes, scores, labels = run_wbf(predictions[0])
    boxes = boxes.astype(np.int32).clip(min=0, max=511)
    for box, label in zip(boxes, labels):
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), label_colors[label], 1)
        xc = int((box[0]+box[2])/2)
        yc = int((box[1]+box[3])/2)
        # (box[0]+box[2])/2, (box[1]+box[3])/2
        cv2.circle(cv_mask, center=(xc,yc),radius=5,color=label_colors[label], thickness=7)
    join = np.concatenate([frame, cv_mask],axis=1)
    cv2.imshow('Frame', join)
    # cv2.imshow('mask',cv_mask)
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