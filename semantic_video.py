import cv2
import torch
from torchvision import transforms as T
import numpy as np
from ensemble_boxes import *
from effdet_old import get_efficientdet_config, EfficientDet, DetBenchEval
from effdet_old.efficientdet import HeadNet
import gc
import csv
import argparse
from utils import load_net, semantic_inference, maskToCV, make_predictions, run_wbf
parser = argparse.ArgumentParser()
parser.add_argument('--semantic_model', help='path to semantic model dump')
parser.add_argument('--detect_model', help='path to detection model dump')
parser.add_argument('--colors_csv', help='path colors dictionary mor semantic map')
parser.add_argument('--inp_video', help='path to input video file')
args = parser.parse_args()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
model = torch.load(args.semantic_model)
model.eval()


colors = {}
with open(args.colors_csv) as csvfile:
    code_reader = csv.reader(csvfile, delimiter=',')
    headers = next(code_reader, None)
    for i, row in enumerate(code_reader):
        colors[i] = ((int(row[1]), int(row[2]), int(row[3])))

net = load_net(args.detect_model)
label_colors = {
    1: (255, 22, 96),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255),
    6: (255, 255, 255)
}
t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
cap = cv2.VideoCapture(args.inp_video)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
frame_num = 0

# out = cv2.VideoWriter('semantic.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1024, 512))
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame_num = frame_num + 1
        # Display the resulting frame
        frame_size = max(1024, frame.shape[0])
        frame = frame[:frame_size, :frame_size]
        frame = cv2.resize(frame, (512, 512))
        img = frame
        img_for_detect = frame
        img_for_detect = cv2.cvtColor(img_for_detect, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_for_detect /= 255.0
        tensor = torch.tensor(img_for_detect).permute(2, 0, 1)
        predictions, det = make_predictions(net, tensor)

        output = semantic_inference(model, img)
        masked = torch.argmax(output, dim=1)
        cv_mask = maskToCV(masked[0], colors)
        boxes, scores, labels = run_wbf(predictions,0)
        boxes = boxes.astype(np.int32).clip(min=0, max=511)
        for box, label in zip(boxes, labels):
            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), label_colors[label], 1)
            xc = int((box[0] + box[2]) / 2)
            yc = int((box[1] + box[3]) / 2)
            cv2.circle(cv_mask, center=(xc, yc), radius=5, color=label_colors[label], thickness=7)
        join = np.concatenate([frame, cv_mask], axis=1)
        cv2.imshow('Frame', join)
        # out.write(join)
        # Press Q on keyboard to  exit
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    else:
        break

# When everything done, release the video capture object
cap.release()
# out.release()

# Closes all the frames
cv2.destroyAllWindows()
