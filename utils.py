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

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def semantic_inference(model, img):
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    img = t(img)

    with torch.no_grad():
        output = model(img.unsqueeze(0).cuda()).cpu()
    return output

def make_predictions(net, image, score_threshold=0.12):
    images = image.unsqueeze(0).cuda().float()
#     images = torch.stack(images).cuda().float()
    predictions = []
    with torch.no_grad():
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

def run_wbf(predictions, image_index, image_size=512, iou_thr=0.12, skip_box_thr=0.12, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [prediction[image_index]['labels'].tolist()  for prediction in predictions]
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