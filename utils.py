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


def detection_inference(net, frame, score_threshold=0.15):
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    images = t(frame).unsqueeze(0)
    images = images.cuda().float()
    predictions = []
    with torch.no_grad():
        det = net(images, torch.tensor([1] * images.shape[0]).float().cuda())
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:, :4]
            scores = det[i].detach().cpu().numpy()[:, 4]
            labels = det[i].detach().cpu().numpy()[:, 5]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
                'labels': labels[indexes]
            })
    return [predictions], det

def segmentation_inference(model, img):
    img = t(img)

    with torch.no_grad():
        output = model(img.unsqueeze(0).cuda()).cpu()
    return output


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