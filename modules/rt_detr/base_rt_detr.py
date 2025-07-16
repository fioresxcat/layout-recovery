import pdb
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing_extensions import List, Dict, Literal, Optional

from modules.base import BaseModule
from ..yolo.yolo_detector import LetterBox
from utils.utils import iou_bbox, poly2box, sort_polys

current_dir = os.path.dirname(os.path.abspath(__file__))


def nms(boxes, scores, classes, threshold=0.3):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    boxes = boxes[keep].tolist()
    scores = scores[keep].tolist()
    classes = classes[keep].tolist()
    return boxes, scores, classes


class BaseRTDETR(BaseModule):
    def __init__(self, common_config, model_config):
        super(BaseRTDETR, self).__init__(common_config, model_config)
        self.threshold = self.model_config['conf_threshold']
        self.resizer = LetterBox((1024, 1024), auto=False, scaleFill=True)
        self.labels = None

    
    def preprocess(self, image):
        image = self.resizer(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, 0).transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        image = np.ascontiguousarray(image, dtype='float32')  # contiguous

        image /= 255  # 0 - 255 to 0.0 - 1.0
        
        return image

    
    def bbox_cxcywh_to_xyxy(self, boxes):
        xmin = boxes[:, 0] - boxes[:, 2]/2
        ymin = boxes[:, 1] - boxes[:, 3]/2
        xmax = boxes[:, 0] + boxes[:, 2]/2
        ymax = boxes[:, 1] + boxes[:, 3]/2
        boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)
        # boxes = np.clip(boxes, 0, 1)
        return boxes


    def predict(self, image):
        assert self.labels is not None, "labels is not set"
        h, w = image.shape[:2]
        processed_img = self.preprocess(image)
        output_dict = self.request(processed_img)
        detections = np.array(output_dict.as_numpy(self.model_config['output_name']))  # shape (bs, 125, 11)
        detections = np.squeeze(detections, axis=0)
        boxes, scores, class_names = [], [], []
        scores = detections[:, 4:]
        boxes = detections[:, :4]
        class_names = np.array([self.labels[i] for i in np.argmax(scores, axis=1)])
        scores = np.max(scores, axis=1).round(3)

        mask = scores > self.threshold
        boxes, scores, class_names = boxes[mask], scores[mask], class_names[mask]
        boxes = self.bbox_cxcywh_to_xyxy(boxes)
        boxes = np.clip(boxes, 0, 1)
        # boxes = boxes #* self.shape
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        # nms for agnostic class
        final_boxes, final_scores, final_class_names = [], [], []
        for cl in self.labels:
            cl_boxes = [b for b, c in zip(boxes, class_names) if c == cl]
            cl_scores = [s for s, c in zip(scores, class_names) if c == cl]
            if len(cl_boxes) == 0: continue
            cl_boxes, cl_scores, cl_class_names = nms(np.array(cl_boxes), np.array(cl_scores), np.array([cl]*len(cl_boxes)))
            final_boxes += cl_boxes
            final_scores += cl_scores
            final_class_names += cl_class_names
        boxes, scores, class_names = final_boxes, final_scores, final_class_names

        return boxes, scores, class_names    