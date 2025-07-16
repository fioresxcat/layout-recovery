from typing import Any, List, Tuple, Dict
from ultralytics import YOLO
import numpy as np
from modules.yolo.yolo_detector import BaseYOLODetector

class YOLOLayoutDetector:
    def __init__(self, common_cfg, model_cfg):
        self.model = BaseYOLODetector(common_cfg, model_cfg)
        self.model.labels = ['text', 'title', 'table', 'list', 'figure']
    

    def predict(self, images) -> Any:
        results: List = self.model.predict(images)
        return results