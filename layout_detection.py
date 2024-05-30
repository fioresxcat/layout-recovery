from typing import Any
from ultralytics import YOLO
import numpy as np

class LayoutDetector:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.model = YOLO(self.model_cfg.model_path)
        self.imgsz = 640
        self.labels = self.model.names
    

    def predict(self, result) -> Any:
        images = result['images']
        result['layout'] = {'boxes': [], 'scores': [], 'class_names': []}
        result['tables'] = []

        for image in images:
            out = self.model.predict(source=image, imgsz=self.imgsz, conf=0.3, iou=0.5, save=False, save_txt=False, verbose=False)
            out = out[0]
            if out.boxes is None:
                boxes, scores, classes = [], [], []
            else:
                data_boxes = out.boxes.data.detach().cpu().numpy()
                boxes = data_boxes[:, :4].astype(np.int32).tolist()
                scores = data_boxes[:, 4].tolist()
                classes = data_boxes[:, 5].astype(np.int32).tolist()
                class_names = [self.labels[i] for i in classes]

            result['layout']['boxes'].append(boxes)
            result['layout']['scores'].append(scores)
            result['layout']['class_names'].append(class_names)

            # process table
            tables = {'images': [], 'boxes': []}
            for box, cl in zip(boxes, class_names):
                if cl != 'table':
                    continue
                table_image = image[box[1]:box[3], box[0]:box[2]]
                tables['images'].append(table_image)
                tables['boxes'].append(box)
            result['tables'].append(tables)

        return result  