import numpy as np
from .yolo.yolo_layout_detection import YOLOLayoutDetector
from .surya.predictor import SuryaLayoutPredictor

class LayoutPredictor:
    def __init__(self, common_cfg, model_cfg, model_type: str):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.model_type = model_type
        
        if self.model_type == 'yolo':
            self.model = YOLOLayoutDetector(common_cfg, model_cfg)
        elif self.model_type == 'surya':
            self.model = SuryaLayoutPredictor(common_cfg, model_cfg)
        else:
            raise ValueError(f'Invalid layout model type: {model_type}')

    def detect_boxes(self, images):
        results = self.model.predict(images)
        return results
    
    def predict(self, result):
        images = result['images']
        result['layout'] = {'boxes': [], 'scores': [], 'class_names': []}
        result['tables'] = []

        det_results = self.detect_boxes(images)

        for image, (boxes, scores, class_names) in zip(images, det_results):
            result['layout']['boxes'].append(boxes)
            result['layout']['scores'].append(scores)
            result['layout']['class_names'].append(class_names)
            # process table
            tables = {'images': [], 'boxes': [], 'index': []}
            for idx, (box, cl) in enumerate(zip(boxes, class_names)):
                if cl != 'table':
                    continue
                table_image = image[box[1]:box[3], box[0]:box[2]]
                tables['images'].append(table_image)
                tables['boxes'].append(box)
                tables['index'].append(idx)
            result['tables'].append(tables)
        return result  
    
    
