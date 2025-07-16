from PIL import Image
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult
import pdb
import cv2
from typing import List

from utils.utils import poly2box
from paddleocr import LayoutDetection

LABEL_MAP = {
    'PP-DocLayout-L': {
        'document_title': 'title',
        'paragraph_title': 'title', 
        'text': 'text',
        'page_number': 'footer',
        'abstract': 'text',
        'table_of_contents': 'table_of_contents',
        'references': 'text',
        'footnotes': 'footnote',
        'header': 'header',
        'footer': 'footer',
        'algorithm': 'text',
        'formula': 'equation',
        'formula_number': 'equation',
        'image': 'figure',
        'figure_caption': 'caption',
        'table': 'table',
        'table_title': 'title',
        'seal': 'text',
        'figure_title': 'title',
        'figure': 'figure',
        'header_image': 'figure',
        'footer_image': 'figure',
        'sidebar_text': 'text'
    }
}

class PaddleLayoutPredictor:
    def __init__(self, common_cfg, model_cfg):
        model_name = 'PP-DocLayout-L'
        self.model = LayoutDetection(model_name=model_name)
        self.batch_size = 4
        self.label_map = LABEL_MAP[model_name]


    def predict(self, images):
        preds = self.model.predict(images, batch_size=self.batch_size, layout_nms=True)

        results = [([], [], []) for _ in range(len(images))] # list of boxes, scores, class_names
        for image_index, pred in enumerate(preds):
            boxes_info = pred['boxes']
            for box_info in boxes_info:
                class_name = self.label_map[box_info['label']]
                score = float(box_info['score'])
                bb = list(map(int, box_info['coordinate']))
                results[image_index][0].append(bb)
                results[image_index][1].append(score)
                results[image_index][2].append(class_name)
        return results