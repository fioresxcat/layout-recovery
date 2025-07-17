from PIL import Image
from surya.layout import LayoutPredictor
from surya.layout.schema import LayoutResult
import pdb
import cv2
from typing import List

from utils.utils import poly2box
from ..label_list import FINAL_LABELS


class SuryaLayoutPredictor:
    def __init__(self, common_cfg, model_cfg):
        self.model = LayoutPredictor()
        self.batch_size = 4
        # self.label_map = {
        #     'Blank': 'blank',
        #     'Text': 'text',
        #     'TextInlineMath': 'equation',
        #     'Code': 'text',
        #     'SectionHeader': 'title',
        #     'Caption': 'caption',
        #     'Footnote': 'footnote',
        #     'Equation': 'equation',
        #     'ListItem': 'list',
        #     'PageFooter': 'footer',
        #     'PageHeader': 'header',
        #     'Picture': 'figure',
        #     'Figure': 'figure',
        #     'Table': 'table',
        #     'Form': 'text',
        #     'TableOfContents': 'table_of_contents',
        #     'Handwriting': 'handwriting'
        # }
        self.label_map = {
            'Blank': 'text',
            'Text': 'text',
            'TextInlineMath': 'text',
            'Code': 'text',
            'SectionHeader': 'title',
            'Caption': 'text',
            'Footnote': 'text',
            'Equation': 'text',
            'ListItem': 'list',
            'PageFooter': 'text',
            'PageHeader': 'text',
            'Picture': 'figure',
            'Figure': 'figure',
            'Table': 'table',
            'Form': 'text',
            'TableOfContents': 'text',
            'Handwriting': 'text'
        }
        assert all(label in FINAL_LABELS for label in self.label_map.values())
        

    def predict(self, images):
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        images = [Image.fromarray(image) for image in images]
        preds: List[LayoutResult] = self.model(images, batch_size=self.batch_size)

        results = [([], [], []) for _ in range(len(images))] # list of boxes, scores, class_names
        for image_index, pred in enumerate(preds):
            pred.bboxes.sort(key=lambda x: x.position)
            for box_info in pred.bboxes:
                class_name = self.label_map[box_info.label]
                score = box_info.confidence
                bb = poly2box(box_info.polygon)
                results[image_index][0].append(bb)
                results[image_index][1].append(score)
                results[image_index][2].append(class_name)
        return results