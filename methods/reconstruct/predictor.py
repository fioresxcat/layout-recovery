import cv2
from copy import deepcopy
from pathlib import Path
from utils import *
import os
import numpy as np
from docx import Document
from docx import shared
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT
from html.parser import HTMLParser
from shapely.geometry import Polygon
import pdb

from .convert_multi import ConverterMulti
from utils.utils import row_polys, is_poly_in_box, iou_bbox, poly2box


class ReconstructPredictor:
    def __init__(self, common_cfg, model_cfg) -> None:
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
    

    def gather_and_sort_boxes(self, result):
        """Gather text polygons into layout boxes and sort them into rows.

        The resulting layout_texts structure looks like:
        layout_texts = [
            # First layout box (e.g. a text block)
            [
                # First row in this layout box
                [
                    (polygon1, "word1"),  # (polygon coordinates, text)
                    (polygon2, "word2"),
                    ...
                ],
                # Second row
                [
                    (polygon3, "word3"),
                    (polygon4, "word4"),
                    ...
                ],
                ...
            ],
            # Second layout box
            [
                # Rows in second box
                [...],
                ...
            ],
            ...
        ]

        Where each polygon is a list of coordinates [x1,y1, x2,y2, x3,y3, x4,y4]
        """
        result['reconstruct'] = []
        for idx in range(len(result['images'])):
            layout_boxes = result['layout']['boxes'][idx]
            layout_texts = [[] for _ in layout_boxes]
            words = result['ocr']['raw_words'][idx]
            for i, poly in enumerate(result['text_detection']['coords'][idx]):
                for box_idx, box in enumerate(layout_boxes):
                    if is_poly_in_box(poly, box):
                        layout_texts[box_idx].append(poly)
                        break
            poly2word = dict(zip(result['text_detection']['coords'][idx], words))
            for i, polys in enumerate(layout_texts):
                poly_rows = row_polys(polys)
                for row in poly_rows:
                    for poly_idx, poly in enumerate(row):
                        row[poly_idx] = (poly, poly2word[tuple(poly)])
                layout_texts[i] = poly_rows
            
            result['reconstruct'].append({})
            result['reconstruct'][-1]['layout_texts'] = layout_texts
        return result
    

    def log_layout_result(self, request_id, imgs, boxes, class_names, scores):
        # visualize result
        save_dir = f'debug/{request_id}/layout_detection'
        os.makedirs(save_dir, exist_ok=True)
        for idx, (img, boxes, class_names, scores) in enumerate(zip(imgs, boxes, class_names, scores)):
            for i, (box, t, score) in enumerate(zip(boxes, class_names, scores)):
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'{i}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(img, f'{t} {score:.2f}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(save_dir, f'layout_detection_{idx}.jpg'), img)


    def predict(self, result):
        imgs = deepcopy(result['images'])

        self.log_layout_result(result['request_id'], imgs, result['layout']['boxes'], 
                               result['layout']['class_names'], result['layout']['scores'])

        # sort lines and texts into layout boxes
        result = self.gather_and_sort_boxes(result)
        converter = ConverterMulti(result)
        doc = converter.reconstruct()
        result['final_doc'] = doc
        return result