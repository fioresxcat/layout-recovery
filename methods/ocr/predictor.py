import onnxruntime as ort
from copy import deepcopy
import numpy as np
import cv2
from PIL import Image
import pdb
import os
from modules.ocr_parseq.base_ocr import BaseOCR

class OCRPredictor:
    def __init__(self, common_cfg, model_cfg):
        self.ocr = BaseOCR(common_cfg, model_cfg)

    def predict(self, result):
        batch_images = []
        text_output = []
        page_lengths = []
        result['ocr'] = {'raw_words': []}

        for page_index, boxes_image in enumerate(result['text_detection']['boxes_image']):
            page_lengths.append(len(boxes_image))
            batch_images.extend(boxes_image)

        words = self.ocr.predict(batch_images)
        
        index = 0
        for page_len in page_lengths:
            result['ocr']['raw_words'].append(words[index:index+page_len])
            index += page_len

        # # map text to table
        # for page_index, table_infos in enumerate(result['tables']):
        #     table_infos['text'] = []
        #     for table_index, list_rois in enumerate(table_infos['text_images']):
        #         table_infos['text'].append([])
        #         for box, roi, text in zip(table_infos['text_boxes'][table_index], list_rois, table_infos['raw_words'][table_index]):
        #             table_infos['text'][table_index].append({'box':box, 'roi':roi, 'text':text})

        return result
