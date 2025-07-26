import onnxruntime as ort
from copy import deepcopy
import numpy as np
import cv2
from PIL import Image
import pdb
import os
from modules.ocr_parseq.base_ocr import BaseOCR
from utils.utils import sort_and_merge_box


class OCRPredictor:
    def __init__(self, common_cfg, model_cfg):
        self.ocr = BaseOCR(common_cfg, model_cfg)
    
    def remove_contained_segments(self, segment_bbs, segment_texts):
        """
        Remove segments that are contained in other segments (>95% of their area is inside another segment).
        If a segment is contained, prepend/append its text to the container's text depending on whether it is nearer the top or bottom.
        """
        from utils.utils import iou_bbox
        keep = [True] * len(segment_bbs)
        updated_texts = list(segment_texts)
        for i, boxA in enumerate(segment_bbs):
            for j, boxB in enumerate(segment_bbs):
                if i == j:
                    continue
                r1, r2, iou = iou_bbox(boxA, boxB)
                if r1 > 0.95:
                    # Determine if A is nearer the top or bottom of B
                    centerA = (boxA[1] + boxA[3]) / 2
                    centerB = (boxB[1] + boxB[3]) / 2
                    if centerA < centerB:
                        # Nearer the top: prepend
                        updated_texts[j] = segment_texts[i] + ' ' + updated_texts[j]
                    else:
                        # Nearer the bottom: append
                        updated_texts[j] = updated_texts[j] + ' ' + segment_texts[i]
                    keep[i] = False
                    break
        filtered_bbs = [bb for bb, k in zip(segment_bbs, keep) if k]
        filtered_texts = [text for text, k in zip(updated_texts, keep) if k]
        return filtered_bbs, filtered_texts


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

        # build segments
        result['ocr']['segments'] = []
        for page_index, (text_bbs, words) in enumerate(zip(result['text_detection']['coords'], result['ocr']['raw_words'])):
            segments = []
            assert len(text_bbs) == len(words)

            p4_bbs = []
            list_hs = []
            for p8_bb in text_bbs:
                xmin, xmax = min(p8_bb[0::2]), max(p8_bb[0::2])
                ymin, ymax = min(p8_bb[1::2]), max(p8_bb[1::2])
                list_hs.append(ymax - ymin)
                p4_bbs.append([xmin, ymin, xmax, ymax])
            avg_h = np.average(list_hs)

            line_text, line_bbs = sort_and_merge_box(p4_bbs, words, d_thres=1.2 * avg_h)
            line_bbs, line_text = self.remove_contained_segments(line_bbs, line_text)
            for line_bb, line_text in zip(line_bbs, line_text):
                segments.append([line_bb, line_text])
            result['ocr']['segments'].append(segments)

        return result
