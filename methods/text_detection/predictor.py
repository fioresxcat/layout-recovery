import pdb
import cv2
import numpy as np
from utils.utils import total_time, iou_poly

from modules.text_detection_db.base_text_detection import BaseTextDetector


class TextDetectPredictor:
    def __init__(self, common_config, model_config):
        self.text_detect = BaseTextDetector(common_config, model_config)
        

    def is_poly_belong(self, text, block):
        r1, r2, iou = iou_poly(text, block)
        return r1 >= 0.5


    def center(self, bb, axis):
        if axis == 'x':
            return (bb[0] + bb[2] + bb [4] + bb [6]) / 4
        elif axis == 'y':
            return (bb[1] + bb [3] + bb [5] + bb[7]) / 4


    def row_bbs(self, bbs):
        bbs.sort(key=lambda x: self.center(x, 'x'))
        clusters, mean_min, mean_max = [], [], []
        for bb in bbs:
            if len(clusters) == 0:
                clusters.append([bb])
                mean_min.append(bb[1])
                mean_max.append(bb[7])
                continue
            mid_y = self.center(bb, 'y')
            mid_x = self.center(bb, 'x')
            matched = None
            for idx, clt in enumerate(clusters):
                last_mid_x = self.center(clt[-1], 'x')
                if mean_min[idx] <= mid_y <= mean_max[idx]:
                    x_dist = mid_x - last_mid_x
                    if matched is None or x_dist < matched[1]:
                        matched = (idx, x_dist)
            if matched is None:
                clusters.append([bb])
                mean_min.append(bb[1])
                mean_max.append(bb[7])
            else:
                idx = matched[0]
                clusters[idx].append(bb)
                mean_min[idx] = (mean_min[idx] + bb[1]) / 2
                mean_max[idx] = (mean_max[idx] + bb[7]) / 2
        zip_clusters = list(zip(clusters, mean_min))
        zip_clusters.sort(key=lambda x: x[1])
        zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    
        return zip_clusters


    def sort_bbs(self, bbs):
        bb_clusters = self.row_bbs(bbs)
        new_clusters = []
        for bb_cluster in bb_clusters:
            bb_cluster.sort(key=lambda b:b[0])
            new_clusters.append(bb_cluster)
        return new_clusters

    @total_time
    def predict(self, result):
        result['text_detection'] = {'coords': [], 'boxes_image': []}
        for image_index, image in enumerate(result['images']):
            bbs, scores = self.text_detect.predict(image)
            bbs_raw = bbs
            new_bbs = self.text_detect.convert_bbs_format(bbs_raw)
            boxes_image = self.text_detect.get_text_images(image, bbs_raw)

            assert len(new_bbs) == len(boxes_image)
            result['text_detection']['coords'].append(new_bbs)
            result['text_detection']['boxes_image'].append(boxes_image)
        
        return result