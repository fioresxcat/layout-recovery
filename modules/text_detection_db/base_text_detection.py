import os
import pdb
import cv2
import math
import pyclipper
import numpy as np

from shapely.geometry import Polygon

from modules.base import BaseModule
from utils.utils import total_time


class BaseTextDetector(BaseModule):
    def __init__(self, common_config, model_config):
        super(BaseTextDetector, self).__init__(common_config, model_config)
        self.model_config = model_config
        self.box_thresh = self.model_config['box_thresh']
    
    
    def resize_image(self, image, image_short_side):
        h, w = image.shape[:2]
        if h < w:
            h_new = image_short_side
            w_new = int(w / h * h_new / 32) * 32
        else:
            w_new = image_short_side
            h_new = int(h / w * w_new / 32) * 32
        resized_img = cv2.resize(image, (w_new, h_new))
        return resized_img

    
    def scale_polys(self, size, h, w, polys):
        scale_w = size / w
        scale_h = size / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        new_anns = []
        for poly in polys:
            poly = np.array(poly).astype(np.float32)
            poly /= scale
            new_anns.append(poly.astype('int32'))
        return new_anns

    
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        subject = [tuple(l) for l in box]
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        return expanded


    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])


    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7, scale=True, unclip_ratio=1.5):
        height, width = bitmap.shape[:2]
        boxes, scores = [], []

        contours, _ = cv2.findContours((bitmap * 255.0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape((-1, 2)))
            if box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=unclip_ratio)
                if len(box) > 1:
                    continue
                box = np.array(box)
            else:
                continue
            box = box.reshape(-1, 2)
            if len(box) == 0: continue
            box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < 5:
                continue
            box = np.array(box)
            if scale:
                box[:, 0] = np.clip(box[:, 0] / width * dest_width, 0, dest_width)
                box[:, 1] = np.clip(box[:, 1] / height * dest_height, 0, dest_height)
            else:
                box[:, 0] = np.clip(box[:, 0], 0, width)
                box[:, 1] = np.clip(box[:, 1], 0, height)
            boxes.append(box.astype('int32'))
            scores.append(score)
        if max_candidates == -1:
            return boxes, scores
        idxs = np.argsort(scores)
        scores = [scores[i] for i in idxs[:max_candidates]]
        boxes = [boxes[i] for i in idxs[:max_candidates]]

        return boxes, scores
    
    
    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect


    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    
    
    def expand_long_box(self, block, x1, y1, x2, y2, x3, y3, x4, y4):
        h, w, _ = block.shape
        box_h = max(y4 - y1, y3 - y2)
        box_w = max(x2 - x1, x3 - x4)
        if box_w / box_h >= 6:
            expand_pxt = math.ceil(0.1 * box_h)
            x1 = max(0, x1 - expand_pxt)
            x2 = min(w, x2 + expand_pxt)
            x3 = min(w, x3 + expand_pxt)
            x4 = max(0, x4 - expand_pxt)
            y1 = max(0, y1 - expand_pxt)
            y2 = max(0, y2 - expand_pxt)
            y3 = min(h, y3 + expand_pxt)
            y4 = min(h, y4 + expand_pxt)
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        return pts
    
    
    def get_edge(self, x1, y1, x2, y2, x3, y3, x4, y4):
        e1 = math.sqrt(abs(x2 - x1)**2 + abs(y2 - y1)**2)
        e2 = math.sqrt(abs(x3 - x2)**2 + abs(y3 - y2)**2)
        e3 = math.sqrt(abs(x4 - x3)**2 + abs(y4 - y3)**2)
        e4 = math.sqrt(abs(x1 - x4)**2 + abs(y1 - y4)**2)
        edge_s = min([e1, e2, e3, e4])
        edge_l = max([e1, e2, e3, e4])
        return edge_s, edge_l
    
    
    def to_2_points(self, image, x1, y1, x2, y2, x3, y3, x4, y4):
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)
        field_image = image[ymin:ymax, xmin:xmax]
        return field_image
    
    def get_text_images(self, image, bbs):
        text_images = []
        for box in bbs:
            x1, y1 = box[0]
            x2, y2 = box[1]
            x3, y3 = box[2]
            x4, y4 = box[3]
            pts = self.expand_long_box(image, x1, y1, x2, y2, x3, y3, x4, y4)
            edge_s, edge_l = self.get_edge(x1, y1, x2, y2, x3, y3, x4, y4)
            if edge_l / edge_s < 1.5:
                text_image = self.to_2_points(image, x1, y1, x2, y2, x3, y3, x4, y4)
            else:
                text_image = self.four_point_transform(image, pts)
            text_images.append(text_image)
        return text_images
    
    def convert_bbs_format(self, bbs_raw):
        # change to 8 value
        new_bbs = []
        for bb in bbs_raw:
            x1, y1 = bb[0]
            x2, y2 = bb[1]
            x3, y3 = bb[2]
            x4, y4 = bb[3]
            new_bbs.append((x1, y1, x2, y2, x3, y3, x4, y4))
        return new_bbs
    
    def predict(self, image):
        src_image = image.copy()

        h, w = image.shape[:2]
        image = self.resize_image(image, image_short_side=640)
        image = np.expand_dims(image[..., ::-1], 0)
        image = image.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        image_input = np.ascontiguousarray(image, dtype='uint8')  # contiguous

        output_dict = self.request(image_input)
        p = np.array(output_dict.as_numpy(self.model_config['output_name']))[0]
        p = p.transpose((1, 2, 0))
        bitmap = p > 0.2
        bbs, scores = self.polygons_from_bitmap(p, bitmap, w, h, box_thresh=self.box_thresh, max_candidates=-1, unclip_ratio=1.2)

        return bbs, scores