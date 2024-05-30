import onnxruntime as ort
import pdb
import json
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import math
from copy import deepcopy


def resize_image(image, image_short_side):
    h, w = image.shape[:2]
    if h < w:
        h_new = image_short_side
        w_new = int(w / h * h_new / 32) * 32
    else:
        w_new = image_short_side
        h_new = int(h / w * w_new / 32) * 32
    resized_img = cv2.resize(image, (w_new, h_new))
    return resized_img    

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    subject = [tuple(l) for l in box]
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded


def get_mini_boxes(contour):
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


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    height, width = bitmap.shape[:2]
    boxes, scores = [], []

    contours, _ = cv2.findContours((bitmap * 255.0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon =  0.002* cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape((-1, 2)))
        if box_thresh > score:
            continue
        
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=1.3)
            if len(box) > 1:
                continue
        else:
            continue
        box = np.array(box).reshape(-1, 2)
        if len(box) == 0: continue
        box, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(box[:, 0] / width * dest_width, 0, dest_width)
        box[:, 1] = np.clip(box[:, 1] / height * dest_height, 0, dest_height)
        boxes.append(box.astype('int32'))
        scores.append(score)
    if max_candidates == -1:
        return boxes, scores
    idxs = np.argsort(scores)
    scores = [scores[i] for i in idxs[:max_candidates]]
    boxes = [boxes[i] for i in idxs[:max_candidates]]
        
    return boxes, scores

def expand_boxes(boxes):
    new_boxes = []
    for box in boxes:
        box = np.array(box)
        box_h = max(box[3][1] - box[0][1], box[2][1] - box[1][1])
        box_w = max(box[1][0] - box[0][0], box[2][0] - box[3][0])
        if box_w / box_h < 4.0: 
            new_boxes.append(box)
            continue
        box[:2, 1] -= int(box_h * 10/100)
        box[2:, 1] += int(box_h * 10/100)
        new_boxes.append(box)
    return new_boxes

def prob2heatmap(prob_map, rgb=True):
    heatmap = cv2.applyColorMap((prob_map*255).astype('uint8'), cv2.COLORMAP_JET)
    if not rgb:
        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    return heatmap

def to_json(coords, size, image_path, json_path, labels=None):
    h, w = size
    json_dicts = {'shapes':[], 'imagePath':image_path, 'imageData':None, 'imageHeight':h, 'imageWidth':w}
    if labels is None:
        for coord in coords:
    #         print(coord)
            json_dicts['shapes'].append({'label':'text', 'points':coord.tolist(), 'shape_type':'polygon', 'flags':{}})
    else:
        for coord, label in zip(coords, labels):
            json_dicts['shapes'].append({'label':label, 'points':coord.tolist(), 'shape_type':'polygon', 'flags':{}})

    with open(json_path, 'w') as f:
        json.dump(json_dicts, f)



class TextDetector:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.session = ort.InferenceSession(self.model_cfg['model_path'], providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

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
    

    def is_poly_belong(self, text, block):
        xmin_block, ymin_block, xmax_block, ymax_block = block
        if not isinstance(text, np.ndarray):
            text = np.array(text)
        xmin, xmax = np.min(text[:, 0]), np.max(text[:, 0])
        ymin, ymax = np.min(text[:, 1]), np.max(text[:, 1])
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        if x_center > xmin_block and x_center < xmax_block and y_center > ymin_block and y_center < ymax_block:
            return True
        return False
    

    def center(self, poly, axis):
        if axis == 'x':
            return (poly[0] + poly[2] + poly [4] + poly [6]) / 4
        elif axis == 'y':
            return (poly[1] + poly [3] + poly [5] + poly[7]) / 4


    def row_polys(self, polys):
        polys.sort(key=lambda x: self.center(x, 'x'))
        clusters, mean_min, mean_max = [], [], []
        for bb in polys:
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


    def sort_polys(self, polys):
        poly_clusters = self.row_polys(polys)
        new_clusters = []
        for poly_cluster in poly_clusters:
            poly_cluster.sort(key=lambda b:b[0])
            new_clusters.append(poly_cluster)
        return new_clusters
    

    def preprocess(self, image):
        h, w= image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image, image_short_side=640)
        image_input = np.expand_dims(image, axis=0).transpose(0, 3, 1, 2)#.astype('float32') #1, 3, x, 640
        return image_input


    def predict(self, result):
        result['text_detection'] = {'coords': [], 'boxes_image': []}

        for image_index, image in enumerate(result['images']):
            h, w = image.shape[:2]
            tensor = self.preprocess(image)
            p = self.session.run(None, {self.input_name: tensor})[0][0]
            p = np.array(p).transpose(1, 2, 0) 
            bitmap = p > 0.3
            bbs, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.3, max_candidates=-1)

            # change to 8 value
            new_bbs = []
            bbs_raw = bbs
            for bb in bbs:
                x1, y1 = bb[0]
                x2, y2 = bb[1]
                x3, y3 = bb[2]
                x4, y4 = bb[3]
                new_bbs.append((x1, y1, x2, y2, x3, y3, x4, y4))

            boxes_image = []
            for box in bbs_raw:
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
                boxes_image.append(text_image)

            assert len(new_bbs) == len(boxes_image)
            result['text_detection']['coords'].append(new_bbs)
            result['text_detection']['boxes_image'].append(boxes_image)
            
            # # map texts into tables
            # if len(bbs) > 0:
            #     table_infos = result['tables'][image_index]
            #     table_infos['text_boxes'] = [[] for _ in range(len(table_infos['images']))]
            #     table_infos['text_images'] = [[] for _ in range(len(table_infos['images']))]

            #     clusters = self.sort_polys(np.reshape(bbs, (-1, 8)).tolist())
            #     text_boxes = np.concatenate(clusters, 0).reshape((-1, 4, 2))
            
            #     for box in text_boxes:
            #         box = np.array(box)  # shape (4, 2)
            #         for i, (table_box, table_image) in enumerate(zip(table_infos['boxes'], table_infos['images'])):
            #             x1, y1, _, _ = table_box
            #             if self.is_poly_belong(box, table_box):
            #                 box[..., 0] -= x1
            #                 box[..., 1] -= y1
            #                 table_infos['text_boxes'][i].append(box)
            #                 roi = self.four_point_transform(table_image, box)
            #                 table_infos['text_images'][i].append(roi)
            #                 break

        return result