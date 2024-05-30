import json
import numpy as np
from shapely.geometry import Polygon


def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    polys.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in polys:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters


def row_bbs(bbs):
    polys = []
    poly2bb = {}
    for bb in bbs:
        poly = [bb[0], bb[1], bb[2], bb[1], bb[2], bb[3], bb[0], bb[3]]
        polys.append(poly)
        poly2bb[tuple(poly)] = bb
    poly_rows = row_polys(polys)
    bb_rows = []
    for row in poly_rows:
        bb_row = []
        for poly in row:
            bb_row.append(poly2bb[tuple(poly)])
        bb_rows.append(bb_row)
    return bb_rows


def sort_bbs(bbs):
    bb2idx_original = {tuple(bb): i for i, bb in enumerate(bbs)}
    bb_rows = row_bbs(bbs)
    sorted_bbs = [bb for row in bb_rows for bb in row]
    sorted_indices = [bb2idx_original[tuple(bb)] for bb in bbs]
    return sorted_bbs, sorted_indices


def sort_polys(polys):
    poly_clusters = row_polys(polys)
    polys = []
    for row in poly_clusters:
        polys.extend(row)
    return polys, poly_clusters


def sort_json(json_data):
    polys = []
    poly2label = {}
    poly2text = {}
    poly2idx_original = {}
    poly2row = {}
    for i, shape in enumerate(json_data['shapes']):
        if len(shape['points']) != 4:
            raise ValueError('Json contains shape with more than 4 points')
        
        poly = shape['points']
        poly = [int(coord) for pt in poly for coord in pt]
        polys.append(poly)
        poly2label[tuple(poly)] = shape['label']
        poly2text[tuple(poly)] = shape['text']
        poly2idx_original[tuple(poly)] = i
    rows = row_polys(polys)
    for row_idx, row in enumerate(rows):
        for poly in row:
            poly2row[tuple(poly)] = row_idx
    return poly2label, poly2text, rows, poly2idx_original, poly2row



def is_poly_in_box(poly, box):
    xmin_block, ymin_block, xmax_block, ymax_block = box
    xmin_text, xmax_text = min(poly[::2]), max(poly[::2])
    ymin_text, ymax_text = min(poly[1::2]), max(poly[1::2])
    x_center = xmin_text + (xmax_text - xmin_text)/2.0
    y_center = ymin_text + (ymax_text - ymin_text)/2.0
    if x_center > xmin_block and x_center < xmax_block and y_center > ymin_block and y_center < ymax_block:
        return True
    return False


def poly2bb(poly):
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return (xmin, ymin, xmax, ymax)



def compute_boxes_iou(box1, box2):
    x1, y1, x2, y2 = box1
    poly1 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    x1, y1, x2, y2 = box2
    poly2 = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersect / union
    max_overlap_ratio = intersect / min(poly1.area, poly2.area)
    return iou, max_overlap_ratio