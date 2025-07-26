import os
import re
import cv2
import fitz
import yaml
import base64
import logging
import numpy as np
from shapely.geometry import Polygon
from typing import Any
from datetime import datetime
from yaml.parser import ParserError
import Levenshtein


_var_matcher = re.compile(r"\${([^}^{]+)}")
_tag_matcher = re.compile(r"[^$]*\${([^}^{]+)}.*")


def _path_constructor(_loader: Any, node: Any):
    def replace_fn(match):
        envparts = f"{match.group(1)}:".split(":")
        return os.environ.get(envparts[0], envparts[1])
    return _var_matcher.sub(replace_fn, node.value)


def load_yaml(filename: str) -> dict:
    yaml.add_implicit_resolver("!envvar", _tag_matcher, None, yaml.SafeLoader)
    yaml.add_constructor("!envvar", _path_constructor, yaml.SafeLoader)
    try:
        with open(filename, "r") as f:
            return yaml.safe_load(f.read())
    except (FileNotFoundError, PermissionError, ParserError):
        return dict()


def pdf_to_images(pdf_content):
    images = []
    mat = fitz.Matrix(2, 2)
    docs = fitz.Document(stream=pdf_content, filetype='pdf')
    for page in docs:
        try:
            pix = page.get_pixmap(matrix=mat)
            shape = (pix.height, pix.width, 3)
            image = np.ndarray(shape, dtype=np.uint8, buffer=pix.samples)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        except:
            continue
    return images


def base64_to_image(string_bytes):
    im_bytes = base64.b64decode(string_bytes)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return image


def total_time(predict):
    def wrapper(self, *args, **kwargs):
        start = datetime.now()
        res = predict(self, *args, **kwargs)
        end = datetime.now()
        # request_id = kwargs.get('request_id', None)
        # print('request_id=' + str(request_id) + ',' + type(self).__name__ + ' predict time: ' + str(end - start))
        return res
    return wrapper


def str_similarity(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    distance = Levenshtein.distance(str1, str2)
    score = 1 - (distance / (max(len(str1), len(str2))))
    return score


# ====================== bbox, poly utils ======================
def iou_poly(poly1, poly2):
    poly1 = np.array(poly1).flatten().tolist()
    poly2 = np.array(poly2).flatten().tolist()

    xmin1, xmax1 = min(poly1[::2]), max(poly1[::2])
    ymin1, ymax1 = min(poly1[1::2]), max(poly1[1::2])
    xmin2, xmax2 = min(poly2[::2]), max(poly2[::2])
    ymin2, ymax2 = min(poly2[1::2]), max(poly2[1::2])

    if xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2:
        return 0, 0, 0

    if len(poly1) == 4:  # if poly1 is a box
        x1, y1, x2, y2 = poly1
        poly1 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    if len(poly2) == 4:  # if poly2 is a box
        x1, y1, x2, y2 = poly2
        poly2 = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if len(poly1) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly1
        poly1 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    if len(poly2) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = poly2
        poly2 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    
    intersect = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    ratio1 = intersect / poly1.area
    ratio2 = intersect / poly2.area
    iou = intersect / union
    
    return ratio1, ratio2, iou


def poly2box(poly):
    poly = np.array(poly).flatten().tolist()
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return [int(xmin), int(ymin), int(xmax), int(ymax)]



def iou_bbox(boxA, boxB):
    if boxA[0] >= boxB[2] or boxA[2] <= boxB[0] or boxA[1] >= boxB[3] or boxA[3] <= boxB[1]:
        return 0, 0, 0
    
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union (IoU)
    r1 = interArea / boxAArea
    r2 = interArea / boxBArea
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return r1, r2, iou


def is_poly_in_box(poly, box):
    r1, r2, iou = iou_poly(poly, box)
    return r1 > 0.7


# ====================== sort utils ======================
def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    if len(polys) == 0:
        return polys
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
    sorted_indices = [bb2idx_original[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_indices


def sort_polys(polys):
    poly2idx_original = {tuple(poly): i for i, poly in enumerate(polys)}
    poly_clusters = row_polys(polys)
    sorted_polys = []
    for row in poly_clusters:
        sorted_polys.extend(row)
    sorted_indices = [poly2idx_original[tuple(poly)] for poly in sorted_polys]
    return sorted_polys, sorted_indices


def sort_texts(bbs, texts):
    bb2text = {tuple(bb): text for bb, text in zip(bbs, texts)}
    sorted_bbs, sorted_indices = sort_bbs(bbs)
    sorted_texts = [bb2text[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_texts


# ======================= object detection utils =======================
def sort_box_by_score(boxes, scores, classes):
    indices = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in indices]
    scores = [scores[i] for i in indices]
    classes = [classes[i] for i in indices]
    return boxes, scores, classes


def convert_docx_to_pdf(docx_path, pdf_path):
    os.system(f'libreoffice --headless --convert-to pdf "{docx_path}" --outdir "{os.path.dirname(pdf_path)}"')



def sort_and_merge_box(bbs, texts, d_thres):
    """
        bbs: List of (xmin, ymin, xmax, ymax) boxes
        texts: List of words corresponding to bbs
        d_thres: height distance threshold between 2 consecutive lines.
    """
    bbs_clusters = [(b, t) for b, t in zip(bbs, texts)]
    bbs_clusters.sort(key=lambda x: x[0][0])

    # group cluster 1st time
    clusters, y_min, cluster_texts = [], [], []
    for tgt_node, text  in bbs_clusters:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            y_min.append(tgt_node[1])
            continue
        matched = None
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            overlap_y = ((src_node[3] - src_node[1]) + (tgt_node[3] - tgt_node[1])) - (max(src_node[3], tgt_node[3]) - min(src_node[1], tgt_node[1]))
            overlap_x = ((src_node[2] - src_node[0]) + (tgt_node[2] - tgt_node[0])) - (max(src_node[2], tgt_node[2]) - min(src_node[0], tgt_node[0]))
            distance = tgt_node[0] - src_node[2]
            if overlap_y > 0.8*min(src_node[3] - src_node[1], tgt_node[3] - tgt_node[1]) and overlap_x < 0.6*min(src_node[2] - src_node[0], tgt_node[2] - tgt_node[0]):
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
            cluster_texts[idx].append(text)
    zip_clusters = list(zip(clusters, y_min, cluster_texts))
    zip_clusters.sort(key=lambda x: x[1])

    # break lines
    page_text_lines = []
    page_bb_lines = []
    for bb_cluster in zip_clusters:
        bbs, _, texts = bb_cluster
        text_lines = []
        bb_lines = []
        text_line = []
        bb_line = []
        for bb, text in zip(bbs, texts):
            if len(text_line) == 0:
                text_line.append(text)
                bb_line.append(bb)
            else:
                if bb[0] - bb_line[-1][2] > d_thres:
                    text_lines.append(text_line)
                    bb_lines.append(bb_line)
                    text_line = [text]
                    bb_line = [bb]
                else:
                    text_line.append(text)
                    bb_line.append(bb)
        if len(text_line) != 0:
            text_lines.append(text_line)
            bb_lines.append(bb_line)
        for text_line, bb_line in zip(text_lines, bb_lines):
            bb_line = np.array(bb_line)
            xmin = np.min(bb_line[:, 0])
            ymin = np.min(bb_line[:, 1])
            xmax = np.max(bb_line[:, 2])
            ymax = np.max(bb_line[:, 3])
            page_text_lines.append(' '.join(text_line))
            page_bb_lines.append([xmin, ymin, xmax, ymax])
    return page_text_lines, page_bb_lines