import os
from pathlib import Path
import pdb
import numpy as np
import cv2
from collections import defaultdict
import pymupdf

from utils.utils import iou_bbox, row_bbs
from .yolo.yolo_layout_detection import YOLOLayoutDetector
from .surya.predictor import SuryaLayoutPredictor
from .paddle.predictor import PaddleLayoutPredictor


class PDFBlock:
    """
    Generate pymupdf-based pdf block for block that is incorrectly detected by layout model
    """

    def __init__(self):
        self.font_dir = Path(__file__).parent.parent / 'reconstruct_pdf' / 'fonts'


    def get_best_font_size(self, page, arch, segment_bb, segment_text, css_template, initial_font_size):
        """
            Create a temp blank page, insert the html text, get the bounding boxes around the text,
            compare its width with the segment_bb and finally adjust the font size
        """
        # Prepare a temp blank page (same size as current page)
        temp_doc = pymupdf.open()  # create a new empty PDF
        temp_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)

        # Try to fit the text in the segment_bb by adjusting font size
        max_iter = 20
        min_font_size = 6
        max_font_size = 48
        font_step = 0.5  # smaller step for finer adjustment
        target_width = int(segment_bb[2] - segment_bb[0]) * 1
        best_font_size = initial_font_size
        best_width_diff = float('inf')

        for _ in range(max_iter):
            css = css_template.format(font_size=initial_font_size)
            temp_page.clean_contents()  # clear previous content
            temp_page.insert_htmlbox(segment_bb, segment_text, css=css, archive=arch)
            # Get the bounding box of the inserted text
            words = temp_page.get_text("words")
            if not words:
                break
            x0s = [w[0] for w in words]
            x1s = [w[2] for w in words]
            text_left = min(x0s)
            text_right = max(x1s)
            text_width = text_right - text_left

            width_diff = abs(text_width - target_width)
            if width_diff < best_width_diff:
                best_width_diff = width_diff
                best_font_size = initial_font_size
                if best_width_diff / target_width < 0.05:
                    break

            # Adjust font size with smaller step
            if text_width > target_width and initial_font_size > min_font_size:
                initial_font_size -= font_step
            elif text_width < target_width and initial_font_size < max_font_size:
                initial_font_size += font_step
            else:
                break
        
        return best_font_size


    def postprocess_blocks(self, page_blocks):
        """
        Postprocess the blocks to remove duplicated or overlapped blocks.
        - If blockA is >70% contained by blockB, merge A into B (expand B's bbox, remove A)
        - If two blocks overlap (IoU > 0.1), cut the upper block (smaller y0) so it no longer overlaps with the lower block, on both axes
        - After any modification, restart the process
        """
        blocks = [dict(block) for block in page_blocks]  # shallow copy
        changed = True
        min_overlap = 0.1
        contain_thresh = 0.7
        
        while changed:
            changed = False
            n = len(blocks)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    boxA = blocks[i]['bbox']
                    boxB = blocks[j]['bbox']
                    r1, r2, iou = iou_bbox(boxA, boxB)
                    # Containment: if A is >70% contained by B, merge A into B
                    if r1 > contain_thresh:
                        # Merge A into B (expand B's bbox)
                        x0 = min(boxA[0], boxB[0])
                        y0 = min(boxA[1], boxB[1])
                        x1 = max(boxA[2], boxB[2])
                        y1 = max(boxA[3], boxB[3])
                        blocks[j]['bbox'] = [x0, y0, x1, y1]
                        del blocks[i]
                        changed = True
                        break
                    # Overlap: if IoU > min_overlap, cut the upper block
                    elif max(r1, r2) > min_overlap:
                        # Determine which is upper (smaller y0)
                        if boxA[1] < boxB[1]:
                            # Cut boxA (upper) so it doesn't overlap with boxB
                            new_y1 = min(boxA[3], boxB[1])
                            if new_y1 > boxA[1]:
                                blocks[i]['bbox'] = [boxA[0], boxA[1], boxA[2], new_y1]
                                changed = True
                                break
                        elif boxB[1] < boxA[1]:
                            new_y1 = min(boxB[3], boxA[1])
                            if new_y1 > boxB[1]:
                                blocks[j]['bbox'] = [boxB[0], boxB[1], boxB[2], new_y1]
                                changed = True
                                break
                        # If y0 is the same, check x-axis (horizontal overlap)
                        else:
                            # leftmost is 'upper' in x
                            if boxA[0] < boxB[0]:
                                new_x1 = min(boxA[2], boxB[0])
                                if new_x1 > boxA[0]:
                                    blocks[i]['bbox'] = [boxA[0], boxA[1], new_x1, boxA[3]]
                                    changed = True
                                    break
                            elif boxB[0] < boxA[0]:
                                new_x1 = min(boxB[2], boxA[0])
                                if new_x1 > boxB[0]:
                                    blocks[j]['bbox'] = [boxB[0], boxB[1], new_x1, boxB[3]]
                                    changed = True
                                    break
                if changed:
                    break
        return blocks

    def predict(self, im, segments):
        im_shape = im.shape[:2]
        doc: pymupdf.Document = pymupdf.open()
        page: pymupdf.Page = doc.new_page(height=im_shape[0], width=im_shape[1])
        for segment in segments:
            segment_bb, segment_text = segment
            arch = pymupdf.Archive(self.font_dir)
            css_template = """
@font-face {{font-family: times; src: url(times.ttf);}}
@font-face {{font-family: times; src: url(times_bold.ttf);font-weight: bold;}}
* {{font-family: times; text-align: left; font-size: {font_size}pt;}}
"""
            font_size = self.get_best_font_size(
                page, arch, segment_bb, segment_text, 
                css_template, initial_font_size=12
            )
            css = css_template.format(font_size=font_size)
            page.insert_htmlbox(segment_bb, segment_text, css=css, archive=arch)
        
        page_blocks = page.get_text("dict", flags=11)["blocks"]
        page_blocks = self.postprocess_blocks(page_blocks)
        bbs = []
        for block in page_blocks:
            bb = list(map(int, block['bbox']))
            bbs.append(bb)
        
        # # debug
        # mat = pymupdf.Matrix(1, 1)
        # pix = page.get_pixmap(matrix=mat)
        # shape = (pix.height, pix.width, 3)
        # image = np.ndarray(shape, dtype=np.uint8, buffer=pix.samples)  # this is rgb image
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # after this it is actually bgr image

        # for b_idx, block in enumerate(page_blocks):
        #     bb = list(map(int, block['bbox']))
        #     # bb = list(map(lambda x: x*2, bb))
        #     cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)
        # cv2.imwrite(f'test.jpg', image)
        # # pdb.set_trace()

        # # debug text segments
        # image = im.copy()
        # for segment in segments:
        #     segment_bb, segment_text = segment
        #     cv2.rectangle(image, (segment_bb[0], segment_bb[1]), (segment_bb[2], segment_bb[3]), (0, 255, 0), 2)
        # cv2.imwrite(f'test_text.jpg', image)
        # pdb.set_trace()

        return bbs


class LayoutPredictor:
    def __init__(self, common_cfg, model_cfg, model_type: str):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.model_type = model_type
        
        if self.model_type == 'yolo':
            self.model = YOLOLayoutDetector(common_cfg, model_cfg)
        elif self.model_type == 'surya':
            self.model = SuryaLayoutPredictor(common_cfg, model_cfg)
        elif self.model_type == 'paddle':
            self.model = PaddleLayoutPredictor(common_cfg, model_cfg)
        else:
            raise ValueError(f'Invalid layout model type: {model_type}')
        
        self.pdf_block_gen = PDFBlock()

    def detect_boxes(self, images):
        results = self.model.predict(images)
        return results
    
    def correct_boxes(self, page_segments, boxes, scores, class_names, image):
        """
        Filter out layout boxes that are too big
        Rules:
        - contains many segments
        - number of segments in each row is not consistent (account for paragraphs)
        - For each wrong layout boxes, replace it with its segments
        """
        boxindex2segment = defaultdict(list)
        for segment in page_segments:
            segment_bb, segment_text = segment
            max_r1, max_idx = 0, None
            for box_index, box in enumerate(boxes):
                cl_name = class_names[box_index]
                if cl_name == 'table':
                    continue
                r1, r2, iou = iou_bbox(segment_bb, box)
                if r1 > max_r1:
                    max_r1, max_idx = r1, box_index
            if max_r1 >= 0.2:
                boxindex2segment[max_idx].append(segment)

        # check incorrect boxes
        cand_indexes = [box_index for box_index, segments in boxindex2segment.items() if len(segments) > 6]
        remove_indexes = []
        for box_index in cand_indexes:
            segments = boxindex2segment[box_index]
            segment_bbs = [segment[0] for segment in segments]
            rows = row_bbs(segment_bbs)
            num_segment_in_rows = [len(row) for row in rows]
            if len(list(set(num_segment_in_rows))) >= 3:
                remove_indexes.append(box_index)
        
        # for each incorrect box, replace it with the blocks generated by pdf_block_gen
        new_boxes, new_scores, new_class_names = [], [], []
        for idx, (box, score, cl_name) in enumerate(zip(boxes, scores, class_names)):
            if idx in remove_indexes:
                segments = boxindex2segment[idx]
                # new_boxes.extend(segment_bbs)
                # new_scores.extend([score] * len(segment_bbs))
                # new_class_names.extend(['text'] * len(segment_bbs))
                new_block_bbs = self.pdf_block_gen.predict(image, segments)
                new_boxes.extend(new_block_bbs)
                new_scores.extend([score] * len(new_block_bbs))
                new_class_names.extend(['text'] * len(new_block_bbs))
            else:
                new_boxes.append(box)
                new_scores.append(score)
                new_class_names.append(cl_name)
        boxes, scores, class_names = new_boxes, new_scores, new_class_names

        # # draw boxes
        # im = image.copy()
        # for box in boxes:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imwrite(f'test_before_extend.jpg', im)
        # pdb.set_trace()

        # extend and merge blocks with text segments
        boxes, scores, class_names = self.extend_and_merge_blocks(boxes, scores, class_names, page_segments)

        # # draw boxes
        # im = image.copy()
        # for box in boxes:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imwrite(f'test_extend.jpg', im)
        # pdb.set_trace()

        boxes, scores, class_names = self.remove_contained_blocks(boxes, scores, class_names, page_segments)

        return boxes, scores, class_names
    

    def extend_and_merge_blocks(self, boxes, scores, class_names, segments):
        """
        Extend and merge blocks with text segments.
        For each text segment, find the block with the highest r1 (from iou_bbox). If r1 >= 0.1 and the block does not fully contain the segment (r1 < 1.0), expand the block to cover the segment. If no block has r1 >= 0.1, add the segment as a new block.
        """
        updated_boxes = list(boxes)
        updated_scores = list(scores)
        updated_class_names = list(class_names)
        for segment_bb, segment_text in segments:
            max_r1 = 0
            max_idx = None
            for idx, box in enumerate(updated_boxes):
                r1, r2, iou = iou_bbox(segment_bb, box)
                if r1 > max_r1:
                    max_r1 = r1
                    max_idx = idx
            threshold = 0.1 if len(segment_text.split()) < 3 else 0.2
            if max_r1 >= threshold:
                # Only expand if not already fully contained
                r1, r2, iou = iou_bbox(segment_bb, updated_boxes[max_idx])
                if r1 < 1.0:
                    # Expand the block to cover the segment
                    box = updated_boxes[max_idx]
                    new_box = [
                        min(box[0], segment_bb[0]),
                        min(box[1], segment_bb[1]),
                        max(box[2], segment_bb[2]),
                        max(box[3], segment_bb[3])
                    ]
                    # Only update if the new box is different
                    if new_box != box:
                        updated_boxes[max_idx] = new_box
            else:
                # No sufficient overlap, add as new block
                updated_boxes.append(list(segment_bb))
                updated_scores.append(1.0)
                updated_class_names.append('text')
        return updated_boxes, updated_scores, updated_class_names
    

    def remove_contained_blocks(self, boxes, scores, class_names, segments=None):
        """
        Remove layout boxes that are fully contained (>95% of their area) in another box.
        Returns updated boxes, scores, and class_names.
        """
        from utils.utils import iou_bbox
        keep = [True] * len(boxes)
        for i, boxA in enumerate(boxes):
            for j, boxB in enumerate(boxes):
                if i == j:
                    continue
                r1, r2, iou = iou_bbox(boxA, boxB)
                if r1 > 0.85:
                    keep[i] = False
                    break
        filtered_boxes = [bb for bb, k in zip(boxes, keep) if k]
        filtered_scores = [s for s, k in zip(scores, keep) if k]
        filtered_class_names = [c for c, k in zip(class_names, keep) if k]
        return filtered_boxes, filtered_scores, filtered_class_names
    

    def predict(self, result):
        images = result['images']
        result['layout'] = {'boxes': [], 'scores': [], 'class_names': []}
        result['tables'] = []

        det_results = self.detect_boxes(images)

        for page_index, (image, (boxes, scores, class_names)) in enumerate(zip(images, det_results)):
            text_segments = result['ocr']['segments'][page_index]

            # # draw segment:
            # im = image.copy()
            # for segment_bb, _ in text_segments:
            #     x1, y1, x2, y2 = segment_bb
            #     cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.imwrite(f'test_segment.jpg', im)
            # pdb.set_trace()

            boxes, scores, class_names = self.correct_boxes(text_segments, boxes, scores, class_names, image)

            # draw boxes
            im = image.copy()
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(f'test.jpg', im)
            pdb.set_trace()

            result['layout']['boxes'].append(boxes)
            result['layout']['scores'].append(scores)
            result['layout']['class_names'].append(class_names)
            # process table
            tables = {'images': [], 'boxes': [], 'index': []}
            for idx, (box, cl) in enumerate(zip(boxes, class_names)):
                if cl != 'table':
                    continue
                table_image = image[box[1]:box[3], box[0]:box[2]]
                tables['images'].append(table_image)
                tables['boxes'].append(box)
                tables['index'].append(idx)
            result['tables'].append(tables)
        return result  
    
    
