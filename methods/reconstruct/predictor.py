import cv2
from copy import deepcopy
from pathlib import Path
import pickle
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
from .utils import *
from utils.utils import *


class ReconstructPredictor:
    def __init__(self, common_cfg, model_cfg) -> None:
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
    

    def gather_and_sort_boxes(self, result):
        """Gather text polygons into layout boxes and sort them into rows.
        
        Creates unified block structures using dataclasses from utils.py.
        Each block contains all necessary information: type, bbox, score, and content.
        """
        result['blocks'] = []
        for page_index in range(len(result['images'])):
            # Get layout information
            layout_boxes = result['layout']['boxes'][page_index]
            text_boxes = result['text_detection']['coords'][page_index]
            class_names = result['layout']['class_names'][page_index]
            scores = result['layout']['scores'][page_index]
            words = result['ocr']['raw_words'][page_index]
            
            # Create unified blocks
            blocks = []
            
            # First, gather text polygons for each layout box
            layout_texts = [[] for _ in layout_boxes]
            for i, poly in enumerate(text_boxes):
                max_r1, max_idx = 0, None
                for block_idx, box in enumerate(layout_boxes):
                    r1, r2, iou = iou_poly(poly, box)
                    if r1 > max_r1:
                        max_r1, max_idx = r1, block_idx
                if max_r1 >= 0.2:
                    layout_texts[max_idx].append(poly)
            
            # Create word mapping
            poly2word = dict(zip(text_boxes, words))
            
            # Process each layout box to create unified blocks
            for block_idx, (box, block_type, score) in enumerate(zip(layout_boxes, class_names, scores)):
                xmin, ymin, xmax, ymax = box
                bbox = (xmin, ymin, xmax, ymax)
                
                if block_type in ['text', 'title', 'list', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                    # Process text blocks
                    polys = layout_texts[block_idx]
                    poly_rows = row_polys(polys)
                    
                    # Convert to Line objects
                    lines = []
                    for row in poly_rows:
                        words_list = []
                        segments_list = []
                        line_xmin, line_xmax, line_ymin, line_ymax = 10000, 0, 10000, 0
                        
                        for poly in row:
                            word = poly2word[tuple(poly)]
                            # Convert polygon to bbox
                            xmin = min(poly[::2])
                            xmax = max(poly[::2])
                            ymin = min(poly[1::2])
                            ymax = max(poly[1::2])
                            word_bbox = (xmin, ymin, xmax, ymax)
                            
                            words_list.append({'bbox': word_bbox, 'text': word})
                            
                            # Update line bbox
                            line_xmin = min(line_xmin, xmin)
                            line_xmax = max(line_xmax, xmax)
                            line_ymin = min(line_ymin, ymin)
                            line_ymax = max(line_ymax, ymax)
                        
                        line_bbox = (line_xmin, line_ymin, line_xmax, line_ymax)
                        line = Line(words=words_list, segments=segments_list, bbox=line_bbox)
                        lines.append(line)
                    
                    # Create TextBlock
                    text_block = TextBlock(
                        type=block_type,
                        bbox=bbox,
                        score=score,
                        image=result['images'][page_index][ymin:ymax, xmin:xmax],
                        lines=lines
                    )
                    blocks.append(text_block)
                    
                elif block_type == 'figure':
                    # Create FigureBlock (using base Block class)
                    figure_block = Block(
                        type=block_type,
                        bbox=bbox,
                        score=score,
                        image=result['images'][page_index][ymin:ymax, xmin:xmax]
                    )
                    blocks.append(figure_block)
                    
                elif block_type == 'table':
                    # Find corresponding table structure
                    table_info = None
                    for table_struct in result['table_structure'][page_index]:
                        if block_idx == table_struct['layout_idx']:
                            table_info = table_struct
                            break
                    
                    if table_info:
                        # Create TableBlock
                        table_block = TableBlock(
                            type=block_type,
                            bbox=bbox,
                            score=score,
                            image=result['images'][page_index][ymin:ymax, xmin:xmax],
                            cells=table_info['extracted_value']
                        )
                        blocks.append(table_block)

            blocks = self.handle_overlapping_layout_boxes(blocks)
            result['blocks'].append(blocks)
        return result
    

    def handle_overlapping_layout_boxes(blocks):
        """Process overlapping layout boxes to handle conflicts between figures/tables and text.

        Main logic:
        1. Find all pairs of overlapping layout boxes
        2. For each overlapping pair:
           - If one is figure/table and other is text, modify or remove the text box
           - If both are figures/tables, leave them unchanged
        3. Remove any layout boxes that were completely overlapped

        This ensures figures and tables take precedence over text when they overlap.
        """
        # Store pairs of overlapping box indices
        overlap_pairs = []
        # Track indices of boxes to remove
        removed_layout = []

        # Find all pairs of overlapping boxes
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                if boxes_overlap(blocks[i].bbox, blocks[j].bbox):
                    overlap_pairs.append((i, j))

        # Process each overlapping pair
        for (i, j) in overlap_pairs:
            # Case 1: First box is figure/table
            if blocks[i].type == 'figure' or blocks[i].type == 'table':
                # Skip if second box is also figure/table
                if blocks[j].type == 'figure' or blocks[j].type == 'table':
                    pass
                else:
                    # Try to cut off overlapping portion of text box
                    modified_box = cut_off_box(blocks[j].bbox, blocks[i].bbox)
                    if modified_box:
                        blocks[i].bbox = list(modified_box)
                    else:
                        # If text box fully overlapped, mark for removal
                        removed_layout.append(i)
                continue

            # Case 2: Second box is figure/table
            if blocks[j].type == 'figure' or blocks[j].type == 'table':
                # Skip if first box is also figure/table
                if blocks[i].type == 'figure' or blocks[i].type == 'table':
                    pass
                else:
                    # Try to cut off overlapping portion of text box
                    modified_box = cut_off_box(blocks[i].bbox, blocks[j].bbox)
                    if modified_box:
                        blocks[j].bbox = list(modified_box)
                    else:
                        # If text box fully overlapped, mark for removal
                        removed_layout.append(j)
                continue
                
        # Remove fully overlapped boxes, starting from highest index
        for idx in sorted(removed_layout, reverse=True):
            del blocks[idx]

        return blocks


    def log_result(self, result):
        # layout detection result
        save_dir = f'debug/{result["request_id"]}/layout_detection'
        os.makedirs(save_dir, exist_ok=True)
        images = deepcopy(result['images'])
        for idx, (img, boxes, class_names, scores) in enumerate(zip(images, result['layout']['boxes'], result['layout']['class_names'], result['layout']['scores'])):
            for i, (box, t, score) in enumerate(zip(boxes, class_names, scores)):
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f'{i}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(img, f'{t} {score:.2f}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(save_dir, f'layout_detection_{idx}.jpg'), img)

        # text detection result
        save_dir = f'debug/{result["request_id"]}/text_detection'
        os.makedirs(save_dir, exist_ok=True)
        images = deepcopy(result['images'])
        for im, coords in zip(images, result['text_detection']['coords']):
            for i, p8_bb in enumerate(coords):
                points = np.array([[p8_bb[i], p8_bb[i+1]] for i in range(0, len(p8_bb), 2)], dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(im, [points], True, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(save_dir, f'text_detection_{idx}.jpg'), im)
                        

    def predict(self, result):
        with open('result.pkl', 'wb') as f:
            pickle.dump(result, f)

        imgs = deepcopy(result['images'])
        self.log_result(result)
        result = self.gather_and_sort_boxes(result)
        converter = ConverterMulti(result)
        doc = converter.reconstruct()
        result['final_doc'] = doc
        return result
    

if __name__ == '__main__':
    from utils.utils import load_yaml, convert_docx_to_pdf
    import pickle

    common_cfg = load_yaml('configs/common.yaml')
    model_cfg = load_yaml('configs/model.yaml')
    predictor = ReconstructPredictor(common_cfg, model_cfg)
    
    with open('result.pkl', 'rb') as f:
        result = pickle.load(f)
    result = predictor.predict(result)
    doc = result['final_doc']
    doc.save('output.docx')
    convert_docx_to_pdf('output.docx', 'output.pdf')
    print('done!')