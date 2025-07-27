from copy import deepcopy
import pdb
import os
import cv2
import numpy as np
from PIL import Image
import math
from typing_extensions import List, Dict, Tuple, Optional, Any, Literal

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx import shared
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from unidecode import unidecode

from methods.layout.label_list import FINAL_LABELS
from .utils import *
from .block_processors import *

class ConverterSingle:
    def __init__(self, request_id, page_idx, image, blocks):
        self.request_id = request_id
        self.page_idx = page_idx
        self.image = deepcopy(image)
        self.blocks: List[Block] = blocks
        self.im_shape = self.image.shape[:2]
        self.im_h, self.im_w = self.im_shape


        self.content_area: List = self.get_page_content_bbox()  # the overall bb covering all blocks
        # List of row, each row is a list of column, each column is a list of block_idx
        self.final_layout = []
        self.row_column_structure = [0]
        
        # init the word doc
        self.doc = Document()
        page_width_inch = 8.5 # always take the width of US Letter: 8.5 x 11 inches, adjust the height accordingly
        self.doc.styles['Normal'].font.name = 'Times New Roman'
        section = self.doc.sections[0]
        section.page_width = shared.Inches(page_width_inch)
        section.page_height = shared.Inches(page_width_inch / self.im_w * self.im_h)
        section.top_margin = shared.Inches(self.inches_per_pixel * self.content_area[1])
        section.bottom_margin = section.top_margin
        section.left_margin = shared.Inches(self.inches_per_pixel * self.content_area[0])
        section.right_margin = section.left_margin

        # some global properties
        self.inches_per_pixel = page_width_inch / self.im_w
        self.size_over_height_ratio, self.mean_char_width, self.mean_word_height = self.calculate_some_spatial_props()
    

    def get_page_content_bbox(self):
        content_xmin, content_ymin, content_xmax, content_ymax = 10000, 10000, 0, 0
        for block in self.blocks:
            xmin, ymin, xmax, ymax = block.bbox
            content_xmin = min(content_xmin, xmin)
            content_ymin = min(content_ymin, ymin)
            content_xmax = max(content_xmax, xmax)
            content_ymax = max(content_ymax, ymax)
        return [content_xmin, content_ymin, content_xmax, content_ymax]


    def reconstruct(self):
        # Group layout block into rows and columns
        self.layout = self.build_row_and_col()
        self.log_layout()

        # merging rows that have the same number of columns
        for row in self.layout:
            num_columns = len(row)
            if num_columns != self.row_column_structure[-1]:
                self.row_column_structure.append(num_columns)
                self.final_layout.append([[] for i in range(num_columns)])
            for i in range(num_columns):
                self.final_layout[-1][i] += row[i]
        self.row_column_structure = self.row_column_structure[1:]

        # init processor for each type of block
        self.init_block_processors()

        # calculate row bboxes
        self._row_bboxes = []
        for row_idx, row in enumerate(self.final_layout):
            # get this row's bb
            row_xmin, row_xmax = self.page_text_bb[0], self.page_text_bb[2]
            row_ymin, row_ymax = 1e9, 0
            for col_idx, col in enumerate(row):
                for block_idx in col:
                    block_bb = self.blocks[block_idx].bbox
                    row_ymin = min(row_ymin, block_bb[1])
                    row_ymax = max(row_ymax, block_bb[3])
            row_bb = [row_xmin, row_ymin, row_xmax, row_ymax]
            self._row_bboxes.append(row_bb)

        print(f'Final layout: {self.final_layout}')
        # Start writing to documents block by block
        for row_idx, row in enumerate(self.final_layout):
            self.process_row(row, row_idx)


    def process_row(self, row, row_idx):
        """Process a row of layout blocks and add them to the document.
        Args:
            row (List[List[int]]): A list of columns, where each column contains indices 
                                  of layout blocks to process
            idx (int): Index of the current row in self._final_layout
        Raises:
            ValueError: If row is empty or if an unsupported alignment is specified
        """
        # Validate row is not empty
        if len(row) == 0:
            raise ValueError('Document row cannot be emptied, check the final_layout')

        column_break, paragraph = False, None
        num_col = len(row)
        section, col_widths_in_pixel = self.create_section(num_columns=num_col, row_idx=row_idx) # Create section with appropriate number of columns
        row_bb = self._row_bboxes[row_idx]
        
        # get all column's bb in the row
        left = row_bb[0]
        col_bbs = []
        row_width = row_bb[2] - row_bb[0]
        for col_idx, col in enumerate(row):
            col_width = col_widths_in_pixel[col_idx]
            col_width_ratio = col_width / sum(col_widths_in_pixel)
            col_bb = [left, row_bb[1], left + col_width_ratio * row_width, row_bb[3]]
            col_bb = list(map(int, col_bb))
            col_bbs.append(col_bb)
            left = col_bb[2]

        # Process each column in the row
        for col_idx, col in enumerate(row):
            col_bb = col_bbs[col_idx]
            # Process each block in the column
            for block_order, block_idx in enumerate(col):
                block = self.blocks[block_idx]
                block_type = block.type
                coordinates = block.bbox
                block_w = coordinates[2] - coordinates[0]  # Block width
                
                # Calculate distance to next block (for spacing)
                if block_order < len(col) - 1:
                    next_block = self.blocks[col[block_order + 1]]
                    next_coordinates = next_block.bbox
                    block_distance = max(next_coordinates[1] - coordinates[3], 0)
                else:
                    block_distance = 0

                if block_type in ['text', 'title', 'list', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                    self.doc, column_break, paragraph = self.text_processor.process(
                        self.doc, block_idx, block_distance, column_break, paragraph, area_bb=col_bb
                    )
                elif block_type == 'figure':
                    self.doc = self.figure_processor.process(self.doc, block_idx)
                elif block_type == 'table': 
                    self.doc = self.table_processor.process(self.doc, block_idx)
            
            # Add column break if needed
            if len(row) > 1 and col_idx < len(row) - 1:
                paragraph = self.doc.add_paragraph()
                self.add_column_break(paragraph)
                column_break = True


    def init_block_processors(self):
        init_args = (self.im_shape, self.size_over_height_ratio, self.inches_per_pixel, self.page_text_bb, self.mean_char_width, self.mean_word_height, 
                     self.blocks, self.final_layout, self.image)
        self.text_processor = TextBlockProcessor(*init_args)
        self.figure_processor = FigureBlockProcessor(*init_args)
        self.table_processor = TableBlockProcessor(*init_args)


    def create_section(self, num_columns: int=None, row_idx=None):
        """Create a section with the specified number of columns.

        Args:
            num_columns (int): Number of columns in the section
            idx (int): Index of the current row in self._final_layout

        Returns:
            section: A new section object with the specified number of columns
        """
        def get_col_widths(row, num_columns):
            # Calculate width of each column based on layout coordinates
            col_widths = []  # List to store column widths
            for col_idx in range(num_columns):
                min_col_x, max_col_x = 10000, 0
                # Find rightmost edge of content in this column
                for block_idx in row[col_idx]:
                    block_bb = self.blocks[block_idx].bbox
                    min_col_x = min(block_bb[0], min_col_x)
                    max_col_x = max(block_bb[2], max_col_x)
                # Column width is distance from left edge to rightmost content
                col_widths.append(max_col_x - min_col_x)
            return col_widths

        # Get the row of content we're working with
        row = self.final_layout[row_idx]
        col_widths_in_pixel = get_col_widths(row, num_columns)
            
        # Convert column widths to Word document units
        total = sum(col_widths_in_pixel)  # Total width of all columns
        total_col_width_in_unit = 12240 * (8.5 - self.margins['left'] - self.margins['right']) / 8.5
        col_widths_in_unit = []
        for col_idx in range(len(col_widths_in_pixel)):
            # Scale width accounting for page margins:
            # - 12240 is Word's internal units per page width
            # - 8.5 is standard page width in inches
            # - self._margins[0] and [2] are left/right margins
            col_widths_in_unit.append(col_widths_in_pixel[col_idx] / total * total_col_width_in_unit)
            
        
        # set distance to last section
        if row_idx > 0:
            last_row_bb = self._row_bboxes[row_idx - 1]
            row_bb = self._row_bboxes[row_idx]
            distance = row_bb[1] - last_row_bb[3]
            if distance > 0:
                # get avg line height in this row
                line_heights = []
                for col_idx in range(num_columns):
                    for block_idx in row[col_idx]:
                        block = self.blocks[block_idx]
                        block_type = block.type
                        if block_type in ['table', 'figure', 'blank']:
                            continue
                        for line in block.lines:
                            try:
                                line_heights.append(line.bbox[3] - line.bbox[1])
                            except:
                                pdb.set_trace()
                avg_line_height = sum(line_heights) / len(line_heights)
                num_line = int(distance // avg_line_height)
                blank_text = '\n' * num_line
                self.doc.add_paragraph(blank_text)
                
        # Create new section with continuous break from previous
        section = self.doc.add_section(WD_SECTION.CONTINUOUS)
        
        # Configure section for multiple columns
        sectPr = section._sectPr  # Get section properties
        # Remove existing cols element if it exists
        for child in sectPr:
            if child.tag.endswith('cols'):
                sectPr.remove(child)
                # print(f'removed {child}')
                break
        cols = OxmlElement('w:cols')  # Create columns element
        cols.set(qn('w:num'), str(num_columns))  # Set number of columns
        cols.set(qn('w:equalWidth'), '0')  # Allow unequal column widths
        
        # Add each column with calculated width
        for col_num in range(num_columns):
            block_idx = OxmlElement('w:col')
            block_idx.set(qn('w:w'), str(col_widths_in_unit[col_num]))  # Set column width
            cols.append(block_idx)
            
        # Add columns configuration to section
        sectPr.append(cols)


        # print(f'row_idx: {row_idx}')
        # if row_idx in [2]:
        #     print(f'col_widths_in_pixel: {col_widths_in_pixel}')
        #     pdb.set_trace()
        
        return section, col_widths_in_pixel
    
    def build_row_and_col(self):
        ### Groups into rows and columns
        for row in group([block.bbox for block in self.blocks], horizontal_align):
            # print(f'row: {row}')
            # pdb.set_trace()
            self.layout.append([]) 
            lrow = list(row)
            for col in group([self.blocks[i].bbox for i in lrow], vertical_align):
                lcol = list(col)
                self.layout[-1].append([lrow[lcol[i]] for i in range(len(lcol))])
        
        # sort rows and columns to reading position: top to bottom, left to right
        self.layout.sort(key=lambda x: min([self.blocks[j].bbox[1] for i in x for j in i]))
        for row in self.layout:
            # sort blocks in same row left2right
            row.sort(key=lambda x: min([self.blocks[i].bbox[0] for i in x]))
            for col in row:
                # sort blocks in same column top2bottom
                col.sort(key=lambda x: self.blocks[x].bbox[1])
        
        return self.layout
        
    
    def add_column_break(self, paragraph):
        """Add a column break to the given paragraph."""
        run = paragraph.add_run()
        br = OxmlElement('w:br')
        br.set(qn('w:type'), 'column')
        run._r.append(br)
        
    

    def calculate_some_spatial_props(self):
        """
        Calculate ratio for converting pixel heights to font sizes.
        Analyzes text blocks to determine appropriate scaling factor.
        Defaults to 0.4 if no text blocks found.
        """
        size_over_height = []
        all_char_width, all_word_height = [], []

        # check if all uppers
        is_all_upper = True
        for block_idx, block in enumerate(self.blocks):
            block_type = block.type
            if block_type not in ['text', 'list', 'title', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                continue
            for line in block.lines:
                for word_info in line.words:
                    word_text = word_info['text']
                    if not any_upper(word_text):
                        is_all_upper = False
                        break
                if not is_all_upper:
                    break

        for idx, block in enumerate(self.blocks):
            block_type = block.type
            if block_type in ['text', 'list', 'title', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                # Calculate average character width and height for lowercase words
                widths, heights = [],[]
                for line in block.lines:
                    for word_info in line.words:
                        word_text = word_info['text']
                        word_box = word_info['bbox']
                        # Skip words with uppercase letters
                        if any_upper(word_text) and not is_all_upper:
                            continue
                        # Calculate width per character
                        char_width = (word_box[2] - word_box[0]) / len(word_text)
                        widths.append(char_width)
                        # Calculate height
                        word_height = word_box[3] - word_box[1]
                        heights.append(word_height)
                all_char_width.extend(widths)
                all_word_height.extend(heights)
                        
                if len(heights) > 0:
                    # for reference: with font size = 12 and no margin, the number of characters is 90
                    # the bigger the font size, the less the number of characters
                    mean_char_width = sum(widths) / len(widths)
                    font_size = 90 * 12 / (self.im_w / mean_char_width)

                    mean_word_height = sum(heights) / len(heights)
                    size_over_height.append(font_size / mean_word_height)

                    # heights = np.array(heights)
                    # lower, upper = np.percentile(heights, [10, 90])
                    # filtered = heights[(heights >= lower) & (heights <= upper)]
                    # if len(filtered) > 0:
                    #     median_height = np.median(filtered)
                    # else:
                    #     median_height = np.median(heights)
                    # size_over_height.append(font_size / median_height)

        self.mean_char_width = sum(all_char_width) / len(all_char_width) if len(all_char_width) > 0 else 0
        self.mean_word_height = sum(all_word_height) / len(all_word_height) if len(all_word_height) > 0 else 0
        if len(size_over_height) > 0:
            self.size_over_height_ratio = sum(size_over_height) / len(size_over_height)
        else:
            self.size_over_height_ratio = 0.4
        
        return self.size_over_height_ratio, self.mean_char_width, self.mean_word_height


    def get_doc(self):
        return self.doc

    def log_layout(self):
        for row_idx, row in enumerate(self.layout):
            for col_idx, col in enumerate(row):
                for block_idx in col:
                    block = self.blocks[block_idx]
                    block_type = block.type
                    if block_type not in ['text', 'title', 'list', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                        continue
                    block_text = []
                    for line in block.lines:
                        for word_info in line.words:
                            block_text.append(word_info['text'])
                        block_text.append('\n')
                    block_text = " ".join(block_text)
                    print(f'Row {row_idx}, Column {col_idx}, Block {block_idx}: {block_text}')
        
        # self.raw_layout
        save_dir = f'debug/{self.request_id}/layout_detection'
        os.makedirs(save_dir, exist_ok=True)
        image = deepcopy(self.image)
        for idx, block in enumerate(self.blocks):
            x1, y1, x2, y2 = block.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f'{idx}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(image, f'{block.type} {block.score:.2f}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(save_dir, f'raw_layout_{self.page_idx}.jpg'), image)
