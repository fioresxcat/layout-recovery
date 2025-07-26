from copy import deepcopy
import pdb
import cv2
import numpy as np
from PIL import Image
import math
from typing_extensions import List, Dict, Tuple, Optional, Any, Literal

import pymupdf
from unidecode import unidecode

from methods.layout.label_list import FINAL_LABELS
from .utils import group, boxes_overlap, cut_off_box, vertical_align, horizontal_align, any_upper
from .block_processors import *



class SinglePageConverter:
    def __init__(self, page_idx, image, class_names, boxes, scores, layout_texts, table_structures):
        self.page_idx = page_idx
        self.image = deepcopy(image)
        self.class_names = class_names
        self.boxes = boxes
        self.scores = scores
        self.layout_texts = layout_texts
        self.table_structures = table_structures
        
        self.im_shape = self.image.shape[:2]
        self.im_h, self.im_w = self.im_shape
        self.inches_per_pixel = 8.5 / self.im_w
        self.supported_classes = FINAL_LABELS
        self.raw_layout = []
        self.layout = []
        self.final_layout = []
        self.layout_content = []
        self.row_column_structure = [0]


    def reconstruct(self, page: pymupdf.Page):
        # Add raw layout as list of layout blocks
        self.raw_layout = self.add_raw_layout()

        # Build layout content by merging layout blocks and OCR results
        self.layout_content = self.build_layout_content()

        # Group layout block into rows and columns
        self.layout = self.build_row_and_col()
        self.log_layout()

        # build final layout from layout by merging rows that have the same number of columns
        for row in self.layout:
            num_columns = len(row)
            if num_columns != self.row_column_structure[-1]:
                self.row_column_structure.append(num_columns)
                self.final_layout.append([[] for i in range(num_columns)])
            for i in range(num_columns):
                self.final_layout[-1][i] += row[i]
        self.row_column_structure = self.row_column_structure[1:]
        self.size_over_height_ratio, self.mean_char_width, self.mean_word_height = self.calculate_some_spatial_props()

        # init processor for each type of block
        self.init_block_processors(page)

        # calculate row bboxes
        self._row_bboxes = []
        for row_idx, row in enumerate(self.final_layout):
            # get this row's bb
            row_xmin, row_xmax = self.page_text_bb[0], self.page_text_bb[2]
            row_ymin, row_ymax = 1e9, 0
            for col_idx, col in enumerate(row):
                for block_idx in col:
                    _, block_bb, _ = self.raw_layout[block_idx]
                    row_ymin = min(row_ymin, block_bb[1])
                    row_ymax = max(row_ymax, block_bb[3])
            row_bb = [row_xmin, row_ymin, row_xmax, row_ymax]
            self._row_bboxes.append(row_bb)
        

        print(f'Final layout: {self.final_layout}')

        # Start writing to documents block by block
        for row_idx, row in enumerate(self.final_layout):
            self.process_row(page, row, row_idx)


    def process_row(self, page: pymupdf.Page, row: List[List[int]], row_idx: int):
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

        num_col = len(row)
        row_bb = self._row_bboxes[row_idx]
        
        # get all column's real bb in the row (calculate by actual bounding box sizes)
        col_widths = []
        for col_idx, col in enumerate(row):
            # get col width
            col_width = 0
            for block_idx in col:
                _, (xmin, ymin, xmax, ymax), _ = self.raw_layout[block_idx]
                col_width = max(col_width, xmax - xmin)
            col_widths.append(col_width)
        
        # get all cols proportional bb in the row
        left = row_bb[0]
        col_bbs = []
        row_width = row_bb[2] - row_bb[0]
        for col_idx, col in enumerate(row):
            col_width = col_widths[col_idx]
            col_width_ratio = col_width / sum(col_widths)
            col_bb = [left, row_bb[1], left + col_width_ratio * row_width, row_bb[3]]
            col_bb = list(map(int, col_bb))
            col_bbs.append(col_bb)
            left = col_bb[2]

        # Process each column in the row
        for col_idx, col in enumerate(row):
            col_bb = col_bbs[col_idx]
            # Process each block in the column
            for block_order, block_idx in enumerate(col):
                layout_content = self.layout_content[block_idx]
                block_type, coordinates, _ = self.raw_layout[block_idx]
                block_w = coordinates[2] - coordinates[0]  # Block width
                
                # Calculate distance to next block (for spacing)
                if block_order < len(col) - 1:
                    next_block = self.raw_layout[col[block_order + 1]]
                    _, (next_xmin, next_ymin, next_xmax, next_ymax), _ = next_block
                    block_distance = max(next_ymin - coordinates[3], 0)
                else:
                    block_distance = 0

                if block_type in ['text', 'title', 'list', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                    self.text_processor.process(
                        page, block_idx, block_distance, area_bb=col_bb
                    )
                # elif block_type == 'figure':
                #     self.figure_processor.process(page, block_idx)
                # elif block_type == 'table': 
                #     self.table_processor.process(page, block_idx)
            

    def init_block_processors(self, page: pymupdf.Page):
        init_args = (page, self.im_shape, self.size_over_height_ratio, self.inches_per_pixel, self.page_text_bb, self.mean_char_width, self.mean_word_height, 
                     self.raw_layout, self.layout_content, self.final_layout, self.image)
        self.text_processor = TextBlockProcessor(*init_args)
        self.figure_processor = FigureBlockProcessor(*init_args)
        self.table_processor = TableBlockProcessor(*init_args)


    def add_raw_layout(self):
        """Add raw layout information to self._raw_layout and set document margins.

        self._raw_layout structure:
        [
            # Each element represents one layout block:
            [
                name,       # String - The class name of the layout block (e.g. 'text', 'table', etc)
                [xmin, ymin, xmax, ymax],  # List[int] - Coordinates of bounding box
                score      # Float - Confidence score from layout detection
            ],
            ...
        ]

        Example:
        self._raw_layout = [
            ['text', [100, 50, 300, 100], 0.95],  # A text block
            ['table', [50, 200, 400, 500], 0.87], # A table block
            ...
        ]
        """
        # Initialize page boundaries to find margins
        self.text_xmin, self.text_xmax, self.text_ymin, self.text_ymax = 10000, 0, 10000, 0

        # Iterate through layout blocks and build self._raw_layout
        for name, coordinates, score in zip(self.class_names, self.boxes, self.scores):
            xmin, ymin, xmax, ymax = coordinates
            
            # Track overall page boundaries
            self.text_xmin, self.text_xmax = min(self.text_xmin, xmin), max(self.text_xmax, xmax)
            self.text_ymin, self.text_ymax = min(self.text_ymin, ymin), max(self.text_ymax, ymax)

            # Add layout block info
            l = [name, [xmin, ymin, xmax, ymax], score]
            self.raw_layout.append(l)
        
        self.page_text_bb = [self.text_xmin, self.text_ymin, self.text_xmax, self.text_ymax]
        # cv2.rectangle(self._image, (self.text_xmin, self.text_ymin), (self.text_xmax, self.text_ymax), (0, 0, 255), 2)
        # cv2.imwrite('test.jpg', self._image)
        # pdb.set_trace()

        # Process layout further
        self.raw_layout = self.handle_overlapping_layout_boxes()
        return self.raw_layout
        

    def handle_overlapping_layout_boxes(self):
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
        for i in range(len(self.raw_layout)):
            for j in range(i + 1, len(self.raw_layout)):
                if boxes_overlap(self.raw_layout[i][1], self.raw_layout[j][1]):
                    overlap_pairs.append((i, j))

        # Process each overlapping pair
        for (i, j) in overlap_pairs:
            # Case 1: First box is figure/table
            if self.raw_layout[i][0] == 'figure' or self.raw_layout[i][0] == 'table':
                # Skip if second box is also figure/table
                if self.raw_layout[j][0] == 'figure' or self.raw_layout[j][0] == 'table':
                    pass
                else:
                    # Try to cut off overlapping portion of text box
                    modified_box = cut_off_box(self.raw_layout[j][1], self.raw_layout[i][1])
                    if modified_box:
                        self.raw_layout[i][1] = list(modified_box)
                    else:
                        # If text box fully overlapped, mark for removal
                        removed_layout.append(i)
                continue

            # Case 2: Second box is figure/table
            if self.raw_layout[j][0] == 'figure' or self.raw_layout[j][0] == 'table':
                # Skip if first box is also figure/table
                if self.raw_layout[i][0] == 'figure' or self.raw_layout[i][0] == 'table':
                    pass
                else:
                    # Try to cut off overlapping portion of text box
                    modified_box = cut_off_box(self.raw_layout[i][1], self.raw_layout[j][1])
                    if modified_box:
                        self.raw_layout[j][1] = list(modified_box)
                    else:
                        # If text box fully overlapped, mark for removal
                        removed_layout.append(j)
                continue
                
        # Remove fully overlapped boxes, starting from highest index
        for idx in sorted(removed_layout, reverse=True):
            del self.raw_layout[idx]

        return self.raw_layout

    def build_layout_content(self):
        """
        Build layout content from raw layout and OCR results.
        The resulting self._layout_content structure looks like:
        [
            # For text/title/list blocks:
            [
                # First line in text block
                {
                    'words': [
                        ((xmin,ymin,xmax,ymax), "word1"),  # (word bbox, text)
                        ((xmin,ymin,xmax,ymax), "word2"),
                        ...
                    ],
                    'coords': (line_xmin,line_ymin,line_xmax,line_ymax)  # Overall line bbox
                },
                # Second line
                {
                    'words': [...],
                    'coords': (...)
                },
                ...
            ],

            # For figure blocks:
            numpy.ndarray,  # Image array of figure region

            # For table blocks:
            {
                'layout_idx': int,      # Index in layout
                'image_index': int,     # Page number
                'extracted_value': [     # Table cell contents
                    ["cell1", "cell2"],  # First row
                    ["cell3", "cell4"],  # Second row
                    ...
                ]
            },
            ...
        ]
        """
        removed = []
        for idx, (block_type, coordinates, _) in enumerate(self.raw_layout):
            xmin, ymin, xmax, ymax = [int(i) for i in coordinates]
            block_image = self.image[ymin:ymax, xmin:xmax]

            if block_type in ['text', 'title', 'list', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                ocr_result = self.layout_texts[idx]
                if len(ocr_result) == 0:
                    removed.append(idx)
                # Straighten skewed boxes ?
                new_result = []
                for line in ocr_result:
                    d = {'words': []}
                    line_xmin = 10000
                    line_xmax = 0
                    line_ymin = 10000
                    line_ymax = 0
                    for poly, word in line:
                        xmin = min(poly[::2])
                        line_xmin = min(line_xmin, xmin)
                        xmax = max(poly[::2])
                        line_xmax = max(line_xmax, xmax)
                        ymin = min(poly[1::2])
                        line_ymin = min(line_ymin, ymin)
                        ymax = max(poly[1::2])
                        line_ymax = max(line_ymax, ymax)
                        d['words'].append(((xmin, ymin, xmax, ymax), word))
                    d['coords'] = (line_xmin, line_ymin, line_xmax, line_ymax)
                    new_result.append(d)
                self.layout_content.append(new_result)

            elif block_type == 'figure':
                self.layout_content.append(block_image)

            elif block_type == 'table':
                for table_info in self.table_structures:
                    if idx == table_info['layout_idx'] and self.page_idx == table_info['image_index']:
                        self.layout_content.append(table_info)
                        if table_info['extracted_value'] == []:
                            removed.append(idx)
                        break
                else:
                    raise ValueError('Table structure not found!')
            
            elif block_type == 'blank':
                removed.append(idx)

            else:
                raise ValueError(f'{block_type} layout is not supported, check layout analysis model')
        
        for idx in sorted(removed, reverse=True):
            del self.raw_layout[idx]
            del self.layout_content[idx]

        return self.layout_content

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
                    _, block_bb, _ = self.raw_layout[block_idx]
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
            
        # Create new section with continuous break from previous
        section = self.doc.add_section(WD_SECTION.CONTINUOUS)

        # set distance to last section
        if row_idx > 0:
            last_row_bb = self._row_bboxes[row_idx - 1]
            row_bb = self._row_bboxes[row_idx]
            distance = row_bb[1] - last_row_bb[3]
            if distance > 0:
                distance_in_inches = distance * self.inches_per_pixel
                section.top_margin = shared.Inches(distance_in_inches)
        
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
        for row in group(self.raw_layout, horizontal_align):
            # print(f'row: {row}')
            # pdb.set_trace()
            self.layout.append([]) 
            lrow = list(row)
            for col in group([self.raw_layout[i] for i in lrow], vertical_align):
                lcol = list(col)
                self.layout[-1].append([lrow[lcol[i]] for i in range(len(lcol))])
        
        #Shuffle rows and columns to reading position: top to bottom, left to right
        self.layout.sort(key=lambda x: min([self.raw_layout[j][1][1] for i in x for j in i]))
        for row in self.layout:
            # sort blocks in same row left2right
            row.sort(key=lambda x: min([self.raw_layout[i][1][0] for i in x]))
            for col in row:
                # sort blocks in same column top2bottom
                col.sort(key=lambda x: self.raw_layout[x][1][1])
        
        return self.layout
        
    
    def add_column_break(self, paragraph):
        """Add a column break to the given paragraph."""
        run = paragraph.add_run()
        br = OxmlElement('w:br')
        br.set(qn('w:type'), 'column')
        run._r.append(br)
        return
    

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
        for block_idx, block_content in enumerate(self.layout_content):
            block_type, _, _ = self.raw_layout[block_idx]
            if block_type not in ['text', 'list', 'title', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                continue
            for line in block_content:
                for _, word in line['words']:
                    if not any_upper(word):
                        is_all_upper = False
                        break
                if not is_all_upper:
                    break

        for idx, (block_type, coordinates, _) in enumerate(self.raw_layout):
            if block_type in ['text', 'list', 'title', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                content = self.layout_content[idx]

                # Calculate average character width and height for lowercase words
                widths, heights = [],[]
                for line in content:
                    for word in line['words']:
                        word_text = word[1]
                        word_box = word[0]
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
                    content = self.layout_content[block_idx]
                    block_type, block_bb, _ = self.raw_layout[block_idx]
                    if block_type not in ['text', 'title', 'list', 'table_of_contents', 'header', 'footer', 'caption', 'equation', 'footnote', 'handwriting']:
                        continue
                    block_text = []
                    for line in content:
                        for word in line['words']:
                            block_text.append(word[1])
                        block_text.append('\n')
                    block_text = " ".join(block_text)
                    print(f'Row {row_idx}, Column {col_idx}, Block {block_idx}: {block_text}')



class PDFConverter:
    def __init__(self, result):
        self.convertes = []
        for page_idx in range(len(result['images'])):
            self.convertes.append(
                SinglePageConverter(
                    page_idx=page_idx, image=result['images'][page_idx],
                    class_names=result['layout']['class_names'][page_idx],
                    boxes=result['layout']['boxes'][page_idx],
                    scores=result['layout']['scores'][page_idx],
                    layout_texts=result['reconstruct'][page_idx]['layout_texts'],
                    table_structures=result['table_structure'][page_idx]
                )
            )
        self.num_pages = len(result['images'])
        # self.paper_size = pymupdf.paper_size('letter')
        self.paper_size = result['images'][0].shape[:2][::-1] # (width, height)

    def reconstruct(self):
        doc = pymupdf.open()
        for page_idx in range(self.num_pages):
            page = doc.new_page(width=self.paper_size[0], height=self.paper_size[1])
            self.convertes[page_idx].reconstruct(page)
        doc.subset_fonts(verbose=True)
        
        return doc