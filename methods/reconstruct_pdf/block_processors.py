from copy import deepcopy
import os
import cv2
import numpy as np
from PIL import Image
import pdb
from unidecode import unidecode
import pymupdf
from pathlib import Path

from .utils import *

current_dir = Path(__file__).parent

class BaseBlockProcessor:
    def __init__(self, page: pymupdf.Page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 raw_layout, layout_content, final_layout, im):
        self.page = page
        self.im_shape = im_shape
        self.im_h, self.im_w = im_shape
        self.size_over_height_ratio = size_over_height_ratio
        self.inches_per_pixel = inches_per_pixel
        self.page_text_bb = page_text_bb
        self.mean_char_width = mean_char_width
        self.mean_char_height = mean_char_height
        self.block_idx = None
        self.raw_layout = raw_layout
        self.layout_content = layout_content
        self.final_layout = final_layout
        self.im = im

# ================= Text blocks =================
class TextBlockProcessor(BaseBlockProcessor):
    def __init__(self, page: pymupdf.Page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 raw_layout, layout_content, final_layout, im):
        super().__init__(page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                         raw_layout, layout_content, final_layout, im)
    
    def get_block_alignment(self, block_layout_content, area_bb, block_bb):
        """
            Return:
            - alignemnt type: 'Left', 'Right', 'Center', 'Justify'
            - block dist from margin
            - is first line indent ?
        """
        # DISTANCE_THRESHOLD = 5
        DISTANCE_THRESHOLD = 2 * self.mean_char_width

        alignment, block_dist_from_margin, is_first_line_indent = 'Left', 0, False
        if len(block_layout_content) == 1: # single line
            line = block_layout_content[0]
            line_text = ' '.join([word for _, word in line['words']])
            line_bb = line['coords']
            mid_x = (line_bb[0] + line_bb[2]) / 2
            mid_area_x = (area_bb[0] + area_bb[2]) / 2
            if abs(line_bb[0] - area_bb[0]) < DISTANCE_THRESHOLD:
                alignment = 'Left'
            elif abs(line_bb[2] - area_bb[2]) < DISTANCE_THRESHOLD:
                alignment = 'Right'
            elif abs(mid_x - mid_area_x) < DISTANCE_THRESHOLD:
                alignment = "Center"

        else:
            first_line_xmin = block_layout_content[0]['coords'][0]
            line_xmins = [line['coords'][0] for line in block_layout_content]
            line_xmaxs = [line['coords'][2] for line in block_layout_content]
            mid_points = [(start + end) / 2 for (start, end) in zip(line_xmins, line_xmaxs)]

            # if starts and ends of all lines are close to each other, return justify
            if (
                len(block_layout_content) >= 3 and
                abs(block_bb[0]-area_bb[0]) < DISTANCE_THRESHOLD and   # block fills column width
                abs(block_bb[2]-area_bb[2]) < DISTANCE_THRESHOLD and   # block fills column width
                all(abs(line_xmin - line_xmins[0]) < DISTANCE_THRESHOLD for line_xmin in line_xmins) and  # all lines have consistent start 
                all(abs(line_xmax - line_xmaxs[0]) < self.mean_char_width for line_xmax in line_xmaxs[:-1])  # all lines have consistent end
            ):
                alignment = "Justify"
                is_first_line_indent = False
            
            elif all(abs(mid - mid_points[0]) < DISTANCE_THRESHOLD for mid in mid_points):
                alignment = "Center"
                is_first_line_indent = False

            elif all(abs(line_xmin - line_xmins[1]) < DISTANCE_THRESHOLD for line_xmin in line_xmins[1:]):  # Excluding the first line
                if first_line_xmin - line_xmins[1] > DISTANCE_THRESHOLD:
                    alignment = 'Left'
                    is_first_line_indent = True
                
                elif all(abs(line_xmin - line_xmins[0]) < DISTANCE_THRESHOLD for line_xmin in line_xmins):
                    alignment = 'Left'
                    is_first_line_indent = False
                
            elif all(abs(line_xmax - line_xmaxs[0]) < DISTANCE_THRESHOLD for line_xmax in line_xmaxs):
                alignment = 'Right'
                is_first_line_indent = False
            
            # if self.block_idx in [0, 1, 2, 3]:
            #     print(f'Block idx: {self.block_idx}, Alignment: {alignment}')
            #     pdb.set_trace()
            
        if alignment == 'Left':
            block_dist_from_margin = block_bb[0] - area_bb[0]
        
        # if self.block_idx in [23, 24]:
        #     print(f'Block idx: {self.block_idx}, Alignment: {alignment}')
        #     im = deepcopy(self.im)
        #     cv2.rectangle(im, (block_bb[0], block_bb[1]), (block_bb[2], block_bb[3]), (0, 0, 255), 2)
        #     cv2.rectangle(im, (area_bb[0], area_bb[1]), (area_bb[2], area_bb[3]), (0, 255, 0), 2)
        #     cv2.imwrite('test.jpg', im)
        #     pdb.set_trace()

            # if self.block_idx == 1:
            #     alignment = 'Right'
            # elif self.block_idx == 2:
            #     alignment = 'Left'
            
        return alignment, block_dist_from_margin, is_first_line_indent

    def _get_font_size(self, layout_content):
        """
        Calculates font size for each line individually,
        then returns the most frequently occurring size.
        """
        # Define adjustment factors - uppercase text appears taller so needs less scaling
        case_adjustment = {
            True: 1.0,  # For uppercase text
            False: 1.1  # For lowercase text
        }
        
        # Calculate font size for each line
        line_font_sizes = []
        for line in layout_content:
            line_heights = []
            for word_box, word_text in line['words']:
                # Get word height from bounding box coordinates
                word_height = word_box[3] - word_box[1]
                # Apply case-based adjustment factor
                adjusted_height = word_height / case_adjustment[any_upper(word_text)]
                line_heights.append(adjusted_height)
            
            if line_heights:
                # Get average height for this line and convert to font size
                line_avg_height = sum(line_heights) / len(line_heights)
                line_font_size = round(line_avg_height * self.size_over_height_ratio)
                line_font_sizes.append(line_font_size)
        
        if not line_font_sizes:
            return 12  # Default font size if no lines found
            
        # Find most common font size
        size_counts = {}
        for size in line_font_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
            
        most_common_size = max(size_counts.items(), key=lambda x: x[1])[0]
        return most_common_size

    def _get_line_distance(self, layout_content, font_size):
        """Calculate the line spacing between text lines.
        
        Args:
            layout_content: List of lines with their coordinates
            font_size: Base font size in points
            
        Returns:
            float: Line spacing in points, ensuring minimum spacing relative to font size
        """
        # Calculate distances between consecutive lines
        distances = []
        for i in range(len(layout_content) - 1):
            current_line_bottom = layout_content[i]['coords'][3]
            next_line_top = layout_content[i + 1]['coords'][1] 
            distances.append(next_line_top - current_line_bottom)
            
        # Get average line spacing, default to 0 if no lines
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        # Convert pixel distance to points (72 points per inch)
        spacing_in_points = 72 * avg_distance * self.inches_per_pixel
        
        # Enforce minimum spacing of 1.2x font size
        min_spacing = font_size * 1.2
        
        return max(spacing_in_points, min_spacing)
    
    
    def is_enter_line(self, layout_content, line_index, block_bb):
        """
            condition not enter line: 
            if line_bb[2] == block_bb[2] and next_line_bb[0] == block_bb[0] and not (line.endswith(.) and next_line.startswith(uppercase))
        """
        if line_index == len(layout_content) - 1:
            return False
        
        line_bb = layout_content[line_index]['coords']
        last_line_word = layout_content[line_index]['words'][-1][1]
        next_line = layout_content[line_index + 1]
        next_line_bb = next_line['coords']
        first_next_line_word = next_line['words'][0][1]
        threshold = 2 * self.mean_char_width

        if (
            abs(line_bb[2] - block_bb[2]) < 6 * self.mean_char_width and  # line bb must stretches near the end of block bb
            abs(next_line_bb[0] - block_bb[0]) < threshold and  # next line bb must starts near the start of block bb
            not (last_line_word.endswith('.') and first_next_line_word[0].isupper())  # next line must not start with uppercase letter
        ):
            return False
        if not first_next_line_word[0].isupper():
            return False
        
        return True


    def get_best_font_size(self, page, arch, block_bb, block_text, css_template, alignment, initial_font_size):
        """
            Create a temp blank page, insert the html text, get the bounding boxes around the text,
            compare its height with the block_bb and finally adjust the font size
        """
        # Prepare a temp blank page (same size as current page)
        temp_doc = pymupdf.open()  # create a new empty PDF
        temp_page = temp_doc.new_page(width=page.rect.width, height=page.rect.height)

        # Try to fit the text in the block_bb by adjusting font size
        max_iter = 10
        min_font_size = 6
        max_font_size = 48
        target_height = int(block_bb[3] - block_bb[1]) * 1
        best_font_size = initial_font_size
        best_height_diff = float('inf')

        for _ in range(max_iter):
            css = css_template.format(alignment=alignment.lower(), font_size=initial_font_size)
            temp_page.clean_contents()  # clear previous content
            temp_page.insert_htmlbox(block_bb, block_text, css=css, archive=arch)
            # Get the bounding box of the inserted text
            words = temp_page.get_text("words")
            if not words:
                break
            y0s = [w[1] for w in words]
            y1s = [w[3] for w in words]
            text_top = min(y0s)
            text_bottom = max(y1s)
            text_height = text_bottom - text_top

            height_diff = abs(text_height - target_height)
            if height_diff < best_height_diff:
                best_height_diff = height_diff
                best_font_size = initial_font_size
                if best_height_diff / target_height < 0.05:
                    break

            # Adjust font size
            if text_height > target_height and initial_font_size > min_font_size:
                initial_font_size -= 1
            elif text_height < target_height and initial_font_size < max_font_size:
                initial_font_size += 1
            else:
                break
        
        return best_font_size

    def process(
        self, 
        page: pymupdf.Page,
        block_idx,
        block_distance, # distance to next block
        area_bb  # row bb if row as a single column, otherwise column bb
    ):
        self.block_idx = block_idx
        layout_content = self.layout_content[block_idx]
        raw_layout = self.raw_layout[block_idx]
        block_type, block_bb, _ = raw_layout

        # Determine alignment and indentation
        alignment, block_dist_from_margin, is_first_line_indent = self.get_block_alignment(layout_content, area_bb, block_bb)

        # Calculate font size based on text height
        # font_size = self._get_font_size(layout_content)
        
        # Calculate line spacing
        # line_space = self._get_line_distance(layout_content, font_size)
                
        # Process each line in the block
        block_text = []
        for line_index, line in enumerate(layout_content):
            if line_index == 0 and is_first_line_indent:
                block_text.append('\t\t')
            for _, word in line['words']:
                block_text.append(f'{word} ')
            is_enter_line = self.is_enter_line(layout_content, line_index, block_bb)
            # if is_enter_line:
            if True:
                block_text.append('\n')
            # if block_idx == 7 and line_index in [1,2,3]:
            #     print('is enter line', is_enter_line)
            #     pdb.set_trace()

        block_text = "".join(block_text).replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n', '<br>')
        # if self.block_idx in [6]:
        #     pdb.set_trace()
        arch = pymupdf.Archive(os.path.join(current_dir, 'fonts'))
        css_template = """
@font-face {{font-family: times; src: url(times.ttf);}}
@font-face {{font-family: times; src: url(times_bold.ttf);font-weight: bold;}}
* {{font-family: times; text-align: {alignment}; font-size: {font_size}pt;}}
"""
        font_size = self.get_best_font_size(
            page, arch, block_bb, block_text, 
            css_template, alignment, initial_font_size=12
        )
        css = css_template.format(alignment=alignment.lower(), font_size=font_size)
        page.insert_htmlbox(block_bb, block_text, css=css, archive=arch)

        return page


# ================= Figure blocks =================
class FigureBlockProcessor(BaseBlockProcessor):
    def __init__(self, page: pymupdf.Page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 raw_layout, layout_content, final_layout, im):
        super().__init__(page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                         raw_layout, layout_content, final_layout, im)

    def process(self, page: pymupdf.Page, block_idx):
        self.block_idx = block_idx
        image_np = self.layout_content[block_idx]
        raw_layout = self.raw_layout[block_idx]
        block_type, block_bb, _ = raw_layout
        block_w = block_bb[2] - block_bb[0]

        if image_np.dtype != np.uint8:
            image_np = (255 * image_np).astype(np.uint8)
        
        # Convert numpy array to Pixmap
        pix = pymupdf.Pixmap(image_np)
        # Insert the image into the page at the block_bb rectangle
        page.insert_image(
            block_bb,              # where to place the image (rect-like)
            pixmap=pix,            # image from pixmap
            keep_proportion=True,  # keep aspect ratio
            overlay=True,          # put in foreground
        )

        return page
    

# ================= Table blocks =================
class TableBlockProcessor(BaseBlockProcessor):
    def __init__(self, page: pymupdf.Page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 raw_layout, layout_content, final_layout, im):
        super().__init__(page, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                         raw_layout, layout_content, final_layout, im)
    
    def process(self, doc, block_idx):
        self.block_idx = block_idx
        layout_content = self.layout_content[block_idx]
        raw_layout = self.raw_layout[block_idx]
        block_type, block_bb, _ = raw_layout

        cell_data = [i['relation'] for i in layout_content['extracted_value']]
        cell_box = [i['box'] for i in layout_content['extracted_value']]
        cell_text = [i['text'] for i in layout_content['extracted_value']]
        num_rows = max(end_row for _, end_row, _, _ in cell_data) + 1
        num_columns = max(end_col for _, _, _, end_col in cell_data) + 1
        table = doc.add_table(num_rows, num_columns)
        for r in table.rows:
            for c in r.cells:
                ps = c.paragraphs
                for p in ps:
                    for run in p.runs:
                        font = run.font
                        font.size= shared.Pt(11)
        table.style = 'Table Grid'

        for i, (start_row, end_row, start_col, end_col) in enumerate(cell_data):
            cell_width = (cell_box[i][2] - cell_box[i][0]) * self.inches_per_pixel
            if start_row != end_row or start_col != end_col:
                merged_cell = table.cell(start_row, start_col).merge(table.cell(end_row, end_col))
                merged_cell.text = cell_text[i]
                merged_cell.width = shared.Inches(cell_width)
            else:
                table.cell(start_row, start_col).text = cell_text[i]
                table.cell(start_row, start_col).width = shared.Inches(cell_width)
        return doc
    