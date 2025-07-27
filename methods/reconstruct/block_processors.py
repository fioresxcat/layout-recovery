from copy import deepcopy
import cv2
import numpy as np
from PIL import Image
import pdb
from unidecode import unidecode

from docx import Document
from docx import shared
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx import shared
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from .utils import *

class BaseBlockProcessor:
    def __init__(self, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 layout_blocks, final_layout, im):
        self.im_shape = im_shape
        self.im_h, self.im_w = im_shape
        self.size_over_height_ratio = size_over_height_ratio
        self.inches_per_pixel = inches_per_pixel
        self.page_text_bb = page_text_bb
        self.mean_char_width = mean_char_width
        self.mean_char_height = mean_char_height
        self.block_idx = None
        self.layout_blocks = layout_blocks
        self.final_layout = final_layout
        self.im = im

# ================= Text blocks =================
class TextBlockProcessor(BaseBlockProcessor):
    def __init__(self, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 layout_blocks, final_layout, im):
        super().__init__(im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                         layout_blocks, final_layout, im)
    
    def get_block_alignment(self, block, area_bb, block_bb):
        """
            Return:
            - alignemnt type: 'Left', 'Right', 'Center', 'Justify'
            - block dist from margin
            - is first line indent ?
        """
        # DISTANCE_THRESHOLD = 5
        DISTANCE_THRESHOLD = 2 * self.mean_char_width

        alignment, block_dist_from_margin, is_first_line_indent = 'Left', 0, False
        lines = block.lines
        if len(lines) == 1: # single line
            line = lines[0]
            line_text = ' '.join([word_info['text'] for word_info in line.words])
            line_bb = line.bbox
            mid_x = (line_bb[0] + line_bb[2]) / 2
            mid_area_x = (area_bb[0] + area_bb[2]) / 2
            if abs(line_bb[0] - area_bb[0]) < DISTANCE_THRESHOLD:
                alignment = 'Left'
            elif abs(line_bb[2] - area_bb[2]) < DISTANCE_THRESHOLD:
                alignment = 'Right'
            elif abs(mid_x - mid_area_x) < DISTANCE_THRESHOLD:
                alignment = "Center"

        else:
            first_line_xmin = lines[0].bbox[0]
            line_xmins = [line.bbox[0] for line in lines]
            line_xmaxs = [line.bbox[2] for line in lines]
            mid_points = [(start + end) / 2 for (start, end) in zip(line_xmins, line_xmaxs)]

            # if starts and ends of all lines are close to each other, return justify
            if (
                len(lines) >= 3 and
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
            
            # if self.block_idx in [6, 8]:
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

    def _get_font_size(self, block):
        """
        Calculates font size for each line individually,
        then returns the most frequently occurring size.
        """
        if not hasattr(block, 'lines'):  # Not a TextBlock
            return 12  # Default font size
            
        # Define adjustment factors - uppercase text appears taller so needs less scaling
        case_adjustment = {
            True: 1.0,  # For uppercase text
            False: 1.1  # For lowercase text
        }
        
        # Calculate font size for each line
        line_font_sizes = []
        for line in block.lines:
            line_heights = []
            for word_info in line.words:
                # Get word height from bounding box coordinates
                word_box = word_info['bbox']
                word_text = word_info['text']
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

        # neu dong tiep theo bat dau voi ki tu la -> la xuong dong
        if (len(first_next_line_word) >= 1 and first_next_line_word[0].isupper()) or (len(first_next_line_word) >= 2 and first_next_line_word[1].isupper()) or (len(first_next_line_word) >= 1 and not first_next_line_word[0].isalnum()):
            return True

        if (
            abs(line_bb[2] - block_bb[2]) < 6 * self.mean_char_width and 
            abs(next_line_bb[0] - block_bb[0]) < threshold and 
            not (last_line_word.endswith('.') and first_next_line_word[0].isupper())
        ):
            return False
        if not first_next_line_word[0].isupper():
            return False
        
        return True


    def process(
        self, 
        doc: Document,
        block_idx,
        block_distance, # distance to next block
        column_break,
        paragraph, # old first paragraph of the new column if column_break is True
        area_bb  # row bb if row as a single column, otherwise column bb
    ):
        self.block_idx = block_idx
        layout_content = self.layout_content[block_idx]
        raw_layout = self.raw_layout[block_idx]
        block_type, block_bb, _ = raw_layout

        # Determine alignment and indentation
        alignment, block_dist_from_margin, is_first_line_indent = self.get_block_alignment(layout_content, area_bb, block_bb)

        # Calculate font size based on text height
        font_size = self._get_font_size(layout_content)
        
        # Calculate line spacing
        line_space = self._get_line_distance(layout_content, font_size)
                
        # Process each line in the block
        block_text = []
        for line_index, line in enumerate(layout_content):
            # Join words and handle special cases
            for _, word in line['words']:
                block_text.append(f'{word} ')
            is_enter_line = self.is_enter_line(layout_content, line_index, block_bb)
            if is_enter_line:
                block_text.append('\n')
            # if block_idx == 7 and line_index in [1,2,3]:
            #     print('is enter line', is_enter_line)
            #     pdb.set_trace()
            
        block_text = "".join(block_text)

        # print block info
        print(f'\n====== Block {block_idx} ======')
        print(f'Alignment: {alignment}')
        print(f'Block text: {block_text}')
        print(f'Is first line indent: {is_first_line_indent}')
        print(f'Font size: {font_size}')


        # Add paragraph and apply formatting
        if column_break == False: # add new paragraph if column_break is False
            paragraph = doc.add_paragraph()
        else:  # if column_break is True, add to the last paragraph and set column_break to False
            column_break = False
            
        # Configure paragraph formatting
        paragraph_format = paragraph.paragraph_format
        paragraph_format.line_spacing = shared.Pt(line_space)
        paragraph_format.space_after = shared.Pt(72 * block_distance * self.inches_per_pixel)
        paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        
        # Add text and apply font settings
        run = paragraph.add_run(block_text)
        run.font.size = shared.Pt(font_size)
        if block_type == 'title':
            run.bold = True
            
        # Apply alignment
        if alignment == 'Left':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif alignment == 'Right':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        elif alignment == 'Center':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif alignment == 'Justify':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        else:
            raise ValueError(f'{alignment} alignment is not supported')
            
        # Apply first line indentation if needed
        if is_first_line_indent:
            paragraph_format.first_line_indent = shared.Inches(0.5)
        else:
            paragraph_format.first_line_indent = None

        if alignment == 'Left' and block_dist_from_margin > 0:
            inch_per_tab = 0.5
            left_indent = block_dist_from_margin * self.inches_per_pixel
            if left_indent > 0.3 and left_indent < 0.6: # indent a single tab
                paragraph_format.left_indent = shared.Inches(inch_per_tab)
            elif left_indent >= 0.6: # indent two tabs
                paragraph_format.left_indent = shared.Inches(inch_per_tab * 2)
            # Do nothing if left_indent <= 0.5
        
        return doc, column_break, paragraph


# ================= Figure blocks =================
class FigureBlockProcessor(BaseBlockProcessor):
    def __init__(self, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 raw_layout, layout_content, final_layout, im):
        super().__init__(im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                         raw_layout, layout_content, final_layout, im)

    def process(self, doc, block_idx):
        self.block_idx = block_idx
        layout_content = self.layout_content[block_idx]
        raw_layout = self.raw_layout[block_idx]
        block_type, block_bb, _ = raw_layout
        block_w = block_bb[2] - block_bb[0]

        if layout_content.dtype != np.uint8:
            layout_content = (255 * layout_content).astype(np.uint8)
        img = Image.fromarray(layout_content)
        img.save('temp.png')
        doc.add_picture('temp.png', shared.Inches(block_w * self.inches_per_pixel))
        return doc
    

# ================= Table blocks =================
class TableBlockProcessor(BaseBlockProcessor):
    def __init__(self, im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
                 raw_layout, layout_content, final_layout, im):
        super().__init__(im_shape, size_over_height_ratio, inches_per_pixel, page_text_bb, mean_char_width, mean_char_height,
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
    