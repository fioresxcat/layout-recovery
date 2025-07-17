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

# ================= Text blocks =================
class TextBlockProcessor:
    def __init__(self, im_shape, size_over_height_ratio, inches_per_pixel):
        self._size = im_shape
        self._size_over_height_ratio = size_over_height_ratio
        self._inches_per_pixel = inches_per_pixel
    
    def _align_determined(self, layout_content):
        if len(layout_content) == 1: # single line
            pdb.set_trace()
            mid_point = (layout_content[0]['coords'][1] + layout_content[0]['coords'][3]) / 2
            half_page = self._size[1] / 2
            if abs(mid_point - half_page) < 0.1 * self._size[1]:
                return "Center", False
            elif half_page - mid_point > 0.1 * self._size[1]:
                return 'Left', False
            else:
                return 'Right', False

        first_line_start = layout_content[0]['coords'][1]
        starts = [i['coords'][1] for i in layout_content]
        ends = [i['coords'][3] for i in layout_content][:-1]
        mid_points = [(start + end) / 2 for (start, end) in zip(starts, ends)]

        if all(abs(start - starts[0]) < 5 for start in starts) and all(abs(end - ends[0]) < 5 for end in ends):
            return "Justify", False
        
        if all(abs(start - starts[1]) < 5 for start in starts[1:]):  # Excluding the first line
            if first_line_start - starts[1]> 5:
                return 'Left', True
        if all(abs(start - starts[0]) < 5 for start in starts):
            return 'Left', False
        if all(abs(mid - mid_points[0]) < 5 for mid in mid_points):
            return "Center", False
        if all(abs(end - ends[0]) < 5 for end in ends):
            return 'Right', False
        return "Justify", False


    def _get_font_size(self, layout_content):
        temp_d = {True: 1, False: 1.1}  # Adjustment factor for uppercase/lowercase
        height = [(layout_content[i]['words'][j][0][2] - layout_content[i]['words'][j][0][0]) / (temp_d[any_upper(layout_content[i]['words'][j][1])]) for i in range(len(layout_content)) for j in range(len(layout_content[i]['words']))]
        mean_height = sum(height) / len(height)
        font_size = round(mean_height * self._size_over_height_ratio)
        return font_size

    def _get_line_distance(self, layout_content, font_size):
        line_distance = [layout_content[i + 1]['coords'][0] - layout_content[i]['coords'][2] for i in range(len(layout_content) - 1)]
        line_distance = sum(line_distance) / len(line_distance) if len(line_distance) > 0 else 0

        # Apply minimum line spacing while preserving calculated spacing when larger
        MIN_LINE_SPACING_PT = font_size * 1.2
        calculated_spacing = 72 * line_distance * self._inches_per_pixel
        effective_spacing = max(calculated_spacing, MIN_LINE_SPACING_PT)

        return effective_spacing
    
    
    def process(
        self, 
        doc: Document,
        layout_content,
        block_distance, # distance to next block
        block_type, # block type
        column_break,
        paragraph
    ):
        # Determine alignment and indentation
        alignment, tab_first = self._align_determined(layout_content)
        
        # Calculate font size based on text height
        font_size = self._get_font_size(layout_content)
        
        # Calculate line spacing
        line_space = self._get_line_distance(layout_content, font_size)
                
        # Process each line in the block
        for line_number, line in enumerate(layout_content):
            # Join words and handle special cases
            line_words = [i[1] for i in line['words']]
            line_text = ' '.join(line_words)
            
            # Special handling for Vietnamese document headers
            if unidecode(line_text.lower()) == "cong hoa xa hoi chu nghia viet nam":
                line_text += '\n'
            if unidecode(line_text.lower()) == "uy ban nhan dan":
                line_text += '\n'
            if line_text == "Nơi nhận:" and line_number == 0:
                line_text += '\n'

            # Add paragraph and apply formatting
            if column_break == False: # add new paragraph if column_break is False
                paragraph = doc.add_paragraph()
            else:  # if column_break is True, add to the last paragraph and set column_break to False
                column_break = False
                
            # Configure paragraph formatting
            paragraph_format = paragraph.paragraph_format
            paragraph_format.line_spacing = shared.Pt(line_space)
            paragraph_format.space_after = shared.Pt(72 * block_distance * self._inches_per_pixel)
            paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
            
            # Add text and apply font settings
            run = paragraph.add_run(line_text)
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
                
            # Apply indentation if needed
            if tab_first:
                paragraph_format.first_line_indent = shared.Inches(0.5)
            else:
                paragraph_format.first_line_indent = None
        
        return doc, column_break, paragraph


# ================= Figure blocks =================
class FigureBlockProcessor:
    def __init__(self):
        pass

    def process(self, doc, layout_content: np.ndarray, block_w: float):
        if layout_content.dtype != np.uint8:
            layout_content = (255 * layout_content).astype(np.uint8)
        img = Image.fromarray(layout_content)
        img.save('temp.png')
        doc.add_picture('temp.png', shared.Inches(block_w * self._inches_per_pixel))
        return doc
    

# ================= Table blocks =================
class TableBlockProcessor:
    def __init__(self):
        pass
    
    def process(self, doc, layout_content):
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
            cell_width = (cell_box[i][2] - cell_box[i][0]) * self._inches_per_pixel
            if start_row != end_row or start_col != end_col:
                merged_cell = table.cell(start_row, start_col).merge(table.cell(end_row, end_col))
                merged_cell.text = cell_text[i]
                merged_cell.width = shared.Inches(cell_width)
            else:
                table.cell(start_row, start_col).text = cell_text[i]
                table.cell(start_row, start_col).width = shared.Inches(cell_width)
        return doc
    