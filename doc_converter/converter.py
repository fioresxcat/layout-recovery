from copy import deepcopy

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH


from docx import shared
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import numpy as np
from PIL import Image
import math

from unidecode import unidecode

from .utils import group, boxes_overlap, cut_off_box, vertical_align, horizontal_align, any_upper

class Converter:

    def __init__(self, page_idx, image, class_names, boxes, scores, layout_texts, table_structures):
        self._page_idx = page_idx
        self._image = deepcopy(image)
        self._class_names = class_names
        self._boxes = boxes
        self._scores = scores
        self._layout_texts = layout_texts
        self._table_structures = table_structures

        self._size = self._image.shape[:2]
        self._supported_classes = ['text', 'title', 'figure', 'table', 'list']
        self._raw_layout = []
        self._layout = []
        self._final_layout = []
        self._doc = Document()
        self._doc.styles['Normal'].font.name = 'Times New Roman'
        self._layout_content = []
        self._row_column_structure = [0]
        self._inches_per_pixel = 8.5 / self._size[1]
        sections = self._doc.sections
        for section in sections:
            section.page_width = shared.Inches(self._inches_per_pixel*self._size[1])
            section.page_height = shared.Inches(self._inches_per_pixel*self._size[0])
        self._size_over_height_ratio = None


    def reconstruct(self):
        self._add_raw_layout()
        self._process_layout()
        #self._convert()

    def _add_raw_layout(self):
        page_xmin = 10000
        page_xmax = 0
        page_ymin = 10000
        page_ymax = 0
        for name, coordinates, score in zip(self._class_names, self._boxes, self._scores):
            ymin, xmin, ymax, xmax = coordinates
            page_xmin = min(page_xmin, xmin)
            page_xmax = max(page_xmax, xmax)
            page_ymin = min(page_ymin, ymin)
            page_ymax = max(page_ymax, ymax)
            l = [name, [xmin, ymin, xmax, ymax], score]
            self._raw_layout.append(l)
        # Set margin
        self._margins = [page_xmin*self._inches_per_pixel, (self._size[0] - page_xmax)*self._inches_per_pixel, page_ymin*self._inches_per_pixel,(self._size[1] - page_ymax)*self._inches_per_pixel]
        sections = self._doc.sections
        for section in sections:
            section.top_margin = shared.Inches(self._margins[0])
            section.bottom_margin = shared.Inches(self._margins[1])
            section.left_margin = shared.Inches(self._margins[2])
            section.right_margin = shared.Inches(self._margins[3])
        self._postprocess_layout()
        self._ocr_layout()
        
        # Set margin


    def _process_layout(self):
        #Calculate margin
        x_minimal = 0
        y_minimal = 0
        x_maximal = self._size[0]
        y_maximal = self._size[0]
        for _, coordinates, _ in self._raw_layout:
            xmin, ymin, xmax, ymax = [int(i) for i in coordinates]
            if xmin > x_minimal:
                x_minimal = xmin
            if ymin > y_minimal:
                y_minimal = ymin
            if xmax < x_maximal:
                x_maximal = xmax
            if ymax < y_maximal:
                y_maximal = ymax
        
        
        #Group layout block into rows and columns
        self._process_structure()
        #Based on layout groups to determine row - columns
        for row in self._layout:
            num_columns = len(row)
            if num_columns != self._row_column_structure[-1]:
                self._row_column_structure.append(num_columns)
                self._final_layout.append([[] for i in range(num_columns)])
            for i in range(num_columns):
                self._final_layout[-1][i] += row[i]
        self._row_column_structure = self._row_column_structure[1:]
        self._calculate_size_ratio()
        #Start writing to documents block by block
        for idx, row in enumerate(self._final_layout):
            self._process_row(row, idx)
    
    def _convert(self):
        self._doc.save('output.docx')
        return
    
    def _process_row(self, row, idx):
        if len(row) == 0:
            raise ValueError('Document row cannot be emptied, check the final_layout')
        column_break = False
        section = self._create_section(num_columns=len(row), idx=idx)
        for counter, col in enumerate(row):
            for block_order, block in enumerate(col):
                content = self._layout_content[block]
                block_type, coordinates, _ = self._raw_layout[block]
                block_w = coordinates[3] - coordinates[1]
                if block_order < len(col) - 1:
                    block_distance = self._raw_layout[col[block_order + 1]][1][0] - coordinates[2]
                else:
                    block_distance = 0
                if block_type == 'text' or block_type == 'title' or block_type == 'list':
                    alignment, tab_first = self._align_determined(content)
                    temp_d = {True: 1, False: 1.1}
                    height = [(content[i]['words'][j][0][2] - content[i]['words'][j][0][0]) / (temp_d[any_upper(content[i]['words'][j][1])]) for i in range(len(content)) for j in range(len(content[i]['words']))]
                    mean_height = sum(height) / len(height)
                    font_size = round(mean_height * self._size_over_height_ratio)
                    line_distance = [content[i + 1]['coords'][0] - content[i]['coords'][2] for i in range(len(content) - 1)]
                    line_distance = sum(line_distance) / len(line_distance) if len(line_distance) > 0 else 0
                    p = []
                    flag = False
                    for line_number, line in enumerate(content):
                        p += [i[1] for i in line['words']]
                        if unidecode(' '.join(p)).lower() == "cong hoa xa hoi chu nghia viet nam":
                            p += '\n'
                        if ' '.join(p) == "Nơi nhận:" and line_number == 0:
                            flag = True
                        if flag:
                            p += '\n'
                    p = ' '.join(p)
                    if not column_break:
                        paragraph = self._doc.add_paragraph()
                    else:
                        column_break = False
                    paragraph_format = paragraph.paragraph_format
                    paragraph_format.line_spacing = shared.Pt(72 * line_distance * self._inches_per_pixel)
                    paragraph_format.space_after = shared.Pt(72 * block_distance * self._inches_per_pixel)  # 0.5 inches in points
                    paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
                                   
                    run = paragraph.add_run(p)
                    run.font.size = shared.Pt(font_size)
                    if block_type == 'title':
                        run.bold = True
                    if alignment == 'Left':
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT  # Options are LEFT, CENTER, RIGHT, JUSTIFY
                    elif alignment == 'Right':
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT  # Options are LEFT, CENTER, RIGHT, JUSTIFY
                    elif alignment == 'Center':
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER  # Options are LEFT, CENTER, RIGHT, JUSTIFY
                    elif alignment == 'Justify':
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY  # Options are LEFT, CENTER, RIGHT, JUSTIFY
                    else:
                        raise ValueError(f'{alignment} alignment is not supported')

                    if tab_first:
                        paragraph_format.first_line_indent = shared.Inches(0.5)
                    else:
                        paragraph_format.first_line_indent = None
                elif block_type == 'figure':
                    self._save_image_from_array(content, 'temp.png')
                    self._doc.add_picture('temp.png', shared.Inches(block_w * self._inches_per_pixel))
                else: 
                    self._create_table(content)
                    
            if len(row) > 1 and counter < len(row) - 1:
                paragraph = self._doc.add_paragraph()
                self._add_column_break(paragraph)
                column_break = True
        return

    def _create_section(self, num_columns: int=None, idx=None):
        row = self._final_layout[idx]
        w = []
        left = 0
        for i in range(num_columns):
            maxy = 0
            for col in row[i]:
                maxy = max(self._raw_layout[col][1][3], maxy)
            w.append(maxy - left)
            left = maxy
        total = sum(w)
        for i in range(len(w)):
            w[i] = w[i] / total  * 12240 * (8.5 - self._margins[0] - self._margins[2]) / 8.5
        section = self._doc.add_section(WD_SECTION.CONTINUOUS)
        sectPr = section._sectPr  # Get the section properties element
        cols = OxmlElement('w:cols')  # Create a new 'w:cols' element
        cols.set(qn('w:num'), str(num_columns))  # Set the number of columns
        cols.set(qn('w:equalWidth'), '0')
        for col_num in range(num_columns):
            col = OxmlElement('w:col')
            col.set(qn('w:w'), str(w[col_num]))
            cols.append(col)
        sectPr.append(cols) 
        return section
        
    
    def _process_structure(self):
        ### Groups into rows and columns
        for row in group(self._raw_layout, vertical_align):
            self._layout.append([]) 
            lrow = list(row)
            for col in group([self._raw_layout[i] for i in lrow], horizontal_align):
                lcol = list(col)
                self._layout[-1].append([lrow[lcol[i]] for i in range(len(lcol))])
        
        #Shuffle rows and columns to reading position: top to bottom, left to right
        self._layout.sort(key=lambda x: min([self._raw_layout[j][1][0] for i in x for j in i]))
        for row in self._layout:
            row.sort(key=lambda x: min([self._raw_layout[i][1][1] for i in x]))
            for col in row:
                col.sort(key=lambda x: self._raw_layout[x][1][0])
        
        
    def _postprocess_layout(self):
        overlap_pairs = []
        removed_layout = []
        for i in range(len(self._raw_layout)):
            for j in range(i + 1, len(self._raw_layout)):
                if boxes_overlap(self._raw_layout[i][1], self._raw_layout[j][1]):
                    overlap_pairs.append((i, j))
        for (i, j) in overlap_pairs:
            if self._raw_layout[i][0] == 'figure' or self._raw_layout[i][0] == 'table':
                if self._raw_layout[j][0] == 'figure' or self._raw_layout[j][0] == 'table':
                    pass
                else:
                    modified_box = cut_off_box(self._raw_layout[j][1], self._raw_layout[i][1])
                    if modified_box:
                        self._raw_layout[i][1] = list(modified_box)
                    else:
                        removed_layout.append(i)
                continue
            if self._raw_layout[j][0] == 'figure' or self._raw_layout[j][0] == 'table':
                if self._raw_layout[i][0] == 'figure' or self._raw_layout[i][0] == 'table':
                    pass
                else:
                    modified_box = cut_off_box(self._raw_layout[i][1], self._raw_layout[j][1])
                    if modified_box:
                        self._raw_layout[j][1] = list(modified_box)
                    else:
                        removed_layout.append(j)
                continue
                
        for idx in sorted(removed_layout, reverse=True):
            del self._raw_layout[idx]
        return
    

    def _ocr_layout(self):
        removed = []
        for idx, (block_type, coordinates, _) in enumerate(self._raw_layout):
            xmin, ymin, xmax, ymax = [int(i) for i in coordinates]
            block_image = self._image[xmin:xmax, ymin:ymax]
            if block_type == 'text' or block_type == 'title' or block_type == 'list':
                ocr_result = self._layout_texts[idx]
                if len(ocr_result) == 0:
                    removed.append(idx)
                # Straighten skewed boxes?
                new_result = []
                for line in ocr_result:
                    d = {'words': []}
                    line_xmin = 10000
                    line_xmax = 0
                    line_ymin = 10000
                    line_ymax = 0
                    for coords, pred in line:
                        xmin = min(coords[1], coords[3], coords[5], coords[7])
                        line_xmin = min(line_xmin, xmin)
                        xmax = max(coords[1], coords[3], coords[5], coords[7])
                        line_xmax = max(line_xmax, xmax)
                        ymin = min(coords[0], coords[2], coords[4], coords[6])
                        line_ymin = min(line_ymin, ymin)
                        ymax = max(coords[0], coords[2], coords[4], coords[6])
                        line_ymax = max(line_ymax, ymax)
                        d['words'].append(((xmin, ymin, xmax, ymax), pred))
                    d['coords'] = (line_xmin, line_ymin, line_xmax, line_ymax)
                    new_result.append(d)
                self._layout_content.append(new_result)
            elif block_type == 'figure':
                self._layout_content.append(block_image)
            elif block_type == 'table':
                for table_info in self._table_structures:
                    if idx == table_info['layout_idx'] and self._page_idx == table_info['image_index']:
                        self._layout_content.append(table_info)
                        if table_info['extracted_value'] == []:
                            removed.append(idx)
                        break

                else:
                    raise ValueError('Table structure not found!')
            else:
                raise ValueError(f'{block_type} layout is not supported, check layout analysis model')
        for idx in sorted(removed, reverse=True):
            del self._raw_layout[idx]
            del self._layout_content[idx]
        return
    
    def _save_image_from_array(self, array, image_path):
        # Ensure the array is in the correct format (uint8)
        if array.dtype != np.uint8:
            array = (255 * array).astype(np.uint8)
        img = Image.fromarray(array)
        img.save(image_path)

    def _add_column_break(self, paragraph):
        """Add a column break to the given paragraph."""
        run = paragraph.add_run()
        br = OxmlElement('w:br')
        br.set(qn('w:type'), 'column')
        run._r.append(br)
        return
    
    def _align_determined(self, content):
        if len(content) == 1:
            mid_point = (content[0]['coords'][1] + content[0]['coords'][3]) / 2
            half_page = self._size[1] / 2
            if abs(mid_point - half_page) < 0.1 * self._size[1]:
                return "Center", False
            elif half_page - mid_point > 0.1 * self._size[1]:
                return 'Left', False
            else:
                return 'Right', False

        first_line_start = content[0]['coords'][1]
        starts = [i['coords'][1] for i in content]
        ends = [i['coords'][3] for i in content][:-1]
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
        import pickle
        with open('content.pkl', 'wb') as f:
            pickle.dump(content, f)
        return "Justify", False

    def _calculate_size_ratio(self):
        size_over_height = []
        t = []
        for idx, (block_type, coordinates, _) in enumerate(self._raw_layout):
            if block_type == 'text' or block_type == 'list':
                content = self._layout_content[idx]
                width = [(content[i]['words'][j][0][3] - content[i]['words'][j][0][1]) / len(content[i]['words'][j][1]) for i in range(len(content)) for j in range(len(content[i]['words'])) if not any_upper(content[i]['words'][j][1])]
                height = [content[i]['words'][j][0][2] - content[i]['words'][j][0][0] for i in range(len(content)) for j in range(len(content[i]['words'])) if not any_upper(content[i]['words'][j][1])]
                if len(height) > 0:
                    mean_height = sum(height) / len(height)
                    mean_width = sum(width) / len(width)
                    font_size =  90 * 12 /(self._size[1] / mean_width)
                    size_over_height.append(font_size / mean_height)
                    t.append(block_type)
        if len(size_over_height) > 0:
            self._size_over_height_ratio = sum(size_over_height) / len(size_over_height)
        else:
            self._size_over_height_ratio = 0.4

    def _create_table(self, content):
        cell_data = [i['relation'] for i in content['extracted_value']]
        cell_box = [i['box'] for i in content['extracted_value']]
        cell_text = [i['text'] for i in content['extracted_value']]
        num_rows = max(end_row for _, end_row, _, _ in cell_data) + 1
        num_columns = max(end_col for _, _, _, end_col in cell_data) + 1
        table = self._doc.add_table(num_rows, num_columns)
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

    def get_doc(self):
        return self._doc