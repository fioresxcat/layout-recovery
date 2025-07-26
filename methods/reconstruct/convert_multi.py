import pdb
from typing_extensions import List, Dict
from .convert_single import ConverterSingle

from docx import Document
from docx.shared import Inches

def set_section_margins(section, top, right, bottom, left):
    """
    Set margins of a section. Margins should be provided in inches.
    """
    section.top_margin = Inches(top)
    section.right_margin = Inches(right)
    section.bottom_margin = Inches(bottom)
    section.left_margin = Inches(left)

def combine_documents_with_margins(documents):
    master_doc = Document()

    for i, doc in enumerate(documents):
        section = doc.sections[0]
        margins = (section.top_margin.inches, section.right_margin.inches,
                   section.bottom_margin.inches, section.left_margin.inches)

        for element in doc.element.body:
            # Copy paragraphs and tables by checking their type
            if element.tag.endswith('tbl'):  # It's a table
                # Add the entire table as is
                tbl = master_doc._element.body.append(element)
            elif element.tag.endswith('p'):  # It's a paragraph
                # Create a new paragraph and preserve style
                new_para = master_doc.add_paragraph()
                new_para._element.append(element)
        
        # Set margins for new section
        if i < len(documents) - 1:
            new_section = master_doc.add_section()
            set_section_margins(new_section, *margins)

    # Set margins for the first section of the combined document
    set_section_margins(master_doc.sections[0], *margins)

    return master_doc


class ConverterMulti:
    def __init__(self, result=None):
        num_pages = len(result['images'])
        self._converters: List[ConverterSingle] = []
        for i in range(num_pages):
            converter = ConverterSingle(
                result['request_id'],
                i, 
                result['images'][i], 
                result['layout']['class_names'][i], 
                result['layout']['boxes'][i],
                result['layout']['scores'][i],
                result['reconstruct'][i]['layout_texts'],
                result['table_structure'][i]
            )
            self._converters.append(converter)
    
    
    def reconstruct(self) -> Document:
        list_doc = []
        minx = 100
        maxx = 100
        miny = 100
        maxy = 100
        for idx, converter in enumerate(self._converters):
            converter.reconstruct()
            list_doc.append(converter.get_doc())
            base_margins = converter.margins
            minx = min(minx, base_margins['left'])
            maxx = min(maxx, base_margins['right'])
            miny = min(miny, base_margins['top'])
            maxy = min(maxy, base_margins['bottom'])

        merged_document = Document()
        merged_document.styles['Normal'].font.name = 'Times New Roman'
        sections = merged_document.sections
        for section in sections:
            section.top_margin = Inches(minx)
            section.bottom_margin = Inches(maxx)
            section.left_margin = Inches(miny)
            section.right_margin = Inches(maxy)
        for index, file in enumerate(list_doc):
            sub_doc = file
            sections = sub_doc.sections
            for section in sections:
                section.top_margin = Inches(minx)
                section.bottom_margin = Inches(maxx)
                section.left_margin = Inches(miny)
                section.right_margin = Inches(maxy)
            # Don't add a page break if you've reached the last file.
            if index < len(list_doc)-1:
                sub_doc.add_page_break()

            for i, element in enumerate(sub_doc.element.body):
                merged_document.element.body.append(element)

        return merged_document



    # def reconstruct(self) -> Document:
    #     from docx.oxml import OxmlElement
    #     from docx.oxml.ns import qn
    #     from docx.enum.section import WD_SECTION
    #     import io

    #     def copy_paragraph_with_images(src_para, dest_doc):
    #         dest_para = dest_doc.add_paragraph()
    #         for run in src_para.runs:
    #             if 'graphic' in run._element.xml:
    #                 for drawing in run._element.findall('.//w:drawing', run._element.nsmap):
    #                     for blip in drawing.findall('.//a:blip', run._element.nsmap):
    #                         rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
    #                         image_part = run.part.related_parts[rId]
    #                         image_bytes = image_part.blob
    #                         dest_para.add_run().add_picture(io.BytesIO(image_bytes))
    #             else:
    #                 new_run = dest_para.add_run(run.text)
    #                 new_run.bold = run.bold
    #                 new_run.italic = run.italic
    #                 new_run.underline = run.underline
    #                 new_run.font.size = run.font.size
    #                 new_run.font.name = run.font.name
    #         dest_para.style = src_para.style
    #         dest_para.alignment = src_para.alignment
    #         if src_para.paragraph_format is not None:
    #             pf = src_para.paragraph_format
    #             dest_para.paragraph_format.left_indent = pf.left_indent
    #             dest_para.paragraph_format.right_indent = pf.right_indent
    #             dest_para.paragraph_format.first_line_indent = pf.first_line_indent
    #             dest_para.paragraph_format.space_before = pf.space_before
    #             dest_para.paragraph_format.space_after = pf.space_after
    #             dest_para.paragraph_format.line_spacing = pf.line_spacing
    #             dest_para.paragraph_format.line_spacing_rule = pf.line_spacing_rule
    #         return dest_para

    #     list_doc = []
    #     minx = 100
    #     maxx = 100
    #     miny = 100
    #     maxy = 100
    #     for idx, converter in enumerate(self._converters):
    #         converter.reconstruct()
    #         list_doc.append(converter.get_doc())
    #         base_margins = converter.margins
    #         minx = min(minx, base_margins['left'])
    #         maxx = min(maxx, base_margins['right'])
    #         miny = min(miny, base_margins['top'])
    #         maxy = min(maxy, base_margins['bottom'])

    #     merged_document = Document()
    #     merged_document.styles['Normal'].font.name = 'Times New Roman'
    #     sections = merged_document.sections
    #     for section in sections:
    #         section.top_margin = Inches(minx)
    #         section.bottom_margin = Inches(maxx)
    #         section.left_margin = Inches(miny)
    #         section.right_margin = Inches(maxy)

    #     for doc_idx, sub_doc in enumerate(list_doc):
    #         for sec_idx, section in enumerate(sub_doc.sections):
    #             # For the first section of the first doc, use the default section
    #             if doc_idx == 0 and sec_idx == 0:
    #                 dest_section = merged_document.sections[0]
    #             else:
    #                 dest_section = merged_document.add_section(WD_SECTION.CONTINUOUS)
    #             # Copy margins
    #             dest_section.top_margin = section.top_margin
    #             dest_section.bottom_margin = section.bottom_margin
    #             dest_section.left_margin = section.left_margin
    #             dest_section.right_margin = section.right_margin
    #             # Copy column settings
    #             sectPr = dest_section._sectPr
    #             # Remove existing cols element if it exists
    #             for child in sectPr:
    #                 if child.tag.endswith('cols'):
    #                     sectPr.remove(child)
    #                     break
    #             # Find the cols element in the source section
    #             src_sectPr = section._sectPr
    #             src_cols = None
    #             for child in src_sectPr:
    #                 if child.tag.endswith('cols'):
    #                     src_cols = child
    #                     break
    #             if src_cols is not None:
    #                 # Deep copy the cols element
    #                 import copy
    #                 sectPr.append(copy.deepcopy(src_cols))

    #             # Copy content belonging to this section
    #             # Find the start and end element indices for this section
    #             body_elements = list(sub_doc.element.body)
    #             # Find the indices of section breaks
    #             sect_break_indices = [i for i, el in enumerate(body_elements) if el.tag.endswith('sectPr')]
    #             # Determine the range for this section
    #             if not sect_break_indices or sec_idx == 0 or (sec_idx - 1) >= len(sect_break_indices):
    #                 start_idx = 0
    #             else:
    #                 start_idx = sect_break_indices[sec_idx - 1] + 1
    #             if sec_idx < len(sect_break_indices):
    #                 end_idx = sect_break_indices[sec_idx]
    #             else:
    #                 end_idx = len(body_elements)
    #             # Copy paragraphs, tables, and images in this section
    #             for el in body_elements[start_idx:end_idx]:
    #                 if el.tag.endswith('tbl'):
    #                     # Table: add as is
    #                     merged_document._element.body.append(el)
    #                 elif el.tag.endswith('p'):
    #                     # Paragraph: copy with images
    #                     # Find the corresponding paragraph object
    #                     for para in sub_doc.paragraphs:
    #                         if para._element == el:
    #                             copy_paragraph_with_images(para, merged_document)
    #                             break
    #         # Add a page break after each doc except the last
    #         if doc_idx < len(list_doc) - 1:
    #             merged_document.add_page_break()

    #     return merged_document