from .converter import Converter

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


class Converters:

    def __init__(self, result=None):
        num_pages = len(result['images'])
        self._converters = []
        for i in range(num_pages):
            converter = Converter(i, 
                                 result['images'][i], 
                                 result['layout']['class_names'][i], 
                                 result['layout']['boxes'][i],
                                 result['layout']['scores'][i],
                                 result['reconstruct'][i]['layout_texts'],
                                 result['table_structure']
                                 )
            self._converters.append(converter)
    
    def reconstruct(self, docx_path='output.docx'):
        list_doc = []
        minx = 100
        maxx = 100
        miny = 100
        maxy = 100
        for idx, converter in enumerate(self._converters):
            converter.reconstruct()
            list_doc.append(converter.get_doc())
            base_margins = converter._margins
            minx = min(minx, base_margins[0])
            maxx = min(maxx, base_margins[1])
            miny = min(miny, base_margins[2])
            maxy = min(maxy, base_margins[3])
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

            for element in sub_doc.element.body:
                merged_document.element.body.append(element)

        merged_document.save(docx_path)
        # for idx, doc in enumerate(list_doc):
        #     if idx == 0:
        #         master = doc
        #         composer = Composer(doc)
        #         sections = master.sections
        #         for section in sections:
        #             section.top_margin = Inches(minx)
        #             section.bottom_margin = Inches(maxx)
        #             section.left_margin = Inches(miny)
        #             section.right_margin = Inches(maxy)
        #     else:
        #         master.add_page_break()
        #         sections = doc.sections
        #         for section in sections:
        #             section.top_margin = Inches(minx)
        #             section.bottom_margin = Inches(maxx)
        #             section.left_margin = Inches(miny)
        #             section.right_margin = Inches(maxy)
        #         composer.append(converter.get_doc())
        # master.save(docx_path)


