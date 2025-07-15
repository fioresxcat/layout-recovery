from docx import Document
from docx import shared
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from ultralytics import YOLO
import pdb

def main():
    doc = Document()
    # get page width and page height of doc
    for section in doc.sections:
        print(section.page_width)  # Inches(8.5)
        print(section.page_height)  # Inches(11)

    section = doc.add_section(WD_SECTION.CONTINUOUS)
    # set w and h for section
    section.page_width = shared.Inches(2)
    section.page_height = shared.Inches(2)
    # add text to the section
    paragraph = doc.add_paragraph()
    paragraph_format = paragraph.paragraph_format
    text_run = paragraph.add_run(' '.join(['hello world ']*100) + ' ')
    text_run.font.size = shared.Pt(10)
    # set margin for the section
    section.left_margin = shared.Inches(4)
    section.right_margin = shared.Inches(2)
    section.top_margin = shared.Inches(4)
    section.bottom_margin = shared.Inches(2.5)
    # save
    doc.save('test.docx')

def nothing():
    model = YOLO('models/layout_detection/best.pt')
    print(model.names)
    pdb.set_trace()


if __name__ == '__main__':
    # main()
    nothing()