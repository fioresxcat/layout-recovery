import os
import cv2
import pdb
import numpy as np
from omegaconf import OmegaConf
import argparse
from typing import Any
from pathlib import Path
import docx
import pymupdf
from pdf2docx import Converter


from methods.layout.predictor import LayoutPredictor
from methods.text_detection.predictor import TextDetectPredictor
from methods.ocr.predictor import OCRPredictor
from methods.table_structure.predictor import TableStructurePredictor
from methods.reconstruct.predictor import ReconstructPredictor
from methods.reconstruct_pdf.predictor import PDFReconstructPredictor

class Processor:
    def __init__(self, common_cfg, model_cfg):
        triton_model_cfg = model_cfg['triton_models']
        triton_server_cfg = common_cfg['triton_server']
        self.modules = [
            TextDetectPredictor(triton_server_cfg, triton_model_cfg['capex_text_detection']),
            OCRPredictor(triton_server_cfg, triton_model_cfg['printing_ocr']),
            LayoutPredictor(triton_server_cfg, triton_model_cfg['tvpl_layout_detection'], model_type='surya'),
            TableStructurePredictor(triton_server_cfg, triton_model_cfg['general_table_structure']),
            # ReconstructPredictor(triton_server_cfg, model_cfg),
            PDFReconstructPredictor(triton_server_cfg, model_cfg)
        ]

    def predict(self, request_id, images):
        result = {
            'images': images,
            'request_id': request_id
        }
        for module in self.modules:
            print(f'running {module.__class__.__name__}')
            result = module.predict(result)
            print(result.keys())
        return result


def main():
    from utils.utils import load_yaml, convert_docx_to_pdf

    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=False, default=None)
    args = parser.parse_args()

    common_cfg = load_yaml('configs/common.yaml')
    model_cfg = load_yaml('configs/model.yaml')
    processor = Processor(common_cfg, model_cfg)

    for item in os.listdir(args.inp_path):
        if args.file_name is not None and args.file_name not in item:
            continue
        print(f'------------- Processing {item} -------------')
        item_path = os.path.join(args.inp_path, item)
        if os.path.isdir(item_path):
            ipaths = sorted([fp for fp in list(Path(item_path).glob('*'))])
            ipaths = ipaths[:1]
            images = [cv2.imread(str(ipath)) for ipath in ipaths]
            result = processor.predict(request_id=ipaths[0].name, images=images)
        else:
            images = [cv2.imread(item_path)]
            result = processor.predict(request_id=Path(item_path).name, images=images)

        doc = result['final_doc']
        # try:
        #     # doc.save(os.path.join('output', item + '.docx'))
        #     doc.save('output.docx')
        # except:

        # doc.ez_save(os.path.join('output', item + '.pdf'))
        doc.ez_save('output.pdf')
        pdf_file = 'output.pdf'
        docx_file = 'output-pdf2docx-2.docx'

        # convert pdf to docx
        cv = Converter(pdf_file)
        cv.convert(docx_file)      # all pages by default
        cv.close()

        print(f'done {item_path}')

if __name__ == '__main__':
    main()
