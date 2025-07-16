import os
import cv2
import pdb
import numpy as np
from omegaconf import OmegaConf
import argparse
from typing import Any
from pathlib import Path
from docx import Document
from fpdf import FPDF

from methods.layout.predictor import LayoutPredictor
from methods.text_detection.predictor import TextDetectPredictor
from methods.ocr.predictor import OCRPredictor
from methods.table_structure.predictor import TableStructurePredictor
from methods.reconstruct.predictor import ReconstructPredictor

class Processor:
    def __init__(self, common_cfg, model_cfg):
        triton_model_cfg = model_cfg['triton_models']
        triton_server_cfg = common_cfg['triton_server']
        self.modules = [
            LayoutPredictor(triton_server_cfg, triton_model_cfg['tvpl_layout_detection'], model_type='surya'),
            TextDetectPredictor(triton_server_cfg, triton_model_cfg['capex_text_detection']),
            OCRPredictor(triton_server_cfg, triton_model_cfg['printing_ocr']),
            TableStructurePredictor(triton_server_cfg, triton_model_cfg['general_table_structure']),
            ReconstructPredictor(triton_server_cfg, model_cfg)
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
    from utils.utils import load_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', type=str, required=True)
    args = parser.parse_args()

    common_cfg = load_yaml('configs/common.yaml')
    model_cfg = load_yaml('configs/model.yaml')
    processor = Processor(common_cfg, model_cfg)

    if os.path.isdir(args.inp_path):
        ipaths = sorted([fp for fp in list(Path(args.inp_path).glob('*'))])
        ipaths = ipaths[:1]
        images = [cv2.imread(str(ipath)) for ipath in ipaths]
        result = processor.predict(request_id=ipaths[0].name, images=images)
    else:
        images = [cv2.imread(args.inp_path)]
        result = processor.predict(request_id=Path(args.inp_path).name, images=images)

    doc: Document = result['final_doc']
    doc.save('output.docx')
    
if __name__ == '__main__':
    main()
