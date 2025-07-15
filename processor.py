from typing import Any
from layout.yolo.yolo_layout_detection import YOLOLayoutDetector
from text_detection import TextDetector
from ocr import OCR
from table_structure import TableStructure
from recovery.reconstruct import Reconstructor
from pathlib import Path
import cv2


class Processor:
    def __init__(self, common_cfg, model_cfg):
        self.common_cfg = common_cfg
        self.model_cfg = model_cfg
        self.modules = [
            YOLOLayoutDetector(common_cfg, model_cfg['layout_detection']),
            TextDetector(common_cfg, model_cfg['text_detection']),
            OCR(common_cfg, model_cfg['ocr']),
            TableStructure(common_cfg, model_cfg['table_structure']),
            Reconstructor(common_cfg, model_cfg['reconstruct'])
        ]

    def predict(self, img_fps):
        images = []
        for img_fp in img_fps:
            image = cv2.imread(str(img_fp))
            images.append(image)
        result = {
            'images': images,
            'request_id': Path(img_fp).stem
        }
        for module in self.modules:
            print(f'running {module.__class__.__name__}')
            result = module.predict(result)
            print(result.keys())
        return result
        

def main():
    import cv2
    import omegaconf
    import pdb
    import os
    import glob
    import tqdm
    import random
    import sys

    common_cfg = omegaconf.OmegaConf.load('configs/common.yaml')
    model_cfg = omegaconf.OmegaConf.load('configs/model.yaml')
    processor = Processor(common_cfg, model_cfg)

    doc_im_dir = sys.argv[1]
    img_fps = [str(fp) for fp in list(Path(doc_im_dir).glob('*'))]
    result = processor.predict(img_fps)

    
if __name__ == '__main__':
    main()
