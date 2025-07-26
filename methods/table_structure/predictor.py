import pdb
import pandas
import xlsxwriter
import cv2
import numpy as np
import onnxruntime as ort
from modules.rt_detr.base_rt_detr import BaseRTDETR
from utils.utils import iou_bbox, poly2box, row_polys, sort_polys


def center(self, poly, axis):
    if axis == 'x':
        return (poly[0] + poly[2] + poly [4] + poly [6]) / 4
    elif axis == 'y':
        return (poly[1] + poly [3] + poly [5] + poly[7]) / 4
        

def is_poly_belong(text_poly, block):
    text_bb = poly2box(text_poly)    
    r1, r2, iou = iou_bbox(text_bb, block)
    return r1 >= 0.5 


class TableStructurePredictor:
    def __init__(self, common_config, model_config):
        self.rt_detr = BaseRTDETR(common_config, model_config)
        self.rt_detr.labels = ['row', 'col', 'span']


    def predict(self, result):
        # predict row, col, span
        for page_index, table_infos in enumerate(result['tables']):
            table_infos['all_boxes'] = []
            table_infos['all_scores'] = []
            table_infos['all_classes'] = []
            if len(table_infos['images']) == 0:
                continue
            for table_index, image in enumerate(table_infos['images']):
                table_infos['all_boxes'].append([])
                table_infos['all_scores'].append([])
                table_infos['all_classes'].append([])
            
            for table_index, image in enumerate(table_infos['images']):
                boxes, scores, class_names = self.rt_detr.predict(image)
                boxes, scores, class_names = self.postprocess(boxes, scores, class_names)
                table_infos['all_boxes'][table_index] = boxes
                table_infos['all_scores'][table_index] = scores  
                table_infos['all_classes'][table_index] = class_names

        # map text to table
        for page_index, table_infos in enumerate(result['tables']):
            page_coords = result['text_detection']['coords'][page_index]
            page_words = result['ocr']['raw_words'][page_index]
            poly2word = {tuple(k): v for k, v in zip(page_coords, page_words)}

            table_infos['text'] = [[] for _ in range(len(table_infos['images']))]
            clusters = self.get_sorted_rows(np.reshape(page_coords, (-1, 8)).tolist())
            page_coords = np.concatenate(clusters, 0).reshape((-1, 4, 2))  # sort from top to bottom, left to right
            
            for coord_index, coords in enumerate(page_coords):  # map each coord to corresponding table
                for table_index in range(len(table_infos['images'])):
                    table_bbox = table_infos['boxes'][table_index]
                    if is_poly_belong(coords, table_bbox):
                        x1, y1, _, _ = table_bbox
                        word = poly2word[tuple(coords.reshape(-1).tolist())]
                        coords[:, 0] -= x1
                        coords[:, 1] -= y1
                        table_infos['text'][table_index].append({'box':coords, 'text': word})
                        break
        
        # form cells
        list_tables = []
        for page_index, table_infos in enumerate(result['tables']):
            page_tables = []
            for table_index in range(len(table_infos['images'])):
                boxes = table_infos['all_boxes'][table_index]
                scores = table_infos['all_scores'][table_index]
                classes = table_infos['all_classes'][table_index]

                ### 5.1
                rows, cols, spans = [], [], []
                for box, label in zip(boxes, classes):
                    if label == 'row':
                        rows.append(box)
                    elif label == 'col':
                        cols.append(box)
                    elif label == 'span':
                        spans.append(box)
                rows = sorted(rows, key=lambda x:x[1]+x[3])
                cols = sorted(cols, key=lambda x:x[0]+x[2])
                cells = self.extract_cells(rows, cols, spans)

                cells = self.texts2cells(table_infos['text'][table_index], cells)
                self.to_excel(cells, 'temp.xlsx')
                table_info = {
                    'name': 'table',
                    'image': table_infos['images'][table_index],
                    'extracted_value': cells,
                    'html_content': self.excel2html('temp.xlsx'),
                    'image_index': page_index,
                    'coordinates': table_infos['boxes'][table_index],
                    'layout_idx': table_infos['index'][table_index]
                }
                page_tables.append(table_info)
            list_tables.append(page_tables)

        result['table_structure'] = list_tables
        return result
    

    def postprocess(self, boxes, scores, class_names):
        boxes_dict = {'rows':[], 'cols':[], 'spans':[]}
        scores_dict = {'rows':[], 'cols':[], 'spans':[]}

        for b, s, c in zip(boxes, scores, class_names):
            if c == 'row': #c == 'table_row':
                boxes_dict['rows'].append(b)
                scores_dict['rows'].append(s)
            elif c == 'col': # c == 'table_column':
                boxes_dict['cols'].append(b)
                scores_dict['cols'].append(s)
            elif c == 'span': #in  ['table projected row header', 'table spanning cell']:
                boxes_dict['spans'].append(b)
                scores_dict['spans'].append(s)

        corrected_boxes, corrected_scores, corrected_class_names = [], [], []
        for key in ['rows', 'cols', 'spans']:
            corrected_boxes += [np.array(b) for b in boxes_dict[key]]
            corrected_scores += [s for s in scores_dict[key]]
            corrected_class_names += [key[:-1] for b in boxes_dict[key]]
        boxes, scores, classes = corrected_boxes, corrected_scores, corrected_class_names

        return boxes, scores, classes
    

    def get_sorted_rows(self, polys):
        poly_clusters = row_polys(polys)
        new_clusters = []
        for poly_cluster in poly_clusters:
            poly_cluster.sort(key=lambda b:b[0])
            new_clusters.append(poly_cluster)
        return new_clusters
    

    def get_span_of_cell(self, cell, spans):
        for span in spans:
            if is_poly_belong(cell, span):
                return span
        return None


    def extract_cells(self, rows, cols, spans):
        cells = []
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                xr1, yr1, xr2, yr2 = row
                xc1, yc1, xc2, yc2 = col
                # cells.append({'box':[xc1, yr1, xc2, yr2], 'relation':[i, i+1, j, j+1]})
                # Now relation of a cell correspond to row and col
                cells.append({'box':[xc1, yr1, xc2, yr2], 'relation':[i, i, j, j]})
            
        ## Replace span cell into cells
        '''
        Idea: Xét 1 cell
            - nếu cell này ko thuộc về span cell nào -> Lấy
            - nếu cell này thuộc về 1 span cell:
                + nếu chưa lấy span cell của cell này 
                    --> Lấy span cell, cho relative của span cell chính là cell đang xét
                + nếu đã lấy spann cell của cell này 
                    --> Tăng relative của span cell lên
        '''
    
        if len(spans) > 0: 
            new_cells = []
            flags = {str(span):False for span in spans}
            for i, cell in enumerate(cells):
                span_of_cell = self.get_span_of_cell(cell['box'], spans)
                if span_of_cell is None:
                    new_cells.append(cell)
                    continue
                if not flags[str(span_of_cell)]:
                    new_cells.append({'box':span_of_cell, 'relation':cell['relation']})
                    flags[str(span_of_cell)] = True
                else:
                    idx = [k for k, cell in enumerate(new_cells) if str(cell['box'])==str(span_of_cell)][0]
                    sr = min(new_cells[idx]['relation'][0], cell['relation'][0])
                    er = max(new_cells[idx]['relation'][1], cell['relation'][1])
                    sc = min(new_cells[idx]['relation'][2], cell['relation'][2])
                    ec = max(new_cells[idx]['relation'][3], cell['relation'][3])
                    new_cells[idx]['relation'] = [sr, er, sc, ec]
        else:
            new_cells = cells
        
        return new_cells


    def texts2cells(self, texts, cells):
        '''
        texts: a list, format of each element is {'box': ..., 'score':..., 'roi':..., 'text':...}
        cells: a list, format of each element is {'box': ..., 'relative': ...}
        '''
        list_cell_texts = [[] for _ in range(len(cells))]
        for text in texts:
            max_r1 = 0
            max_index = None
            for cell_index, cell in enumerate(cells):
                text_bbox = poly2box(text['box'])
                cell_bbox = cell['box']
                if text_bbox[2] < cell_bbox[0] or text_bbox[0] > cell_bbox[2] or text_bbox[3] < cell_bbox[1] or text_bbox[1] > cell_bbox[3]:
                    continue
                r1, r2, iou = iou_bbox(text_bbox, cell['box'])
                if r1 > max_r1:
                    max_index = cell_index
                    max_r1 = r1
            if max_index is not None:
                list_cell_texts[max_index].append(text)

        mask = [0 for i in range(len(texts))]
        for cell_index, cell in enumerate(cells):
            cell_texts = list_cell_texts[cell_index]
            # sort text
            if len(cell_texts) > 0:
                bb2text = {}
                bbs = []
                score = 1.
                for text in cell_texts:
                    bb = text['box'].flatten().tolist()
                    bb2text[tuple(bb)] = text['text']
                    bbs.append(bb)
                    score = min(score, text.get('score', 1.))
                sorted_bbs, _ = sort_polys(bbs)
                sorted_texts = [bb2text[tuple(bb)] for bb in sorted_bbs]
                cell['text'] = ' '.join(sorted_texts)
                cell['score'] = score
            else:
                cell['text'] = ''
                cell['score'] = 1.
                
        return cells


    def cells2excel(self, cells, file_name: str = 'temp.xlsx'):
        import xlsxwriter

        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(file_name)
        worksheet = workbook.add_worksheet()
        # Write cell by cell to excel by relation
        for cell in cells:
            rel = cell['relation']
            # print(cell)
            # if rel[0] == rel[1] - 1 and rel[2] == rel[3] - 1: # Not spanning cell
            if rel[0] == rel[1] and rel[2] == rel[3]: #Not spanning cell
                worksheet.write(rel[0], rel[2], cell['text'])
            else: # If cell is span, do merge
                ## this method include last_row, last_col that different from cells format
                worksheet.merge_range(rel[0], rel[2], rel[1], rel[3], cell['text'])
                # worksheet.merge_range(rel[0], rel[2], rel[1]-1, rel[3]-1, cell['text'])

        workbook.close()
    

    def is_single_cell(self, cell):
        r1, r2, c1, c2 = cell['relation']
        return r1 - r2 == 0 and c1 - c2 == 0


    def get_cells(self, boxes, scores, classes, text_boxes, table_bbox, page_img):
        rows, cols, spans = [], [], []
        for box, label in zip(boxes, classes):
            if label == 'row':
                rows.append(box)
            elif label == 'col':
                cols.append(box)
            elif label == 'span':
                spans.append(box)
        rows = sorted(rows, key=lambda x:x[1]+x[3])
        cols = sorted(cols, key=lambda x:x[0]+x[2])
        cells = self.extract_cells(rows, cols, spans)

        # # debug plot
        # for box, label in zip(boxes, classes):
        #     if label == 'row':
        #         cv2.rectangle(table_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        #     elif label == 'col':
        #         cv2.rectangle(table_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        #     elif label == 'span':
        #         cv2.rectangle(table_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        # cv2.imwrite('test.png', table_img)
        # pdb.set_trace()

        ### 5.2
        cells = self.texts2cells(text_boxes, cells, table_bbox, page_img.shape[:2])

        return cells

    def excel2html(self, excel_path):
        sheet = pandas.read_excel(excel_path)
        html_content = sheet.to_html(index=False, header=False, escape=True)
        html_content = html_content.replace(' ', '').replace('\n', '')
        return html_content
    
    def to_excel(self, cells, out_path):
        # Create an new Excel file and add a worksheet.
        workbook = xlsxwriter.Workbook(out_path)
        worksheet = workbook.add_worksheet()
        # Write cell by cell to excel by relation
        for cell in cells:
            rel = cell['relation']
            # print(cell)
            # if rel[0] == rel[1] - 1 and rel[2] == rel[3] - 1: # Not spanning cell
            if rel[0] == rel[1] and rel[2] == rel[3]: #Not spanning cell
                worksheet.write(rel[0], rel[2], cell['text'])
            else: # If cell is span, do merge
                ## this method include last_row, last_col that different from cells format
                worksheet.merge_range(rel[0], rel[2], rel[1], rel[3], cell['text'])
                # worksheet.merge_range(rel[0], rel[2], rel[1]-1, rel[3]-1, cell['text'])
                
        workbook.close()

