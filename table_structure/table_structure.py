import pdb
import pandas
import xlsxwriter
import cv2
import numpy as np
import onnxruntime as ort


class TableStructure:
    instance = None
    def __init__(self, common_config, model_config):
        self.labels = ['table', 'table_column', 'table_row', 'table column header', 'table projected row header', 'table spanning cell', 'no object']
        self.no_object_index = len(self.labels)-1
        self.MEAN, self.STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.common_config = common_config
        self.model_config = model_config
        self.session = ort.InferenceSession(self.model_config['model_path'], providers=['CUDAExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:]
                
        
    @staticmethod
    def get_instance(common_config, model_config):
        if TableStructure.instance is None:
            TableStructure.instance = TableStructure(common_config, model_config)
        return TableStructure.instance
    
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        current_max_size = max(w, h)
        scale = 1000 / current_max_size
        image = cv2.resize(image, (int(round(scale*w)), int(round(scale*h))))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # comment this line if image is already rgb ???

        im = image / 255.0
        im = (im - self.MEAN) / self.STD
        im = np.expand_dims(im, 0).transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im, dtype='float32')  # contiguous

        return im
    

    def is_poly_belong(self, text, block):
        xmin_block, ymin_block, xmax_block, ymax_block = block
        xmin_text, xmax_text = min(text[:, 0]), max(text[:, 0])
        ymin_text, ymax_text = min(text[:, 1]), max(text[:, 1])
        x_center = (xmin_text + xmax_text) / 2
        y_center = (ymin_text + ymax_text) / 2
        if x_center > xmin_block and x_center < xmax_block and y_center > ymin_block and y_center < ymax_block:
            return True
        return False
    
    
    def postprocess(self, boxes, scores, class_names):
        ## postprocess name
        corrected_boxes, corrected_scores, corrected_class_names = [], [], []
        for b, s, c in zip(boxes, scores, class_names):
            # has_text = False
            # for poly in text_boxes:
            #     if self.is_poly_belong(poly, b):
            #         has_text = True
            #         break
            has_text = True
            if has_text:
                if c == 'table': 
                    continue
                elif 'column' in c:
                    corrected_class_names.append('col')
                elif 'row' in c:
                    corrected_class_names.append('row')
                else:
                    corrected_class_names.append('span')
                corrected_boxes.append(b)
                corrected_scores.append(s)
        
        return corrected_boxes, corrected_scores, corrected_class_names
    

    def remove_overlapping_boxes(self, bboxes, scores, class_names, threshold=0.4):
        # Sort boxes by descending order of scores
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        sorted_bboxes = [bboxes[i] for i in sorted_indices]
        sorted_class_names = [class_names[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # List to store non-overlapping boxes
        non_overlapping_boxes = []
        non_overlapping_scores = []
        non_overlapping_class_names = []
        
        # Iterate through sorted boxes
        for bbox, score, class_name in zip(sorted_bboxes, sorted_scores, sorted_class_names):
            # Check for overlap with previously selected non-overlapping boxes
            if all(self.iou(bbox, prev_bbox) <= threshold for prev_bbox in non_overlapping_boxes):
                non_overlapping_boxes.append(bbox)
                non_overlapping_scores.append(score)
                non_overlapping_class_names.append(class_name)
        
        return non_overlapping_boxes, non_overlapping_scores, non_overlapping_class_names

    
    def iou(self, bbox1, bbox2):
        # Calculate the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        # Calculate intersection area
        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        # Calculate areas of each bounding box
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate IOU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

        return iou
    

    def row_col_detect(self, result):
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
                h, w = image.shape[:2]
                processed_img = self.preprocess(image)
                detections = self.session.run(None, {self.input_name: processed_img})[0][0]
                # pdb.set_trace()
                # detections = np.squeeze(detections, axis=0)
                boxes, scores, class_names = [], [], []
                for detection in detections:
                    probs = detection[4:]
                    index = np.argmax(probs)
                    score = probs[index]
                    if score >= self.model_config['conf_threshold']:
                        if index != self.no_object_index:
                            x1 = (detection[0] - detection[2]/2)*w
                            y1 = (detection[1] - detection[3]/2)*h
                            x2 = (detection[0] + detection[2]/2)*w
                            y2 = (detection[1] + detection[3]/2)*h
                            boxes.append([x1, y1, x2, y2])
                            scores.append(score)
                            #class_ids.append(index)
                            class_names.append(self.labels[index])
                
                # remove overlap boxes
                boxes, scores, class_names = self.remove_overlapping_boxes(boxes, scores, class_names, threshold=0.4)
                # text_boxes = table_infos['text_boxes'][table_index]
                corrected_boxes, corrected_scores, corrected_class_names = self.postprocess(boxes, scores, class_names)
                table_infos['all_boxes'][table_index] = corrected_boxes
                table_infos['all_scores'][table_index] = corrected_scores  
                table_infos['all_classes'][table_index] = corrected_class_names
                
        return result


    def bb_iou_rect_mode(self, boxA, boxB):
        xminA, yminA, xmaxA, ymaxA = boxA
        xminB, yminB, xmaxB, ymaxB = boxB
        
        xA = max(xminA, xminB)
        yA = max(yminA, yminB)
        xB = min(xmaxA, xmaxB)
        yB = min(ymaxA, ymaxB)
        
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (xmaxA - xminA + 1) * (ymaxA - yminA + 1)
        boxBArea = (xmaxB - xminB + 1) * (ymaxB - yminB + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / boxAArea
        return iou
    

    def is_belong(self, text, block):
        return self.bb_iou_rect_mode(text, block) > 0.5
    

    def get_span_of_cell(self, cell, spans):
        for span in spans:
            if self.is_belong(cell, span):
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
        mask = [0 for i in range(len(texts))]
        for cell in cells:
            cell_texts = [] 
            for i, text in enumerate(texts):
                if mask[i] == 1: continue
                if self.is_poly_belong(text['box'], cell['box']):
                    cell_texts.append(text['text'])
                    mask[i] = 1
            cell['text'] = ' '.join(cell_texts)
            # cell.pop('box', None)
        return cells
    

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


    def excel2html(self, excel_path):
        sheet = pandas.read_excel(excel_path)
        html_content = sheet.to_html(index=False, header=False, escape=True)
        html_content = html_content.replace(' ', '').replace('\n', '')
        return html_content



    def center(self, poly, axis):
        if axis == 'x':
            return (poly[0] + poly[2] + poly [4] + poly [6]) / 4
        elif axis == 'y':
            return (poly[1] + poly [3] + poly [5] + poly[7]) / 4


    def row_polys(self, polys):
        polys.sort(key=lambda x: self.center(x, 'x'))
        clusters, mean_min, mean_max = [], [], []
        for bb in polys:
            if len(clusters) == 0:
                clusters.append([bb])
                mean_min.append(bb[1])
                mean_max.append(bb[7])
                continue
            mid_y = self.center(bb, 'y')
            mid_x = self.center(bb, 'x')
            matched = None
            for idx, clt in enumerate(clusters):
                last_mid_x = self.center(clt[-1], 'x')
                if mean_min[idx] <= mid_y <= mean_max[idx]:
                    x_dist = mid_x - last_mid_x
                    if matched is None or x_dist < matched[1]:
                        matched = (idx, x_dist)
            if matched is None:
                clusters.append([bb])
                mean_min.append(bb[1])
                mean_max.append(bb[7])
            else:
                idx = matched[0]
                clusters[idx].append(bb)
                mean_min[idx] = (mean_min[idx] + bb[1]) / 2
                mean_max[idx] = (mean_max[idx] + bb[7]) / 2
        zip_clusters = list(zip(clusters, mean_min))
        zip_clusters.sort(key=lambda x: x[1])
        if zip_clusters == []:
            return []
        zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    
        return zip_clusters


    def sort_polys(self, polys):
        poly_clusters = self.row_polys(polys)
        new_clusters = []
        for poly_cluster in poly_clusters:
            poly_cluster.sort(key=lambda b:b[0])
            new_clusters.append(poly_cluster)
        return new_clusters
    

    def detect_table_postprocess(self, result):
        final_result = []

        # map text to table
        for page_index, table_infos in enumerate(result['tables']):
            page_coords = result['text_detection']['coords'][page_index]
            page_words = result['ocr']['raw_words'][page_index]
            poly2word = {tuple(k): v for k, v in zip(page_coords, page_words)}

            table_infos['text'] = [[] for _ in range(len(table_infos['images']))]
            clusters = self.sort_polys(np.reshape(page_coords, (-1, 8)).tolist())
            page_coords = np.concatenate(clusters, 0).reshape((-1, 4, 2))  # sort from top to bottom, left to right
            
            for coord_index, coords in enumerate(page_coords):  # map each coord to corresponding table
                for table_index in range(len(table_infos['images'])):
                    table_bbox = table_infos['boxes'][table_index]
                    if self.is_poly_belong(coords, table_bbox):
                        x1, y1, _, _ = table_bbox
                        word = poly2word[tuple(coords.reshape(-1).tolist())]
                        coords[:, 0] -= x1
                        coords[:, 1] -= y1
                        table_infos['text'][table_index].append({'box':coords, 'text': word})
                        break

        for page_index, table_infos in enumerate(result['tables']):
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
                # rows = refine_rows(rows)
                # cols = refine_cols(cols)
                cells = self.extract_cells(rows, cols, spans)

                ### 5.2
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
                final_result.append(table_info)

        result['table_structure'] = final_result
        return result
    

    def predict(self, result):
        result = self.row_col_detect(result)
        result = self.detect_table_postprocess(result)
        return result
