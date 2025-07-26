import pymupdf
import numpy as np
import pickle
import unidecode


def sort_and_merge_box(bbs, texts, d_thres):
    """
        bbs: List of (xmin, ymin, xmax, ymax) boxes
        texts: List of words corresponding to bbs
        d_thres: height distance threshold between 2 consecutive lines.
    """
    bbs_clusters = [(b, t) for b, t in zip(bbs, texts)]
    bbs_clusters.sort(key=lambda x: x[0][0])

    # group cluster 1st time
    clusters, y_min, cluster_texts = [], [], []
    for tgt_node, text  in bbs_clusters:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            y_min.append(tgt_node[1])
            continue
        matched = None
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            overlap_y = ((src_node[3] - src_node[1]) + (tgt_node[3] - tgt_node[1])) - (max(src_node[3], tgt_node[3]) - min(src_node[1], tgt_node[1]))
            overlap_x = ((src_node[2] - src_node[0]) + (tgt_node[2] - tgt_node[0])) - (max(src_node[2], tgt_node[2]) - min(src_node[0], tgt_node[0]))
            distance = tgt_node[0] - src_node[2]
            if overlap_y > 0.8*min(src_node[3] - src_node[1], tgt_node[3] - tgt_node[1]) and overlap_x < 0.6*min(src_node[2] - src_node[0], tgt_node[2] - tgt_node[0]):
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            cluster_texts.append([text])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
            cluster_texts[idx].append(text)
    zip_clusters = list(zip(clusters, y_min, cluster_texts))
    zip_clusters.sort(key=lambda x: x[1])

    # break lines
    page_text_lines = []
    page_bb_lines = []
    for bb_cluster in zip_clusters:
        bbs, _, texts = bb_cluster
        text_lines = []
        bb_lines = []
        text_line = []
        bb_line = []
        for bb, text in zip(bbs, texts):
            if len(text_line) == 0:
                text_line.append(text)
                bb_line.append(bb)
            else:
                if bb[0] - bb_line[-1][2] > d_thres:
                    text_lines.append(text_line)
                    bb_lines.append(bb_line)
                    text_line = [text]
                    bb_line = [bb]
                else:
                    text_line.append(text)
                    bb_line.append(bb)
        if len(text_line) != 0:
            text_lines.append(text_line)
            bb_lines.append(bb_line)
        for text_line, bb_line in zip(text_lines, bb_lines):
            bb_line = np.array(bb_line)
            xmin = np.min(bb_line[:, 0])
            ymin = np.min(bb_line[:, 1])
            xmax = np.max(bb_line[:, 2])
            ymax = np.max(bb_line[:, 3])
            page_text_lines.append(' '.join(text_line))
            page_bb_lines.append([xmin, ymin, xmax, ymax])
    return page_text_lines, page_bb_lines


class PDFReconstructPredictor:
    def __init__(self, common_cfg, model_cfg):
        self.pdf_size = (595, 842)

    def predict(self, result):
        with open('result.pkl', 'wb') as f:
            pickle.dump(result, f)

        doc = pymupdf.open()
        for page_index, (p8_bbs, words, image) in enumerate(zip(result['text_detection']['coords'], result['ocr']['raw_words'], result['images'])):
            im_h, im_w = image.shape[:2]
            page = doc.new_page(-1, width=self.pdf_size[0], height=self.pdf_size[1])

            bbs = []
            for p8_bb in p8_bbs:
                xmin, xmax = min(p8_bb[::2]), max(p8_bb[::2])
                ymin, ymax = min(p8_bb[1::2]), max(p8_bb[1::2])
                bbs.append([xmin, ymin, xmax, ymax])
            list_hs = []
            for x1, y1, x2, y2 in bbs:
                list_hs.append(y2 - y1)
            avg_h = np.average(list_hs)
            page_text_lines, page_bb_lines = sort_and_merge_box(bbs, words, 2 * avg_h)

            for text_line, bb_line in zip(page_text_lines, page_bb_lines):
                p = pymupdf.Point(bb_line[0], bb_line[1])
                rc = page.insert_text(p,  # bottom-left of 1st char
                                unidecode.unidecode(text_line),  # the text (honors '\n')
                                fontname = "helv",  # the default font
                                fontsize = 11,  # the default font size
                                rotate = 0,  # also available: 90, 180, 270
                                )
                print(f'write {text_line} to {bb_line}')
        
        return doc


if __name__ == '__main__':
    with open('result.pkl', 'rb') as f:
        result = pickle.load(f)
    predictor = PDFReconstructPredictor({}, {})
    doc = predictor.predict(result)
    doc.save('result.pdf')
    doc.close()
