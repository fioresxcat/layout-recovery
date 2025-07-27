from dataclasses import dataclass
from typing_extensions import List, Dict, Tuple, Any, Optional
import numpy as np

@dataclass
class Block:
    type: str
    bbox: tuple[int, int, int, int]
    score: float
    image: np.ndarray


@dataclass
class Line:
    words: List[Dict]
    segments: List[Dict]
    bbox: tuple[int, int, int, int]

@dataclass
class TextBlock(Block):
    lines: List[Line]

@dataclass
class TableBlock(Block):
    cells: List[Dict]


def boxes_overlap(box1, box2):
    """
    Check if two boxes overlap.
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Check for overlap
    return not (x1_max <= x2_min or x1_min >= x2_max or
                y1_max <= y2_min or y1_min >= y2_max)

def cut_off_box(box1, box2):
    """
    Adjust box2 dimensions to avoid overlap with box1.
    Assumes box1 is fixed and box2 should be reduced to avoid overlap.
    Args:
        box1: The coordinates of the first box as (xmin, ymin, xmax, ymax).
        box2: The coordinates of the second box as (xmin, ymin, xmax, ymax).
    Returns:
        Modified box2 coordinates.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # If box2 is to the right of box1
    if x2_min < x1_max < x2_max:
        x2_min = x1_max

    # If box2 is to the left of box1
    if x1_min < x2_max < x1_max:
        x2_max = x1_min

    # If box2 is below box1
    if y2_min < y1_max < y2_max:
        y2_min = y1_max

    # If box2 is above box1
    if y1_min < y2_max < y1_max:
        y2_max = y1_min

    # Ensure the resulting box still makes sense
    if x2_min >= x2_max or y2_min >= y2_max:
        return None

    return (x2_min, y2_min, x2_max, y2_max)

def any_upper(text):
    return any(c.isupper() or c.isnumeric() for c in text)

def horizontal_align(box1, box2):
    box1 = box1[1]
    box2 = box2[1]
    L1 = box1[3] - box1[1]
    L2 = box2[3] - box2[1]
    L = max(box1[3], box2[3]) - min(box1[1], box2[1])
    return L1 + L2 > L


# def vertical_align(box1, box2):
#     box1 = box1[1]
#     box2 = box2[1]
#     L1 = box1[2] - box1[0]
#     L2 = box2[2] - box2[0]
#     L = max(box1[2], box2[2]) - min(box1[0], box2[0])
    
#     return L1 + L2 > L

def vertical_align(box1, box2):
    box1 = box1[1]
    box2 = box2[1]
    # box: (x1, y1, x2, y2)
    x1_min, x1_max = box1[0], box1[2]
    x2_min, x2_max = box2[0], box2[2]
    box1_w = x1_max - x1_min
    box2_w = x2_max - x2_min
    union_min = min(x1_min, x2_min)
    union_max = max(x1_max, x2_max)
    union_w = union_max - union_min
    # intersection
    inter_min = max(x1_min, x2_min)
    inter_max = min(x1_max, x2_max)
    inter_w = max(0, inter_max - inter_min)
    r1 = box1_w / union_w if union_w > 0 else 0
    r2 = box2_w / union_w if union_w > 0 else 0
    iou = inter_w / union_w if union_w > 0 else 0

    return r1 > 0.7 or r2 > 0.7 or iou > 0.5


from collections import deque

def _graph_bfs_from_node(graph, start):
    '''Breadth First Search connected graph with start node.
    
    Args:
        graph (list): GRAPH represented by adjacent list, [set(1,2,3), set(...), ...].
        start (int): Index of any start vertex.
    '''
    search_queue = deque()    
    searched = set()

    search_queue.append(start)
    while search_queue:
        cur_node = search_queue.popleft()
        if cur_node in searched: continue
        yield cur_node
        searched.add(cur_node)
        for node in graph[cur_node]:
            search_queue.append(node)

def graph_bfs(graph): 
    '''Breadth First Search graph (may be disconnected graph).
    
    Args:
        graph (list): GRAPH represented by adjacent list, [set(1,2,3), set(...), ...]
    
    Returns:
        list: A list of connected components
    '''
    # search graph
    # NOTE: generally a disconnected graph
    counted_indexes = set() # type: set[int]
    groups = []
    for i in range(len(graph)):
        if i in counted_indexes: continue
        # connected component starts...
        indexes = set(_graph_bfs_from_node(graph, i))
        groups.append(indexes)
        counted_indexes.update(indexes)
    
    return groups

def group(layout_blocks, criterion):
    num_block = len(layout_blocks)
    index_groups = [set() for i in range(num_block)]
    for i, block in enumerate(layout_blocks):
        for j in range(i+1, num_block):
            if criterion(block, layout_blocks[j]):
                index_groups[i].add(j)
                index_groups[j].add(i)
    groups = graph_bfs(index_groups)
    
    return groups