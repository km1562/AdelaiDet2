from detectron2.layers import batched_nms
from shapely.geometry import Polygon
from pyxllib.algo.geo import ComputeIou

def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="ori_annotation_file_list"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    
    Args:
        boxlist (detectron2.structures.Boxes): 
        nms_thresh (float): 
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str): 
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist


def poly_ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="ori_annotation_file_list"):
    """
    Performs polygon non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.top_feat.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    # keep = batched_nms(boxes, scores, labels, nms_thresh)
    keep = ComputeIou.nms_polygon(boxes=boxes, iou=nms_thresh, index=True)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist