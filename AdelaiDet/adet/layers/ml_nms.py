from detectron2.layers import batched_nms
from shapely.geometry import Polygon
from pyxllib.algo.geo import ComputeIou
import torch

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

    # sorted(boxlist, key=lambda boxlis: -boxlist.scores)

    # keep = batched_nms(boxes, scores, labels, nms_thresh)
    boxes = boxlist.top_feat
    scores = boxlist.scores
    labels = boxlist.pred_classes

    keep = ComputeIou.nms_polygon(boxes=boxes, iou=nms_thresh, index=True)
    if max_proposals > 0:
        keep = keep[: max_proposals]

    return boxlist[keep]

# def sort_boxlist(instance_lists):
#     image_size = instance_lists[0].image_size
#     ret = Instances(image_size)
#
#     key = instance_lists[0]._fiels.keys()
#     scores = instance_lists[key["scores"]]
#     scores, indics = scores.sort()
#     for k in instance_lists[0]._fiels.keys():
#         if k != "scores":
#             values = [i.get(k) for i in instance_lists]
