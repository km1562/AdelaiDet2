import torch
from shapely.geometry import Polygon
from pyxllib.algo.geo import ComputeIou

def compute_bezier_iou(bezier_pred, bezier_targets):
    beziers_iou = []
    bezier_nums = len(bezier_pred)
    for i in range(bezier_nums):
        pts1 = Polygon(bezier_pred[i].reshape(-1, 2))
        pts2 = Polygon(bezier_targets[i].reshape(-1, 2))
        bezier_iou = ComputeIou.polygon(pts1, pts2)
        beziers_iou.append(bezier_iou)

    return beziers_iou
