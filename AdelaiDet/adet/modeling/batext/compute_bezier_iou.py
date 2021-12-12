import torch
from shapely.geometry import Polygon
from pyxllib.algo.geo import ComputeIou
import numpy as np

def bezier_to_poly(bezier):
    # bezier to polygon
    u = np.linspace(0, 1, 20)
    bezier = np.array(bezier)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
             + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
             + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
             + np.outer(u ** 3, bezier[:, 3])
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=None)
    points = points.reshape(-1).tolist()
    return points

def compute_bezier_iou(bezier_pred, bezier_targets):
    beziers_iou = []
    bezier_nums = len(bezier_pred)
    for i in range(bezier_nums):
        pts1 = bezier_to_poly(bezier_pred[i].cpu())
        pts2 = bezier_to_poly(bezier_targets[i].cpu())
        pts1 = Polygon(torch.from_numpy(pts1))
        pts2 = Polygon(torch.from_numpy(pts2))
        bezier_iou = ComputeIou.polygon(pts1, pts2)
        beziers_iou.append(bezier_iou)

    return beziers_iou