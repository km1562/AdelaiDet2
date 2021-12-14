import torch
from shapely.geometry import Polygon
from pyxllib.algo.geo import ComputeIou
import numpy as np

# def bezier_to_poly(bezier):
#     # bezier to polygon
#     u = np.linspace(0, 1, 20)
#     bezier = np.array(bezier)
#     bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
#     points = np.outer((1 - u) ** 3, bezier[:, 0]) \
#              + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
#              + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
#              + np.outer(u ** 3, bezier[:, 3])
#     points = np.concatenate((points[:, :2], points[:, 2:]), axis=None)
#     points = points.reshape(-1).tolist()
#     return points

def bezier_to_poly(bezier):
    # bezier to polygon
    u = torch.linspace(0, 1, 20, device='cuda')
    bezier = bezier.reshape(2, 4, 2).permute(0, 2, 1).reshape(4, 4)
    points = torch.outer((1 - u) ** 3, bezier[:, 0]) \
             + torch.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
             + torch.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
             + torch.outer(u ** 3, bezier[:, 3])
    points = torch.cat((points[:, :2], points[:, 2:]), dim=0)
    # points = points.reshape(-1)
    return points

def compute_bezier_iou(bezier_pred, bezier_targets, ctrness_targets):
    beziers_iou = []
    bezier_nums = len(bezier_pred)
    iou_weight = []
    for i in range(bezier_nums):
        pts1 = bezier_to_poly(bezier_pred[i])
        pts2 = bezier_to_poly(bezier_targets[i])
        try:
            pts1 = Polygon(pts1)
            pts2 = Polygon(pts2)
            bezier_iou = ComputeIou.polygon(pts1, pts2)
        except:
            continue

        if bezier_iou == 0:
            bezier_iou = 0.05

        beziers_iou.append(bezier_iou)
        iou_weight.append(ctrness_targets[i])

    # print("beziers_iou's value\n", beziers_iou)
    return torch.tensor(beziers_iou, device='cuda'), torch.tensor(iou_weight, device='cuda')