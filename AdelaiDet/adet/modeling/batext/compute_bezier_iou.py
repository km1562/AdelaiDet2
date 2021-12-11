from pyxllib.algo.geo import ComputeIou

def compute_bezier_iou(bezier_pred, bezier_targets):
    beziers_iou = []
    bezier_nums = len(bezier_pred)
    for i in range(bezier_nums):
        bezier_iou = ComputeIou.polygon2(bezier_pred[i], bezier_targets[i])
        beziers_iou.append(bezier_iou)

    return beziers_iou
