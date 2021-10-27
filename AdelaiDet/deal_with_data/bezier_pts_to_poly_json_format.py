"""
思路：
输入一个json，
把他的字段改成poly，
再集成成segementation字段，
然后输出一个json字段


"""

import itertools
import json

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

def text_bezier_pts_to_segementation(source_path, output):
    """
    :param source_path:
    :param output:
    :return:
    取出参数，
    然后生成segmentation，
    添加进去，
    """
    with open(source_path, "r") as f:
        data = json.load(f)
        annotations = data["annotations"]
        # data["annotations"] = annotations
        # del data["[annotations]"]
        for i in range(len(annotations)):
            segementation = []
            bezier_pts = annotations[i]["bezier_pts"]
            poly_points = bezier_to_poly(bezier_pts)
            segementation.append(poly_points)
            annotations[i]["segmentation"] = segementation

        data["annotations"] = annotations

    with open(output, "w") as f:
        json.dump(data, f)

"""
对text_json，里面的bezierr转成segmengtation字段

"""
# source_path = "/home/wengkangming/map_file/AdelaiDet/datasets/totaltext/test.json"
# output = "/home/wengkangming/map_file/AdelaiDet/datasets/totaltext/text_cocoformat.json"

# source_path = "/home/datasets/textGroup/syntext/syntext1/copy_train.json"
# output = "/home/datasets/textGroup/syntext/syntext1/train_add_seg.json"

# source_path = "/home/wengkangming/map_file/AdelaiDet2/AdelaiDet/datasets/totaltext/test.json"
# output = "/home/wengkangming/map_file/AdelaiDet2/AdelaiDet/datasets/totaltext/test_add_seg.json"

# source_path = "/home/wengkangming/map_file/AdelaiDet2/AdelaiDet/datasets/CTW1500/annotations/train_ctw1500_maxlen100_v2.json"
# output = "/home/wengkangming/map_file/AdelaiDet2/AdelaiDet/datasets/CTW1500/annotations/add_seg_train_ctw1500_maxlen100_v2.json"

source_path = "/home/datasets/textGroup/mlt2017/train.json"
output = "/home/datasets/textGroup/mlt2017/train_add_seg.json"
text_bezier_pts_to_segementation(source_path, output)
