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
    # points = points.reshape(-1).tolist()
    return points.tolist()

resut_source_path = r"/home/wengkangming/map_file/AdelaiDet/output/batext/totaltext/attn_R_50/inference/text_results.json"
result_output = r"/home/wengkangming/map_file/AdelaiDet/output/batext/totaltext/attn_R_50/inference/coco_format_text_result.json"


def result_bezier_pts_to_segementation(resut_source_path, result_output):
    with open(resut_source_path, "r") as f:
        data = json.load(f)
        for annotations in data:
            polys = annotations["polys"]
            segmentation = list(itertools.chain.from_iterable(polys))
            annotations["segmentation"] = []
            annotations["segmentation"].append(segmentation)

    with open(result_output, "w") as f:
        json.dump(data, f)

"""
对输出结果的poly字段，转为segmentation
"""

result_bezier_pts_to_segementation(resut_source_path, result_output)
