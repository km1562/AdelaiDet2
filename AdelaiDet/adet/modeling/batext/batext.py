import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, IOULoss
from .batext_outputs import BATextOutputs
from detectron2.layers import DeformConv

__all__ = ["BAText"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

# class Predict_pre_nms_thresh(nn.Module):
#     """
#     就是利用batext做预测bezier的几个层，
#     1、融合
#     2、卷积+池化，加flatten，+sigmoid衰减一个维度出来
#     3、预测一个阈值出来
#
#     """
#     def __init__(self, cfg):
#         super(Predict_pre_nms_thresh, self).__init__()
#         self.channels = cfg.MODEL.PREDICT_PROB.CHANNELS
#         self.num_fpn_features = cfg.MODEL.PREDICT_PROB.NUM_FPN_FEATURES
#         self.num_conv3x3 = cfg.MODEL.PREDICT_PROB.NUM_CONV3
#         self.feature_level = cfg.MODEL.PREDICT_PROB.FPN_FEATURES_FUSED_LEVEL
#         self.conv1x1_list = nn.ModuleList()
#         for i in range(self.num_fpn_features):
#             self.conv1x1_list.append(nn.Conv2d(self.channels, self.channels, 1,padding=0, bias=False))
#
#         self.conv3x3_list = nn.ModuleList()
#         for i in range(self.num_conv3x3):
#             self.conv3x3_list.append(nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False))
#         self.conv_pool = nn.Sequential(
#             nn.MaxPool2d((2, 2), stride=2),
#             nn.Conv2d(self.channels, 1, 3, padding=1, bias=False),
#             nn.AdaptiveMaxPool2d((7, 7)),
#             nn.Flatten(),
#             nn.Linear(7 * 7, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, features):
#         feature_shape = features[self.feature_level].shape[-2:]
#         feature_fuse = self.conv1x1_list[self.feature_level](features[self.feature_level])
#
#         for i, feature in enumerate(features):
#             if i != self.feature_level:
#                 feature = F.interpolate(feature, size=feature_shape, mode='bilinear', align_corners=True)
#                 feature_fuse += self.conv1x1_list[i](feature)
#
#         for i in range(self.num_conv3x3):
#             feature_fuse = self.conv3x3_list[i](feature_fuse)
#         prob = self.conv_pool(feature_fuse)
#         print("prob's value", prob)
#         return prob

@PROPOSAL_GENERATOR_REGISTRY.register()
class BAText(nn.Module):
    """
    A modified version of FCOS with Bezier regression
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides              = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius               = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.FCOS.NMS_TH
        self.yield_proposal       = cfg.MODEL.FCOS.YIELD_PROPOSAL
        self.post_nms_topk_train  = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.FCOS.THRESH_WITH_CTR
        # fmt: on
        self.iou_loss = IOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        # self.predict_pre_nms_thresh = Predict_pre_nms_thresh(cfg)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `ori_annotation_file_list` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]

        #TODO,这里就进行multi_fuse,+卷积softmax对他进行一个阈值的输出，并且融合的feature_fuse要传出去

        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            # pre_nms_thresh = self.predict_pre_nms_thresh(features)
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            # pre_nms_thresh = self.predict_pre_nms_thresh(features)
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        outputs = BATextOutputs(
            images,
            locations,
            logits_pred,
            reg_pred,
            ctrness_pred,
            top_feats,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.fcos_head.num_classes,
            pre_nms_thresh,
            pre_nms_topk,
            self.nms_thresh,
            post_nms_topk,
            self.thresh_with_ctr,
            gt_instances
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)}

        if self.training:
            losses = outputs.losses()
            results = outputs.predict_proposals(top_feats)
        else:
            losses = {}
            with torch.no_grad():
                proposals = outputs.predict_proposals(top_feats)
            if self.yield_proposal:
                results["proposals"] = proposals
            else:
                results = proposals
        return results, losses

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                False),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  cfg.MODEL.FCOS.USE_DEFORMABLE)}
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            if use_deformable:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            for i in range(num_convs):
                tower.append(conv_func(
                    in_channels, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers
