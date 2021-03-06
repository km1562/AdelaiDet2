import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F
from torch.nn import Parameter, Module
from ..poolers import ROIPooler, TopPooler


class Segmentation_head(nn.Module):
    """
    Segmentation module. given fpn features, get seg logits and global context

    Args:
        features (lsit[Tensor]): fpn features
        feature_level (int): select which level to align
        proposal_boxes (list[Tensor]): selected positive proposals
        image_shape: reshaped size of training img
    Returns:
        seg_logits (Tensor): segmentation result
        global_context (Tensor): features of global context
    """
    def __init__(self, cfg):
        super(Segmentation_head, self).__init__()

        self.channels = cfg.MODEL.TEXTFUSENET_SEG_HEAD.CHANNELS
        self.num_fpn_features = cfg.MODEL.TEXTFUSENET_SEG_HEAD.NUM_FPN_FEATURES
        self.num_conv3x3 = cfg.MODEL.TEXTFUSENET_SEG_HEAD.NUM_CONV3
        self.num_classes = cfg.MODEL.TEXTFUSENET_SEG_HEAD.NUM_CLASSES
        self.pooler_resolution = cfg.MODEL.TEXTFUSENET_SEG_HEAD.POOLER_RESOLUTION
        self.pooler_scales = cfg.MODEL.TEXTFUSENET_SEG_HEAD.POOLER_SCALES
        self.sampling_ratio = cfg.MODEL.TEXTFUSENET_SEG_HEAD.SAMPLING_RATIO
        self.pooler_type = cfg.MODEL.TEXTFUSENET_SEG_HEAD.POOLER_TYPE

        self.canonical_size = cfg.MODEL.BATEXT.CANONICAL_SIZE

        # layers----get global fused features and global context
        self.conv1x1_list = nn.ModuleList()
        for i in range(self.num_fpn_features):
            # self.conv1x1_list.append(nn.Conv2d(self.channels, self.channels, 1, padding=1, bias=False))
            self.conv1x1_list.append(nn.Conv2d(self.channels, self.channels, 1, padding=0, bias=False))

        self.conv3x3_list = nn.ModuleList()
        for i in range(self.num_conv3x3):
            self.conv3x3_list.append(nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False))

        # TODO 换成bezier_align会好点吗？
        # self.seg_pooler = ROIPooler(
        #     output_size=self.pooler_resolution,
        #     scales=self.pooler_scales,
        #     sampling_ratio=self.sampling_ratio,
        #     pooler_type=self.pooler_type,
        # )
        self.seg_pooler = TopPooler(
            output_size=self.pooler_resolution,
            scales=self.pooler_scales,
            sampling_ratio=self.sampling_ratio,
            pooler_type="BezierAlign",
            # canonical_box_size=self.canonical_size,
            # canonical_level=3,
            assign_crit="bezier")

        self.conv3x3_list_roi = nn.ModuleList()
        for i in range(self.num_conv3x3):
            self.conv3x3_list_roi.append(nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False))


        # layers---segmentation logits
        self.conv1x1_seg_logits = nn.Conv2d(self.channels, self.channels, 1, padding=0, bias=False)
        self.seg_logits = nn.Conv2d(self.channels, self.num_classes, 1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, features, feature_level, proposal_boxes, image_shape):
        feature_shape = features[feature_level].shape[-2:]
        if isinstance(features,list):
            feature_fuse = self.conv1x1_list[feature_level](features[feature_level])
        else: #单个特征图
           feature_fuse = self.conv1x1_list[feature_level](features)

        # get global fused features
        for i, feature in enumerate(features):
            if i != feature_level:
                feature = F.interpolate(feature, size=feature_shape, mode='bilinear', align_corners=True)
                feature_fuse += self.conv1x1_list[i](feature)

        for i in range(self.num_conv3x3):
            feature_fuse = self.conv3x3_list[i](feature_fuse)


        # get global context
        global_context = self.seg_pooler([feature_fuse], proposal_boxes)
        for i in range(self.num_conv3x3):
            global_context = self.conv3x3_list_roi[i](global_context)
        global_context = self.relu(global_context)

        #TODO 这里不用融合了，只需要一系列卷积+pooler就可以了，

        # get segmentation logits
        # if len(feature_fuse) > 1:
        #     #batch > 1,实际上每个batch就是一个
        #     feature_preds = []
        #     feature_fuse_list = torch.split(feature_fuse, 1)
        #     for i in range(len(feature_fuse_list)):
        #         feature_pred = F.interpolate(feature_fuse_list[i],size=image_shape[i],mode='bilinear',align_corners=True)
        #         feature_pred = self.conv1x1_seg_logits(feature_pred)
        #         feature_pred = self.seg_logits(feature_pred)
        #         feature_preds.append(feature_pred)
        #     seg_logits = torch.cat(feature_preds, dim=0)
        # else:
        feature_pred = F.interpolate(feature_fuse, size=image_shape[0], mode='bilinear', align_corners=True)
        feature_pred = self.conv1x1_seg_logits(feature_pred)
        seg_logits = self.seg_logits(feature_pred)

        return seg_logits, global_context


def build_seg_head(cfg):
    return Segmentation_head(cfg)


############################  seg head loss #############################
def make_segmentation_gt(targets):

    W = targets[0].image_size[1]
    H = targets[0].image_size[0]

    classes = targets[0].gt_classes
    word_indx = (classes==0).nonzero() #非0就是单词

    gt_polygon_list = targets[0].gt_masks.polygons

    imglist = []
    for i in word_indx:
        point = gt_polygon_list[i][0]
        point = np.array(point.reshape(int(len(point) / 2), 2), dtype=np.int32)
        img = np.zeros([W, H], dtype="uint8")
        imglist.append(torch.Tensor(cv2.fillPoly(img, [point], (1, 1, 1))))

    imglist = torch.stack(imglist)
    segmentation_gt = torch.sum(imglist, dim=0)
    segmentation_gt = segmentation_gt > 0
    segmentation_gt = segmentation_gt.reshape(1, W, H)
    segmentation_gt = segmentation_gt.cuda()

    return segmentation_gt


class Segmentation_LossComputation(Module):
    """
    compute seg loss

    """
    def __init__(self):
        super(Segmentation_LossComputation, self).__init__()

        self.weight = 0.2
        self.loss = nn.CrossEntropyLoss()

    def forward(self, seg_logits, targets):

        seg_gt = make_segmentation_gt(targets)
        loss_seg = self.weight * self.loss(seg_logits, seg_gt.long())

        return loss_seg

def build_seg_head_loss():
    loss_evaluator = Segmentation_LossComputation()
    return loss_evaluator


