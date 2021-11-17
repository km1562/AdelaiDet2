import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone, build_resnet_backbone
from detectron2.modeling import BACKBONE_REGISTRY
from .mobilenet import build_mnv2_backbone
# from .resizer import Resizer

__all__ = []


def swish(x):
    return x * x.sigmoid()


def split_name(name):
    for i, c in enumerate(name):
        if not c.isalpha():
            return name[:i], int(name[i:])
    raise ValueError()


class FeatureMapResampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=""):
        super(FeatureMapResampler, self).__init__()
        if in_channels != out_channels:
            self.reduction = Conv2d(
                in_channels, out_channels, kernel_size=1,
                bias=(norm == ""),
                norm=get_norm(norm, out_channels),
                activation=None
            )
        else:
            self.reduction = None

        assert stride <= 2
        self.stride = stride

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)

        if self.stride == 2:
            x = F.max_pool2d(
                x, kernel_size=self.stride + 1,
                stride=self.stride, padding=1
            )
        elif self.stride == 1:
            pass
        else:
            raise NotImplementedError()
        return x


class BackboneWithTopLevels(Backbone):
    def __init__(self, backbone, out_channels, num_top_levels, norm=""):
        super(BackboneWithTopLevels, self).__init__()
        self.backbone = backbone
        backbone_output_shape = backbone.output_shape()

        self._out_feature_channels = {name: shape.channels for name, shape in backbone_output_shape.items()}
        self._out_feature_strides = {name: shape.stride for name, shape in backbone_output_shape.items()}
        self._out_features = list(self._out_feature_strides.keys())

        last_feature_name = max(self._out_feature_strides.keys(), key=lambda x: split_name(x)[1])
        self.last_feature_name = last_feature_name
        self.num_top_levels = num_top_levels

        last_channels = self._out_feature_channels[last_feature_name]
        last_stride = self._out_feature_strides[last_feature_name]

        prefix, suffix = split_name(last_feature_name)
        prev_channels = last_channels
        for i in range(num_top_levels):
            name = prefix + str(suffix + i + 1)
            self.add_module(name, FeatureMapResampler(
                prev_channels, out_channels, 2, norm
            ))
            prev_channels = out_channels

            self._out_feature_channels[name] = out_channels
            self._out_feature_strides[name] = last_stride * 2 ** (i + 1)
            self._out_features.append(name)

    def forward(self, x):
        outputs = self.backbone(x)
        last_features = outputs[self.last_feature_name]
        prefix, suffix = split_name(self.last_feature_name)

        x = last_features
        for i in range(self.num_top_levels):
            name = prefix + str(suffix + i + 1)
            x = self.__getattr__(name)(x)
            outputs[name] = x

        return outputs


class SingleBiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, in_channels_list, out_channels, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super(SingleBiFPN, self).__init__()

        self.out_channels = out_channels
        # build 5-levels bifpn
        if len(in_channels_list) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels_list) == 6:
            self.nodes = [
                {'feat_level': 4, 'inputs_offsets': [4, 5]},
                {'feat_level': 3, 'inputs_offsets': [3, 6]},
                {'feat_level': 2, 'inputs_offsets': [2, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 8]},
                {'feat_level': 0, 'inputs_offsets': [0, 9]},
                {'feat_level': 1, 'inputs_offsets': [1, 9, 10]},
                {'feat_level': 2, 'inputs_offsets': [2, 8, 11]},
                {'feat_level': 3, 'inputs_offsets': [3, 7, 12]},
                {'feat_level': 4, 'inputs_offsets': [4, 6, 13]},
                {'feat_level': 5, 'inputs_offsets': [5, 14]},
            ]
        elif len(in_channels_list) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels_list]

        num_output_connections = [0 for _ in in_channels_list]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    lateral_conv = Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        norm=get_norm(norm, out_channels)
                    )
                    self.add_module(
                        "lateral_{}_f{}".format(input_offset, feat_level), lateral_conv
                    )
            node_info.append(out_channels)
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            self.add_module(name, Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm=get_norm(norm, out_channels),
                bias=(norm == "")
            ))
        # self.Resize = Resizer()

    def forward(self, feats):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        feats = [_ for _ in feats]
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_f{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            # mode="nearest"
                            mode="bilinear"
                        )
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()
        return output_feats


class BiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
        self, bottom_up, in_features, out_channels, num_top_levels, num_repeats, norm=""
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            num_top_levels (int): the number of the top levels (p6 or p7).
            num_repeats (int): the number of repeats of BiFPN.
            norm (str): the normalization to use.
        """
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # add extra feature levels (i.e., 6 and 7)
        self.bottom_up = BackboneWithTopLevels(
            bottom_up, out_channels,
            num_top_levels, norm
        )
        bottom_up_output_shapes = self.bottom_up.output_shape()

        in_features = sorted(in_features, key=lambda x: split_name(x)[1])
        self._size_divisibility = bottom_up_output_shapes[in_features[-1]].stride
        self.out_channels = out_channels
        self.min_level = split_name(in_features[0])[1]

        # add the names for top blocks
        prefix, last_suffix = split_name(in_features[-1])
        for i in range(num_top_levels):
            in_features.append(prefix + str(last_suffix + i + 1))
        self.in_features = in_features

        # generate output features
        self._out_features = ["p{}".format(split_name(name)[1]) for name in in_features]
        self._out_feature_strides = {
            out_name: bottom_up_output_shapes[in_name].stride
            for out_name, in_name in zip(self._out_features, in_features)
        }
        self._out_feature_channels = {k: out_channels for k in self._out_features}

        # build bifpn
        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_repeats):
            if i == 0:
                in_channels_list = [
                    bottom_up_output_shapes[name].channels for name in in_features
                ]
            else:
                in_channels_list = [
                    self._out_feature_channels[name] for name in self._out_features
                ]
            self.repeated_bifpn.append(SingleBiFPN(
                in_channels_list, out_channels, norm
            ))

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """
        bottom_up_features = self.bottom_up(x)
        feats = [bottom_up_features[f] for f in self.in_features]

        for bifpn in self.repeated_bifpn:
             feats = bifpn(feats)

        return dict(zip(self._out_features, feats))


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.MOBILENET:
        bottom_up = build_mnv2_backbone(cfg, input_shape)
    else:
        bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.BiFPN.IN_FEATURES
    out_channels = cfg.MODEL.BiFPN.OUT_CHANNELS
    num_repeats = cfg.MODEL.BiFPN.NUM_REPEATS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS

    backbone = BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        num_top_levels=top_levels,
        num_repeats=num_repeats,
        norm=cfg.MODEL.BiFPN.NORM
    )
    return backbone
