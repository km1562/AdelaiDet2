import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm
from adet.layers import NaiveGroupNorm

class FPA(nn.Module):
    def __init__(self, channels=2048):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False) #2048,2048
        # self.bn_master = NaiveSyncBatchNorm(channels)
        self.bn_master = NaiveGroupNorm(32, channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)#2048,2048
        # self.bn_gpb = NaiveSyncBatchNorm(channels)
        self.bn_gpb = NaiveGroupNorm(32, channels)

        # C333 because of the shape of last feature maps is (16, 16).
        # self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False) #2048,512
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3,
                                   bias=False)  # 2048,512
        # self.bn1_1 = NaiveSyncBatchNorm(channels_mid)
        self.bn1_1 = NaiveGroupNorm(32, channels_mid)

        # self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False) #512,512
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2,
                                   bias=False)  # 512,512
        # self.bn2_1 = NaiveSyncBatchNorm(channels_mid)
        self.bn2_1 = NaiveGroupNorm(32, channels_mid)

        # self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)#512,512
        # self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)#512,512
        # self.bn1_2 = NaiveSyncBatchNorm(channels_mid)
        self.bn1_2 = NaiveGroupNorm(32, channels_mid)

        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)#512,512
        # self.bn2_2 = NaiveSyncBatchNorm(channels_mid)
        self.bn2_2 = NaiveGroupNorm(32, channels_mid)
        # self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)#512,512
        # self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        # self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        # self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        # self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = NaiveSyncBatchNorm(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        # x3_1 = self.conv3x3_1(x2_1)
        # x3_1 = self.bn3_1(x3_1)
        # x3_1 = self.relu(x3_1)
        # x3_2 = self.conv3x3_2(x3_1)
        # x3_2 = self.bn3_2(x3_2)

        # Merge branch 3 and 2
        # x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        # x2_merge = self.relu(x2_2 + x3_upsample)
        # x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        # x1_merge = self.relu(x1_2 + x2_upsample)

        # x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_2)))
        # x1_merge = self.relu(x1_2 + x2_upsample)
        x1_merge = x2_2 + x1_2

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))
###????????????????????????????????????????? x_master + x_gdb???????
        #print('x_master:{},x_gpb:{}'.format(x_master.shape,x_gpb.shape))
        out = self.relu(x_master + x_gpb)

        return out