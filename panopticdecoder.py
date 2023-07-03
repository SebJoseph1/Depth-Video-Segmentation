# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from aspp import ASPP
from diff_conv import stacked_conv


class PanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_keys, low_level_channels_decoder,
                 decoder_channels):
        super(PanopticDeepLabDecoder, self).__init__()
        self.feature_key = feature_key
        self.low_level_feature_keys = low_level_keys
        self.first_decoder = stacked_conv(in_planes=in_channels, out_planes=decoder_channels[0], kernel_size=1,
                                          padding=0, stride=16, num_stack=1, conv_type='basic_conv')
        self.backbone_first_decoder = stacked_conv(in_planes=low_level_channels[0],
                                                   out_planes=low_level_channels_decoder[0], kernel_size=1, padding=0,
                                                   num_stack=1, conv_type='basic_conv')
        self.third_upsample_decoder = stacked_conv(in_planes=decoder_channels[0] + low_level_channels_decoder[0],
                                                   out_planes=decoder_channels[1], kernel_size=5, padding=2,
                                                   num_stack=1, conv_type='depthwise_separable_conv')
        self.backbone_second_decoder = stacked_conv(in_planes=low_level_channels[1],
                                                    out_planes=low_level_channels_decoder[1], kernel_size=1, padding=0,
                                                    num_stack=1, conv_type='basic_conv')
        self.fourth_decoder = stacked_conv(in_planes=decoder_channels[1] + low_level_channels_decoder[1],
                                           out_planes=decoder_channels[2], kernel_size=5, num_stack=1, padding=2,
                                           conv_type='basic_conv')

    def forward(self, features):
        x1 = self.first_decoder(features[self.feature_key])
        l2 = self.backbone_first_decoder(features[self.low_level_feature_keys[0]])
        x2 = F.interpolate(x1, size=l2.size()[2:], mode='bilinear', align_corners=True)
        xl2 = torch.cat((x2, l2), dim=1)
        x3 = self.third_upsample_decoder(xl2)
        l3 = self.backbone_second_decoder(features[self.low_level_feature_keys[1]])
        x3 = F.interpolate(x3, size=l3.size()[2:], mode='bilinear', align_corners=True)
        xl3 = torch.cat((x3, l3), dim=1)
        x4 = self.fourth_decoder(xl3)
        return x4


class PanopticDeepLabHead(nn.Module):
    def __init__(self, in_channel, decoder_channel, num_classes,with_relu=True):
        super(PanopticDeepLabHead, self).__init__()
        self.first_layer = stacked_conv(in_planes=in_channel, out_planes=decoder_channel, kernel_size=5, num_stack=1,
                                        padding=2, conv_type='basic_conv')
        self.second_layer = stacked_conv(in_planes=decoder_channel, out_planes=num_classes, kernel_size=1, padding=0,
                                         num_stack=1, conv_type='basic_conv',with_relu=with_relu)

    def forward(self, x):
        x1 = self.first_layer(x)
        x2 = self.second_layer(x1)
        return x2


class PanopticDeepLabDepthHead(nn.Module):
    def __init__(self, in_channel, decoder_channel, num_classes):
        super(PanopticDeepLabDepthHead, self).__init__()
        self.first_layer = stacked_conv(in_planes=in_channel, out_planes=decoder_channel[0], kernel_size=5, num_stack=1,
                                        padding=2, conv_type='basic_conv')
        self.second_layer = stacked_conv(in_planes=decoder_channel[0], out_planes=decoder_channel[1], kernel_size=3,
                                         num_stack=1, conv_type='depthwise_separable_conv')
        self.third_layer = stacked_conv(in_planes=decoder_channel[1], out_planes=num_classes, kernel_size=1, padding=0,
                                        num_stack=1, conv_type='basic_conv',with_relu=False)

    def forward(self, x):
        x1 = self.first_layer(x)
        x2 = self.second_layer(x1)
        x3 = self.third_layer(x2)
        return x3


if __name__ == "__main__":
    model = PanopticDeepLabDecoder(in_channels=256, low_level_channels=[512, 256], low_level_keys=["res3", "res2"],
                                   low_level_channels_decoder=[64, 32], feature_key="res5",
                                   decoder_channels=[256, 128, 128])
    input_tensor1 = torch.randn(8, 256, 13, 13)
    input_tensor2 = torch.randn(8, 256, 50, 50)
    input_tensor3 = torch.randn(8, 512, 25, 25)
    input_tensor = {}
    input_tensor["res5"] = input_tensor1
    input_tensor["res2"] = input_tensor2
    input_tensor["res3"] = input_tensor3
    y = model(input_tensor)
    k = PanopticDeepLabDepthHead(in_channel=128, decoder_channel=[32, 64], num_classes=1)

    q = k(y)
    u = 0
