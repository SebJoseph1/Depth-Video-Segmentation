from collections import OrderedDict

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from aspp import ASPP
from panopticdecoder import (PanopticDeepLabDecoder, PanopticDeepLabDepthHead,
                             PanopticDeepLabHead)
from resnet import resnet50


class MonoDVPS(nn.Module):
    def __init__(self,num_classes = 20):
        super(MonoDVPS, self).__init__()
        self.resnet_backbone = resnet50()
        self.instance_aspp = ASPP(2048, 256, [6, 12, 18])
        self.semantic_aspp = ASPP(2048, 256, [6, 12, 18])
        self.semantic_aspp.requires_grad_(requires_grad=False)
        self.dense_aspp = nn.ModuleList([ASPP(4096, 256, [6, 12, 18])] * 4)
        self.semantic_decoder = PanopticDeepLabDecoder(in_channels=256, feature_key="aspp",
                                                       low_level_channels=[512, 256], low_level_keys=["res3", "res2"],
                                                       low_level_channels_decoder=[64, 32],
                                                       decoder_channels=[256, 256, 256])
        self.semantic_decoder.requires_grad_(requires_grad=False)
        self.instance_decoder = PanopticDeepLabDecoder(in_channels=256, feature_key="aspp",
                                                       low_level_channels=[512, 256], low_level_keys=["res3", "res2"],
                                                       low_level_channels_decoder=[64, 32],
                                                       decoder_channels=[256, 128, 128])
        self.next_frame_instance_decoder = PanopticDeepLabDecoder(in_channels=1024, feature_key="aspp",
                                                                  low_level_channels=[512, 256],
                                                                  low_level_keys=["res3", "res2"],
                                                                  low_level_channels_decoder=[64, 32],
                                                                  decoder_channels=[256, 128, 128])
        self.depth_pred_head = PanopticDeepLabDepthHead(in_channel=256, decoder_channel=[32, 64], num_classes=1)
        #self.depth_pred_head.requires_grad_(requires_grad=False)
        self.semantic_pred_head = PanopticDeepLabHead(in_channel=256, decoder_channel=256, num_classes=num_classes)
        #self.semantic_pred_head.requires_grad_(requires_grad=False)
        self.instance_center_pred_head = PanopticDeepLabHead(in_channel=128, decoder_channel=32, num_classes=1)
        self.instance_center_regr_head = PanopticDeepLabHead(in_channel=128, decoder_channel=32, num_classes=2,with_relu=False)
        self.next_frame_instance_center_regr_head = PanopticDeepLabHead(in_channel=128, decoder_channel=32,
                                                                        num_classes=2,with_relu=False)

        # PanopticDeepLabDecoder(in_channels=2048,low_level_channels=[1024,512,256],low_level_key=["res4","res3","res2"],low_level_channels_project=[128,64,32],atrous_rates=[3,6,9],feature_key="res5",decoder_channels=256,num_classes=10)
        # self.fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=True)
        # self.fcn_resnet50.classifier[4] = nn.Conv2d(512, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # x = self.fcn_resnet50(x)['out']
        xt_0 = x[:, :3, :, :]
        xt_1 = x[:, 3:, :, :]
        bt_0 = self.resnet_backbone(xt_0)
        bt_1 = self.resnet_backbone(xt_1)
        inst_aspp = self.instance_aspp(bt_0["res5"])
        sem_aspp = self.semantic_aspp(bt_0["res5"])
        bt_01 = torch.cat([bt_0["res5"], bt_1["res5"]], dim=1)
        den_aspp = []
        for aspp in self.dense_aspp:
            den_aspp.append(aspp(bt_01))
        den_aspp = torch.cat(den_aspp, dim=1)
        sem_feature = {}
        bt_0["aspp"] = sem_aspp
        sem_dec = self.semantic_decoder(bt_0)
        bt_0["aspp"] = inst_aspp
        inst_dec = self.instance_decoder(bt_0)
        bt_1["aspp"] = den_aspp
        nxt_inst_dec = self.next_frame_instance_decoder(bt_1)
        depth = self.depth_pred_head(sem_dec)
        depth_resized = F.interpolate(depth, size=x.size()[2:], mode='bilinear', align_corners=True)
        sem_pred = self.semantic_pred_head(sem_dec)
        sem_pred_resized = F.interpolate(sem_pred, size=x.size()[2:], mode='bilinear', align_corners=True)
        inst_cntr_pred = self.instance_center_pred_head(inst_dec)
        inst_cntr_pred_resized = F.interpolate(inst_cntr_pred, size=x.size()[2:], mode='bilinear', align_corners=True)
        inst_cntr_regr = self.instance_center_regr_head(inst_dec)
        inst_cntr_regr_resized = F.interpolate(inst_cntr_regr, size=x.size()[2:], mode='bilinear', align_corners=True)
        nxt_inst_cntr_regr = self.next_frame_instance_center_regr_head(nxt_inst_dec)
        nxt_inst_cntr_regr_resized = F.interpolate(nxt_inst_cntr_regr, size=x.size()[2:], mode='bilinear',
                                                   align_corners=True)
        depth_resized_post_processed = F.sigmoid(depth_resized) * 88 # max depth
        # Crossentropy fix (?)
        sem_pred_resized_post = F.softmax(sem_pred_resized, dim=1)
        x = OrderedDict()
        x["depth"] = depth_resized_post_processed
        # Crossentropy fix (?)
        x["semantic"] = sem_pred_resized_post
        x["center"] = inst_cntr_pred_resized
        x["offset"] = inst_cntr_regr_resized
        x["nxtoffset"] = nxt_inst_cntr_regr_resized
        return x


if __name__ == "__main__":
    model = MonoDVPS()
    input_tensor = torch.randn(1, 6, 320, 320)
    output = model(input_tensor)
    print(model)
