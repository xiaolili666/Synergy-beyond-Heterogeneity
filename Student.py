import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from einops import rearrange
from torch.nn.functional import interpolate
from backbone.Shunted_Transformer.SSA import shunted_b, shunted_s, shunted_t
# from backbone.Dformer.Dformer import DFormer_Tiny
stage1_channel = 64
stage2_channel = 128
stage3_channel = 256
stage4_channel = 512
# stage1_channel = 32
# stage2_channel = 64
# stage3_channel = 128
# stage4_channel = 256
# h_pic = 256//32

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


# class LayerNorm(nn.Module):
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x



class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                # nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)


# class WF_V2(nn.Module):
#     def __init__(self):
#         super(WF_V2, self).__init__()
#         self.alph = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_q = rearrange(x, 'b c h w -> b c (h w)')
#         x_k = rearrange(x, 'b c h w -> b (h w) c')
#
#         x_dot_c = torch.bmm(x_q, x_k)
#
#         x_q_c = torch.norm(x_q, p=2, dim=2).view(b, -1, c).permute(0, 2, 1)
#         x_k_c = torch.norm(x_k, p=2, dim=1).view(b, -1, c)
#         x_dot_c_ = torch.bmm(x_q_c, x_k_c) + 1e-08
#         atten_map_c = torch.div(x_dot_c, x_dot_c_)
#         x_v_c = x.view(b, c, -1)
#         out_c = torch.bmm(atten_map_c, x_v_c)
#         out_c = out_c.view(b, c, h, w)
#
#         out = self.alph * out_c + x
#
#         return out
class LC_V(nn.Module):
    """
    全面增强版模块：融合多尺度空间注意力与跨维度交互
    无需传入通道数，参数量几乎为零，仅保留原有的 alph 增益系数。
    """

    def __init__(self):
        super(LC_V, self).__init__()
        self.alph = nn.Parameter(torch.zeros(1))
        # 这里的池化层是无参数的
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.shape

        # --- 分支 1: 全局统计门控 (Global Stats Gate) ---
        # 捕捉通道的显著性权重
        channel_attn = torch.sigmoid(self.max_pool(x) + self.avg_pool(x))
        x_c = x * channel_attn

        # --- 分支 2: 多尺度空间自适应 (Multi-scale Spatial Awareness) ---
        # 通过 AvgPool 模拟不同感受野，捕捉大面积显著区域和局部细节
        s1 = F.interpolate(F.avg_pool2d(x_c, kernel_size=3, stride=2, padding=1), size=(h, w), mode='bilinear')
        s2 = F.interpolate(F.avg_pool2d(x_c, kernel_size=5, stride=4, padding=2), size=(h, w), mode='bilinear')

        # 融合多尺度特征并生成空间掩码
        spatial_attn = torch.sigmoid(s1 + s2)
        x_s = x_c * spatial_attn

        # --- 分支 3: 跨维度交互 (Cross-Dim Interaction) ---
        # 参考 Triplet Attention，将 H 或 W 与 C 交换，捕捉跨维度的结构信息
        # 处理宽维度交互 (C 与 W)
        x_w = x_s.permute(0, 2, 1, 3)  # (B, H, C, W)
        x_w_attn = torch.sigmoid(torch.mean(x_w, dim=1, keepdim=True))
        x_s = (x_w * x_w_attn).permute(0, 2, 1, 3)

        # --- 最终融合 ---
        # 保持原有的残差结构
        out = self.alph * x_s + x
        return out
class student_single(nn.Module):
    def __init__(self):
        super(student_single, self).__init__()
        # Backbone model
        self.rgb = shunted_t(pretrained=True)

        self.Head1 = SalHead(stage1_channel)
        self.Head2 = SalHead(stage2_channel)
        self.Head3 = SalHead(stage3_channel)
        self.Head4 = SalHead(stage4_channel)

        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.lc_1 = LC_V()
        self.lc_2 = LC_V()
        self.lc_3 = LC_V()
        self.lc_4 = LC_V()

        self.conv_cross_1 = convbnrelu(stage2_channel, stage1_channel, k=1, s=1, p=0, relu=True)
        self.conv_cross_2 = convbnrelu(stage3_channel, stage2_channel, k=1, s=1, p=0, relu=True)
        self.conv_cross_3 = convbnrelu(stage4_channel, stage3_channel, k=1, s=1, p=0, relu=True)

        self.conv_concat_1 = convbnrelu(2 * stage1_channel, stage1_channel, k=1, s=1, p=0, relu=True)
        self.conv_concat_2 = convbnrelu(2 * stage2_channel, stage2_channel, k=1, s=1, p=0, relu=True)
        self.conv_concat_3 = convbnrelu(2 * stage3_channel, stage3_channel, k=1, s=1, p=0, relu=True)
        self.conv_concat_4 = convbnrelu(2 * stage4_channel, stage4_channel, k=1, s=1, p=0, relu=True)
    def forward(self, x_rgb):
        rgb_list = self.rgb(x_rgb)
        rgb_1 = rgb_list[0]
        rgb_2 = rgb_list[1]
        rgb_3 = rgb_list[2]
        rgb_4 = rgb_list[3]

        stage1_out = self.conv_concat_1(torch.cat([rgb_1, rgb_1], dim=1))
        stage2_out = self.conv_concat_2(torch.cat([rgb_2, rgb_2], dim=1))
        stage3_out = self.conv_concat_3(torch.cat([rgb_3, rgb_3], dim=1))
        stage4_out = self.conv_concat_4(torch.cat([rgb_4, rgb_4], dim=1))



        am_out_4 = self.lc_4(stage4_out)
        am_out_3 = self.lc_3(self.upsample2(self.conv_cross_3(am_out_4)) + stage3_out)
        am_out_2 = self.lc_2(self.upsample2(self.conv_cross_2(am_out_3)) + stage2_out)
        am_out_1 = self.lc_1(self.upsample2(self.conv_cross_1(am_out_2)) + stage1_out)

        fuse_1 = self.upsample4(self.Head1(am_out_1))
        fuse_2 = self.upsample8(self.Head2(am_out_2))
        fuse_3 = self.upsample16(self.Head3(am_out_3))
        fuse_4 = self.upsample32(self.Head4(am_out_4))

        return F.sigmoid(fuse_1),F.sigmoid(fuse_2) , F.sigmoid(fuse_3), F.sigmoid(fuse_4),stage1_out, stage2_out, stage3_out, stage4_out,am_out_1, am_out_2, am_out_3, am_out_4


if __name__ == '__main__':

    a = torch.randn(2, 3, 256, 256)

    model = student_single()
    out = model(a)
    for out in out:
        print("out1", out.shape)
    from FLOP import CalParams2

    CalParams2(model, a)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # [Statistics Information]
# FLOPs: 5.589G
# Params: 11.913M

# out1 torch.Size([2, 1, 256, 256])
# out1 torch.Size([2, 1, 256, 256])
# out1 torch.Size([2, 1, 256, 256])
# out1 torch.Size([2, 1, 256, 256])
# out1 torch.Size([2, 64, 64, 64])
# out1 torch.Size([2, 128, 32, 32])
# out1 torch.Size([2, 256, 16, 16])
# out1 torch.Size([2, 512, 8, 8])