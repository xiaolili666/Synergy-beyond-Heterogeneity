import torch
from torch import nn
import torch.nn.functional as F
# from NBSAM.image_encoder_neck4_LoRA_TransAda import ImageEncoderViT
from P3.Teacher1.old.Teacher.en1 import  DualBranchImageEncoderViT
from P3.Teacher1.old.Teacher.mask_decoder_ import MaskDecoder
from P3.Teacher1.old.Teacher.prompt_encoder_ import PromptEncoder
from P3.Teacher1.old.Teacher.transformer import TwoWayTransformer
from typing import Any, Dict, List, Tuple
from functools import partial
from torch.nn.functional import interpolate
from backbone.PVTv2.pvtv2_encoder import pvt_v2_b4


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in1, in2, in3, in4):
        super(Decoder, self).__init__()
        self.bcon4 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in4, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3 = BasicConv2d(in3, in4, kernel_size=3, stride=1, padding=1)
        self.bcon2 = BasicConv2d(in2, in3, kernel_size=3, stride=1, padding=1)
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in2, kernel_size=1, stride=1, padding=0)

        self.bcon4_3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4 * 2, in3, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3_2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3 * 2, in2, kernel_size=3, stride=1, padding=1)
        )
        self.bcon2_1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )

        self.conv_d1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in2, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d3 = BasicConv2d(in2, in1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        f[3] = self.bcon4(f[3])
        f[2] = self.bcon3(f[2])
        f[1] = self.bcon2(f[1])
        f[0] = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f[3], f[2]), 1))
        d32 = self.bcon3_2(torch.cat((d43, f[1]), 1))
        d21 = self.bcon2_1(torch.cat((d32, f[0]), 1))
        out = d21

        d43 = self.conv_d1(d43)
        d32 = torch.cat((d43, d32), dim=1)
        d32 = self.conv_d2(d32)
        d21 = torch.cat((d32, d21), dim=1)
        d21 = self.conv_d3(d21)

        return d21, out, d32, d43


class SAMT(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()

        self.pixel_mean: List[float] = [123.675, 116.28, 103.53]
        self.pixel_std: List[float] = [58.395, 57.12, 57.375]
        self.up1_4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear')
        # 图像和深度编码器
        self.image_encoder = DualBranchImageEncoderViT(  #
            depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256
        )
        # self.prompt_encoder = PromptEncoder()
        image_embedding_size = img_size // 16
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.out_sam = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True)
        )
        self.out = nn.Conv2d(1,256,1)

        self.c64 = nn.Conv2d(256, 64, 1)
        self.c128 = nn.Conv2d(256, 128, 1)
        self.c512 = nn.Conv2d(256, 512, 1)
        self.up1_2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
    def forward(self, img,depth, p=None):
        rgb ,dep= self.image_encoder(img,depth)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=p,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=rgb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            dense_prompt_embeddings=self.up1_4(dense_embeddings),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=False,
        )
        low_res_masks1, iou_predictions1 = self.mask_decoder(
            image_embeddings=dep,
            image_pe=self.prompt_encoder.get_dense_pe(),
            dense_prompt_embeddings=self.up1_4(dense_embeddings),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=False,
        )
        low_res_masks2, iou_predictions2 = self.mask_decoder(
            image_embeddings=dep+rgb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            dense_prompt_embeddings=self.up1_4(dense_embeddings),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=False,
        )

        low = low_res_masks+low_res_masks1+low_res_masks2
        return F.sigmoid(self.out_sam(low_res_masks2)), F.sigmoid(self.out_sam(low_res_masks1)),F.sigmoid(self.out_sam(low_res_masks)),F.sigmoid(self.out_sam(low)),self.up4(self.c64(rgb+dep)),self.up2(self.c128(rgb+dep)),(rgb+dep),self.up1_2(self.c512(rgb+dep)),dep#,self.up4(self.c64(dep)),self.up2(self.c128(dep)),rgb,self.up1_2(self.c512(rgb))



def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed


def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in k or '8' in k or '11' in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]


def load(net, ckpt, img_size):
    ckpt = torch.load(ckpt, map_location='cpu')
    from collections import OrderedDict
    dict = OrderedDict()
    # for k, v in ckpt.items():
    #     if 'pe_layer' in k:
    #         dict[k[15:]] = v
    #         continue
    #     if 'pos_embed' in k:
    #         dict[k] = reshapePos(v, img_size)
    #         continue
    #     if 'rel_pos' in k:
    #         dict[k] = reshapeRel(k, v, img_size)
    #     else:
    #         dict[k] = v

    for k, v in ckpt.items():
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k:
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            dict[k] = v
        # if "prompt_encoder" in k:
        #     dict[k] = v
        # if "mask_decoder" in k:
        #     dict[k] = v

    # 加载权重
    state1, state2 = net.load_state_dict(dict, strict=False)
    print("----------------------:", state1)
    print("----------------------:", state2)

    return "TRUE"
class PG1(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = pvt_v2_b4(pretrained=True)

        self.conv = nn.Conv2d(320, 256, 1)
        self.decoder_f1 = Decoder(64, 128, 256, 512)
        self.pre1_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.pre2_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.pre3_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        self.pre4_best = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, 3, 1, 1)
        )

    def forward(self,img,depth):
        # Net1
        rgb_1, rgb_2, rgb_3, rgb_4 = self.rgb_backbone(img)
        rgb_3 = self.conv(rgb_3)
        # # d
        # x_d = torch.cat([depth, depth, depth], dim=1)
        d_1, d_2, d_3, d_4 = self.rgb_backbone(depth)
        d_3 = self.conv(d_3)

        fuse_4 = rgb_4+ d_4
        fuse_3 = rgb_3+ d_3
        fuse_2 = rgb_2+ d_2
        fuse_1 = rgb_1+ d_1

        pre1, pre2, pre3, pre4 = self.decoder_f1([fuse_1, fuse_2, fuse_3, fuse_4])

        # return  F.sigmoid(self.pre1_best(pre1)), F.sigmoid(self.pre2_best(pre2)), F.sigmoid(self.pre3_best(pre3)), F.sigmoid(self.pre4_best(pre4))#,fuse_1, fuse_2, fuse_3, fuse_4
        return self.pre1_best(pre1),self.pre2_best(pre2),self.pre3_best(pre3), self.pre4_best(pre4)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = PG1()

        self.SAM = SAMT()

        load(self.SAM,
             "/media/pc12/data/LYQ/SAM/COD_LYQ/SAM_Box/sam_vit_b_01ec64.pth", 256)

    def forward(self, img,depth):
        pre1_1,pre1_2,pre1_3,pre1_4 =self.P(img,depth)

        # load(self.SAM,
        #              "/media/pc24/data/LYQ/SAM_Box/sam_vit_b_01ec64.pth", 256)
        # self.DINO.load_state_dict(torch.load('/media/pc12/data/LYQ/SAM/COD_LYQ/SAM_Box/SAM_boxRSDD-129.pth'), False)
        out1,out2,out3,out4,b1,b2,b3,b4,B = self.SAM(img,depth,p=(pre1_1+pre1_2))#self.conv2_1(pro)
        return (out1,out2,out3,out4,
                F.sigmoid(pre1_1), F.sigmoid(pre1_2),F.sigmoid(pre1_3),F.sigmoid(pre1_4),b1,b2,b3,b4,B#, F.sigmoid(pre1_3)#,F.sigmoid(pre1_4),
                # F.sigmoid(pre2_1), F.sigmoid(pre2_2), F.sigmoid(pre2_3), F.sigmoid(pre2_4)
                # pre2_1, pre2_2, pre2_3, pre2_4
                )




# 测试代码
# if __name__ == "__main__":
#     # 创建输入: [batch_size, channels, height, width]
#     x = torch.randn(2, 256, 16, 16)
#
#     # 创建模型
#     model = M2FMoE_ImageBlock(dim=256)
#
#     # 前向传播
#     output = model(x)
#
#     # 验证输出形状
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Output shape matches input shape: {output.shape == x.shape}")
#
#     # 验证模型可训练
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
if __name__ == '__main__':
    model = Net().cuda()
    right = torch.randn(1, 3, 256, 256).cuda()
    left = torch.randn(1, 3, 256, 256).cuda()
    out = model(right, left)

    from FLOP import CalParams

    CalParams(model,right, left)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    for i in out:
        print(i.shape)


# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 1, 256, 256])
# torch.Size([2, 64, 64, 64])
# torch.Size([2, 128, 32, 32])
# torch.Size([2, 256, 16, 16])
# torch.Size([2, 512, 8, 8])

# FLOPs: 213.102G
# Params: 165.204M
#  ####################
# Total params  162.98M