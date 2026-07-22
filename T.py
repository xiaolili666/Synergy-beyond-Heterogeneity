

import torch.nn as nn
import torch
import torch.nn.functional as F

from P3.DINO4_backbone import DINOv3_vit


class TriExpertDispatcher(nn.Module):
    def __init__(self, proto_c=512, low_c=64, mid_c=128, high_c=256):
        super().__init__()
        # 针对不同层级的特征维度，设计独立的专家投影器
        # 每一层都会从原始原型中学习到不同的“博弈策略”
        self.to_low = nn.Sequential(
            nn.Linear(proto_c, low_c),
            nn.LayerNorm(low_c),
            nn.ReLU()
        )
        self.to_mid = nn.Sequential(
            nn.Linear(proto_c, mid_c),
            nn.LayerNorm(mid_c),
            nn.ReLU()
        )
        self.to_high = nn.Sequential(
            nn.Linear(proto_c, high_c),
            nn.LayerNorm(high_c),
            nn.ReLU()
        )

    def forward(self, refined_proto):
        """
        refined_proto: 经过语义培训模块优化后的原型 (K, 512)
        """
        # 分离出三组面向不同层级的专家向量
        p_low = self.to_low(refined_proto)  # (K, 64)
        p_mid = self.to_mid(refined_proto)  # (K, 128)
        p_high = self.to_high(refined_proto)  # (K, 256)

        return p_low, p_mid, p_high


class MVGR_Refiner(nn.Module):
    def __init__(self, low_c, proto_c):
        """
        low_c: 当前待校准层特征的通道数 (如 64, 128, 256)
        proto_c: 原型专家的通道数，实际传入的是 TriExpertDispatcher 输出的维度 (64/128/256)
        """
        super().__init__()
        # 1. 视角投影矩阵：将原型映射为博弈中的两个参与者
        self.player_salience = nn.Linear(proto_c, low_c)
        self.player_consistency = nn.Linear(proto_c, low_c)

        # 2. 交互权重
        self.coalition_weights = nn.Parameter(torch.ones(1, 2) * 0.5)

        # 3. 空间拓扑门控
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(low_c, low_c, 3, padding=1, groups=low_c),
            nn.BatchNorm2d(low_c),
            nn.Sigmoid()
        )

        # 4. 融合输出
        self.out_proj = nn.Conv2d(low_c, low_c, 1)

    def forward(self, f_low, refined_proto):
        """
        f_low: 待校准特征图 (B, C_low, H, W)
        refined_proto: 来自 TriExpertDispatcher 的专家向量 (K, C_proto)
                       C_proto 对应 low_c/mid_c/high_c (64/128/256)
        """
        B, C, H, W = f_low.shape

        # 聚合 K 个原型，增加 batch 维度
        p_avg = refined_proto.mean(dim=0).unsqueeze(0)  # (1, C_proto)

        v_salience = self.player_salience(p_avg).view(1, C, 1, 1)
        v_consistency = self.player_consistency(p_avg).view(1, C, 1, 1)

        f_v1 = f_low * torch.sigmoid(v_salience)
        f_v2 = f_low * torch.sigmoid(v_consistency)
        f_joint = f_low * torch.sigmoid(v_salience + v_consistency)

        marginal_salience = f_joint - f_v2
        marginal_consistency = f_joint - f_v1

        refined_feat = self.coalition_weights[0, 0] * marginal_salience + \
                       self.coalition_weights[0, 1] * marginal_consistency

        mask = self.spatial_gate(refined_feat)
        out = f_low + self.out_proj(refined_feat * mask)

        return out


class TriExpertDispatcher(nn.Module):
    def __init__(self, proto_c=512, low_c=64, mid_c=128, high_c=256):
        super().__init__()
        # 针对不同层级的特征维度，设计独立的专家投影器
        # 每一层都会从原始原型中学习到不同的“博弈策略”
        self.to_low = nn.Sequential(
            nn.Linear(proto_c, low_c),
            nn.LayerNorm(low_c),
            nn.ReLU()
        )
        self.to_mid = nn.Sequential(
            nn.Linear(proto_c, mid_c),
            nn.LayerNorm(mid_c),
            nn.ReLU()
        )
        self.to_high = nn.Sequential(
            nn.Linear(proto_c, high_c),
            nn.LayerNorm(high_c),
            nn.ReLU()
        )

    def forward(self, refined_proto):
        """
        refined_proto: 经过语义培训模块优化后的原型 (K, 512)
        """
        # 分离出三组面向不同层级的专家向量
        p_low = self.to_low(refined_proto)  # (K, 64)
        p_mid = self.to_mid(refined_proto)  # (K, 128)
        p_high = self.to_high(refined_proto)  # (K, 256)

        return p_low, p_mid, p_high
class DecoupledPrototypeExpert(nn.Module):
    def __init__(self, in_c, out_c, num_proto=16, temp=0.07):
        super().__init__()
        self.num_proto = num_proto
        self.temp = temp

        # 显式定义两组原型参数
        self.geo_prototypes = nn.Parameter(torch.randn(num_proto // 2, in_c))
        self.sem_prototypes = nn.Parameter(torch.randn(num_proto // 2, in_c))

        self.proj = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # B, C, N

        # 合并原型 (K, C)
        all_protos = torch.cat([self.geo_prototypes, self.sem_prototypes], dim=0)

        # L2归一化计算余弦相似度，增强对比学习效果
        x_norm = F.normalize(x_flat, dim=1)
        proto_norm = F.normalize(all_protos, dim=1)

        # sim: (B, K, N)
        sim = torch.einsum('bcn,kc->bkn', x_norm, proto_norm) / self.temp
        attn = torch.softmax(sim, dim=1)

        # 重构 F4 特征
        out = torch.einsum('bkn,kc->bcn', attn, all_protos)
        out = out.view(B, C, H, W)

        # 返回：优化后的F4, 几何原型, 语义原型
        return self.proj(out), self.geo_prototypes, self.sem_prototypes


# ==========================================================
# 2. 原型引导优化模块 (Prototype-Guided Refiner)
# 使用 F4 的专家去优化 F1/F2/F3
# ==========================================================

# 1. 拓扑感知过滤器 (Topology-Aware Filter)
# ==========================================================
class TopologyFilter(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # 利用非对称卷积模拟拓扑持久性，捕捉轨道线和裂纹的线性连通性 [cite: 2026-03-24]
        self.topo_h = nn.Conv2d(in_c, in_c, kernel_size=(1, 7), padding=(0, 3), groups=in_c)
        self.topo_v = nn.Conv2d(in_c, in_c, kernel_size=(7, 1), padding=(3, 0), groups=in_c)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c * 2, kernel_size=1),  # 改用 1x1 卷积
            nn.Sigmoid()
        )

    def forward(self, f4):
        # 增强线性拓扑结构
        f_topo = self.topo_h(f4) + self.topo_v(f4)
        f_refined = f4 + f_topo

        # 分离几何流与语义流
        weights = self.gate(f_refined)
        w_geo, w_sem = weights.chunk(2, dim=1)

        f_geo = f_refined * w_geo.view(-1, f4.size(1), 1, 1)
        f_sem = f_refined * w_sem.view(-1, f4.size(1), 1, 1)

        return f_geo, f_sem


# ==========================================================
# 2. 几何分支：各向异性扩散优化 (Geometric Optimizer)
# ==========================================================



# ==========================================================
# 3. 语义分支：因果变分优化 (Advanced Semantic Optimizer)
# ==========================================================
class AdvancedSemanticOptimizer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # 潜在分布建模 (VIB) [cite: 2026-03-18]
        self.fc_mu = nn.Conv2d(in_c, in_c, 1)
        self.fc_var = nn.Conv2d(in_c, in_c, 1)

        # 因果背景字典：用于 P-intervention 去混淆 [cite: 2026-03-18]
        self.context_dict = nn.Parameter(torch.randn(1, in_c, 1, 1))

        self.causal_gate = nn.Sequential(
            nn.Conv2d(in_c * 2, in_c, 1),
            nn.Sigmoid()
        )

    def forward(self, f_sem):
        B, C, H, W = f_sem.shape

        # 变分采样：增强对光照变化的鲁棒性 [cite: 2026-03-18]
        mu = self.fc_mu(f_sem)
        log_var = self.fc_var(f_sem)
        std = torch.exp(0.5 * log_var)
        f_v = mu + torch.randn_like(std) * std if self.training else mu

        # 因果去混淆：减去全局背景混淆项 [cite: 2026-03-18]
        global_ctx = self.context_dict.expand(B, -1, H, W)
        gate = self.causal_gate(torch.cat([f_v, global_ctx], dim=1))

        f_final = f_v * (1 - gate)
        return f_final, mu, log_var


class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_backbone = DINOv3_vit()


        self.decoder = Decoer()
        self.topo_filter = TopologyFilter(512)
        self.sem_expert = AdvancedSemanticOptimizer(512)

        # 解耦原型专家
        self.proto_expert = DecoupledPrototypeExpert(512, 512, num_proto=16)

        # 跨层校准优化器 (Refiners)
        self.refiner3 = MVGR_Refiner(256, 256)
        self.refiner2 = MVGR_Refiner(128, 128)
        self.refiner1 = MVGR_Refiner(64, 64)
        self.OT = TriExpertDispatcher()
        self.sa1 = SalHead(64)
        self.sa2 = SalHead(128)
        self.sa3 = SalHead(256)
        self.sa4 = SalHead(512)
    def forward(self, x_rgb):
        # rgb  rgb1, rgb2 ,rgb3, rgb4   ([2, 76, 64, 64])  ([2, 152, 32, 32])  ([2, 336, 16, 16])  ([2, 672, 8, 8])
        f1, f2, f3, f4 = self.rgb_backbone(x_rgb)

        f4_geo, f4_sem = self.topo_filter(f4)

        f4_final, geo_p, sem_p = self.proto_expert(f4_sem+f4_geo)

        rp  =torch.cat([geo_p, sem_p],dim=0)

        f4_sem_opt, mu, log_var = self.sem_expert(rp.unsqueeze(-1).unsqueeze(-1))
        f4_sem_opt = f4_sem_opt.squeeze(-1).squeeze(-1)
        # 3. 使用 F4 的专家去校准优化前三层
        low,mid,high = self.OT(f4_sem_opt)
        # 4. 跨层校准：用高层专家优化底层细节
        f3_opt = self.refiner3(f3, high)
        f2_opt = self.refiner2(f2, mid)
        f1_opt = self.refiner1(f1, low)

        # 4. 进入解码器
        p1, p2, p3, p4 = self.decoder(f1_opt, f2_opt, f3_opt, f4_final)

        pre1 = F.interpolate(self.sa1(p1), 256, mode="bilinear", align_corners=True)
        pre2 = F.interpolate(self.sa2(p2), 256, mode="bilinear", align_corners=True)
        pre3 = F.interpolate(self.sa3(p3), 256, mode="bilinear", align_corners=True)
        pre4 = F.interpolate(self.sa4(p4), 256, mode="bilinear", align_corners=True)



        return F.sigmoid(pre1), F.sigmoid(pre2), F.sigmoid(pre3),F.sigmoid( pre4),mu, log_var,geo_p,sem_p, f1, f2, f3, f4,f1_opt,f2_opt,f3_opt,f4_final
        # return  f1, f2, f3, f4
class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0))

    def forward(self, x):
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.BatchNorm2d(depth),
                                           nn.ReLU())
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
                                           nn.BatchNorm2d(depth),
                                           nn.ReLU())
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 12), dilation=(1, 12)),
                                            # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(12, 0), dilation=(12, 1)),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
                                            # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 18), dilation=(1, 18)),
                                            # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(18, 0), dilation=(18, 1)),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 4, 4*depth, 1),
                                             nn.BatchNorm2d(4*depth),
                                             nn.ReLU())

    def forward(self, x):

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        net = net + x

        return net
class Decoer(nn.Module):
    def __init__(self):
        super().__init__()

        # self.aspp5 = ASPP(in_channel=768, depth=192)
        self.aspp4 = ASPP(in_channel=512, depth=128)
        self.aspp3 = ASPP(in_channel=256, depth=64)
        self.aspp2 = ASPP(in_channel=128, depth=32)
        self.aspp1 = ASPP(in_channel=64, depth=16)


        self.up2 = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
        self.conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv3 = nn.Sequential(nn.Conv2d(768, 256, 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv2 = nn.Sequential(nn.Conv2d(384, 128, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv1 = nn.Sequential(nn.Conv2d(192, 64, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))

    def forward(self, f1, f2, f3, f4):

        cat4 = self.aspp4(self.up2(f4))
        cat3 = torch.cat((f3, cat4), dim=1)
        cat3 = self.aspp3(self.conv3(cat3))
        cat2 = torch.cat((f2, cat3), dim=1)
        # print("cat2",cat2.shape)
        cat2 = self.aspp2(self.conv2(cat2))
        cat1 = torch.cat((cat2, f1), dim=1)
        cat1 = self.aspp1(self.conv1(cat1))

        return cat1,cat2,cat3,cat4











if __name__ == '__main__':

    input_rgb = torch.randn(1, 3, 256, 256).cuda()
    input_d = torch.randn(1, 1, 256, 256)
    # x = torch.randn(2, 512, 8, 8)
    # x1 = torch.randn(2, 512, 8, 8)
    net = PNet().cuda()

    out=net(input_rgb)
    from FLOP import CalParams2

    CalParams2(net, input_rgb)
    print('Total params % .2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    for out in out:
        print("out1", out.shape)


# FLOPs: 1.104T
# Params: 495.197M
#  ####################
# Total params  343.84M

