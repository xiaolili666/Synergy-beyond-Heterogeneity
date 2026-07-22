import torch
import torch.nn as nn
import torch.nn.functional as F


class QSDModule(nn.Module):
    """
    基于查询的软蒸馏模块 (参考 GKD 论文)
    用于让学生模型作为 Query，去检索教师的知识池
    """

    def __init__(self, student_dim, teacher_dim, inner_dim=512):
        super().__init__()
        self.query_conv = nn.Conv2d(student_dim, inner_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(teacher_dim, inner_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(teacher_dim, inner_dim, kernel_size=1)
        self.out_conv = nn.Conv2d(inner_dim, student_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f_st, f_tea):
        # f_st: [B, C_st, H, W], f_tea: [B, C_tea, H, W]
        B, _, H, W = f_st.shape

        q = self.query_conv(f_st).view(B, -1, H * W).permute(0, 2, 1)  # [B, N, D]
        k = self.key_conv(f_tea).view(B, -1, H * W)  # [B, D, N]
        v = self.value_conv(f_tea).view(B, -1, H * W).permute(0, 2, 1)  # [B, N, D]

        # 计算注意力/亲和力矩阵
        attn = self.softmax(torch.bmm(q, k) * (q.shape[-1] ** -0.5))  # [B, N, N]

        # 检索教师知识
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, -1, H, W)
        return self.out_conv(out)


class Stage1Distiller(nn.Module):
    """
    第一阶段蒸馏模块：并行上课，传授通用知识

    输入:
        f_student: 学生特征 [B, C_st, H, W]
        f_teacher: 教师特征 [B, C_tea, H, W]

    输出:
        loss: 蒸馏损失（QSD损失 + 亲和力损失）
    """

    def __init__(self, student_dim, teacher_dim, inner_dim=512,
                 affinity_weight=0.1, qsd_weight=1.0):
        """
        Args:
            student_dim: 学生特征维度
            teacher_dim: 教师特征维度
            inner_dim: QSD模块内部维度
            affinity_weight: 亲和力损失的权重
            qsd_weight: QSD损失的权重
        """
        super().__init__()
        self.qsd = QSDModule(student_dim, teacher_dim, inner_dim)
        self.affinity_weight = affinity_weight
        self.qsd_weight = qsd_weight

    def _compute_affinity(self, f):
        """计算空间亲和力矩阵"""
        f = F.normalize(f, p=2, dim=1)
        b, c, h, w = f.shape
        f = f.view(b, c, -1)
        return torch.bmm(f.transpose(1, 2), f)  # [B, N, N]

    def _get_affinity_loss(self, f1, f2):
        """计算两个特征图之间的亲和力损失（求同存异）"""
        aff_1 = self._compute_affinity(f1)
        aff_2 = self._compute_affinity(f2)
        return F.mse_loss(aff_1, aff_2)

    def forward(self, f_student, f_teacher, return_details=False):
        """
        Args:
            f_student: 学生特征 [B, C_st, H, W]
            f_teacher: 教师特征 [B, C_tea, H, W]
            return_details: 是否返回详细损失信息

        Returns:
            loss: 总蒸馏损失
            details (可选): 包含各项损失的字典
        """
        # 1. QSD 损失：学生向教师检索知识
        f_refined = self.qsd(f_student, f_teacher)
        loss_qsd = F.mse_loss(f_student, f_refined)

        # 2. Affinity 共识损失：强制学生模仿教师对"物体实体"的边界认知
        loss_affinity = self._get_affinity_loss(f_student, f_teacher)

        # 总损失
        total_loss = self.qsd_weight * loss_qsd + self.affinity_weight * loss_affinity

        if return_details:
            details = {
                'loss_qsd': loss_qsd.item(),
                'loss_affinity': loss_affinity.item(),
                'total_loss': total_loss.item()
            }
            return total_loss, details

        return total_loss


class DualTeacherStage1Distiller(nn.Module):
    """
    双教师第一阶段蒸馏模块

    输入:
        f_student: 学生特征 [B, C_st, H, W]
        f_teacher1: 教师1特征 [B, C_t1, H, W]
        f_teacher2: 教师2特征 [B, C_t2, H, W]

    输出:
        loss: 总蒸馏损失（两个教师的QSD损失 + 双教师亲和力损失）
    """

    def __init__(self, student_dim, teacher1_dim, teacher2_dim,
                 inner_dim=512, affinity_weight=0.1, qsd_weight=1.0):
        super().__init__()
        # 两个独立的蒸馏器
        self.distiller1 = Stage1Distiller(
            student_dim, teacher1_dim, inner_dim,
            affinity_weight=affinity_weight, qsd_weight=qsd_weight
        )
        self.distiller2 = Stage1Distiller(
            student_dim, teacher2_dim, inner_dim,
            affinity_weight=affinity_weight, qsd_weight=qsd_weight
        )

        # 双教师之间的亲和力损失权重
        self.teacher_affinity_weight = affinity_weight

    def _compute_affinity(self, f):
        """计算空间亲和力矩阵"""
        f = F.normalize(f, p=2, dim=1)
        b, c, h, w = f.shape
        f = f.view(b, c, -1)
        return torch.bmm(f.transpose(1, 2), f)

    def _get_affinity_loss(self, f1, f2):
        """计算两个特征图之间的亲和力损失"""
        aff_1 = self._compute_affinity(f1)
        aff_2 = self._compute_affinity(f2)
        return F.mse_loss(aff_1, aff_2)

    def forward(self, f_student, f_teacher1, f_teacher2, return_details=False):
        """
        Args:
            f_student: 学生特征 [B, C_st, H, W]
            f_teacher1: 教师1特征 [B, C_t1, H, W]
            f_teacher2: 教师2特征 [B, C_t2, H, W]
            return_details: 是否返回详细损失信息

        Returns:
            loss: 总蒸馏损失
        """
        # 分别计算两个教师的蒸馏损失
        loss1 = self.distiller1(f_student, f_teacher1)
        loss2 = self.distiller2(f_student, f_teacher2)

        # 双教师之间的共识损失（亲和力）
        loss_teacher_affinity = self._get_affinity_loss(f_teacher1, f_teacher2)

        # 总损失
        total_loss = loss1 + loss2 + self.teacher_affinity_weight * loss_teacher_affinity

        if return_details:
            # 获取详细损失
            _, details1 = self.distiller1(f_student, f_teacher1, return_details=True)
            _, details2 = self.distiller2(f_student, f_teacher2, return_details=True)

            details = {
                'teacher1_qsd': details1['loss_qsd'],
                'teacher1_affinity': details1['loss_affinity'],
                'teacher2_qsd': details2['loss_qsd'],
                'teacher2_affinity': details2['loss_affinity'],
                'teacher_affinity': loss_teacher_affinity.item(),
                'total_loss': total_loss.item()
            }
            return total_loss, details

        return total_loss

class BoundaryConstrainedDistillLoss(nn.Module):
    """
    借鉴 DIKDNet 思想的边界约束蒸馏损失
    用于将 Depth 视图的几何先验蒸馏到 Backbone 的底层特征中
    """
    def __init__(self, margin=0.7):
        super(BoundaryConstrainedDistillLoss, self).__init__()
        self.margin = margin

    def forward(self, feat_student, feat_teacher):
        """
        feat_student: 主网络特征 (例如 256 维度层), shape: [B, C, H, W]
        feat_teacher: Depth 视图特征, shape: [B, C, H, W]
        """
        # 1. 确保特征尺寸一致 (如果 Depth 特征尺寸不同，需进行插值)
        if feat_student.shape[2:] != feat_teacher.shape[2:]:
            feat_teacher = F.interpolate(feat_teacher, size=feat_student.shape[2:], mode='bilinear', align_corners=True)

        # 2. 计算像素级的余弦相似度
        # 在通道维度 (C) 计算相似度
        similarity = F.cosine_similarity(feat_student, feat_teacher, dim=1) # shape: [B, H, W]

        # 3. 边界约束逻辑 (Boundary Constraint)
        # DIKDNet 核心：loss = max(0, margin - similarity)
        # 只有当相似度低于 margin 时才计算 Loss
        loss = torch.clamp(self.margin - similarity, min=0)

        # 4. 返回均值损失
        return loss.mean()
# 使用示例
if __name__ == "__main__":
    # 假设特征维度
    # B, C_st, C_t1, C_t2, H, W = 4, 256, 1024, 256, 32, 32

    # 创建随机特征
    f_student = torch.randn(2, 64, 64, 64)
    f_teacher1 = torch.randn(2, 256, 16, 16)
    f_teacher2 = torch.randn(2, 256, 16, 16)

    # 单教师蒸馏
    # distiller = Stage1Distiller(student_dim=256, teacher_dim=256, inner_dim=512)
    # loss = distiller(f_student, f_teacher1)
    # print(f"Single teacher loss: {loss.item():.4f}")
    #
    # # 带详细信息的单教师蒸馏
    # loss, details = distiller(f_student, f_teacher1, return_details=True)
    # print(f"Single teacher details: QSD={details['loss_qsd']:.4f}, Affinity={details['loss_affinity']:.4f}")

    # 双教师蒸馏
    dual_distiller = DualTeacherStage1Distiller(
        student_dim=64,
        teacher1_dim=256,
        teacher2_dim=256,
        inner_dim=512,
        affinity_weight=0.1
    )
    loss = dual_distiller(f_student, f_teacher1, f_teacher2)
    print(f"Dual teacher loss: {loss.item():.4f}")

    # 带详细信息的双教师蒸馏
    loss, details = dual_distiller(f_student, f_teacher1, f_teacher2, return_details=True)
    print(f"Dual teacher details: T1_QSD={details['teacher1_qsd']:.4f}, "
          f"T1_Aff={details['teacher1_affinity']:.4f}, "
          f"Teacher_aff={details['teacher_affinity']:.4f}")