import torch
from torch import optim, nn
from torch.autograd import Variable
from Train_Text_Val.Config_loss.dataset_txt1 import get_loader
from Train_Text_Val.Config_loss.adjuse_lr import adjust_learning_rate
from Train_Text_Val.Config_loss.basic_loss import *
import logging
import math
import os
from Train_Text_Val.Train._Text_TXT_ import val_usod_
from Train_Text_Val.Coll_loss.kl_loss import KLDLoss
from Train_Text_Val.Coll_loss.change import OFD
from Result.logKD.FKD import DualTeacherStage1Distiller   #BoundaryConstrainedDistillLoss
from Result.logKD.FKD import Stage1Distiller
from Result.logKD.FKD import BoundaryConstrainedDistillLoss
# 模型加载
from P3.newmodel.MoeNetf import PNet as T1
from P3.Teacher1.old.Teacher.SAM_T1 import Net as T2
from P3.Student.student7 import student_single as S

model1_name = "Student_KD_4_Fb"
bl = BoundaryConstrainedDistillLoss().cuda()
KLD = KLDLoss().cuda()
dual_distiller256 = DualTeacherStage1Distiller(
    student_dim=256,
    teacher1_dim=256,
    teacher2_dim=256,
    inner_dim=512,
    affinity_weight=0.1
).cuda()
dual_distiller64 = DualTeacherStage1Distiller(
    student_dim=64,
    teacher1_dim=64,
    teacher2_dim=64,
    inner_dim=512,
    affinity_weight=0.1
).cuda()
dual_distiller128 = DualTeacherStage1Distiller(
    student_dim=128,
    teacher1_dim=128,
    teacher2_dim=128,
    inner_dim=512,
    affinity_weight=0.1
).cuda()
dual_distiller512 = DualTeacherStage1Distiller(
    student_dim=512,
    teacher1_dim=512,
    teacher2_dim=512,
    inner_dim=512,
    affinity_weight=0.1
).cuda()
#
distiller256 = Stage1Distiller(student_dim=256, teacher_dim=256, inner_dim=512).cuda()
distiller64 = Stage1Distiller(student_dim=64, teacher_dim=64, inner_dim=512).cuda()
distiller128 = Stage1Distiller(student_dim=128, teacher_dim=128, inner_dim=512).cuda()
distiller512 = Stage1Distiller(student_dim=512, teacher_dim=512, inner_dim=512).cuda()
# mid64 = OFD(64,64).cuda()
# mid128 = OFD(128,128).cuda()
# mid256 = OFD(256,256).cuda()
# mid512 = OFD(512,512).cuda()
def apply_balanced_projection(shared_params, grads_sem, grads_geo):
    """
    平衡投影：处理语义梯度和几何梯度之间的冲突
    """
    for p, gs, gg in zip(shared_params, grads_sem, grads_geo):
        # 跳过无效梯度
        if gs is None or gg is None:
            continue

        # 确保 p.grad 不为 None
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)

        # 展平为一维向量计算点积
        gs_flat = gs.view(-1)
        gg_flat = gg.view(-1)

        dot_p = torch.dot(gs_flat, gg_flat)

        if dot_p < 0:  # 方向相反时才处理冲突
            # 计算范数，添加 epsilon 防止溢出
            norm_s = torch.dot(gs_flat, gs_flat) + 1e-4
            norm_g = torch.dot(gg_flat, gg_flat) + 1e-4

            # 对称投影：互相减去对方方向上的投影分量
            gs_corr = gs - (dot_p / norm_g) * gg
            gg_corr = gg - (dot_p / norm_s) * gs
            p.grad.data.copy_(gs_corr + gg_corr)
        else:
            # 方向一致时直接相加
            p.grad.data.copy_(gs + gg)


def train_net(args):
    # 设置日志
    logging.basicConfig(filename=args.save_model_dir + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # ==================== 加载教师模型 ====================
    # 教师模型1
    net2 = T1().to(device)
    net2 = nn.DataParallel(net2)
    net2.load_state_dict(torch.load("/media/pc12/data/LYQ/SAM/COD_LYQ/Result/logMoE1/MoERSDD-10.pth"), False)
    net2.eval()  # 设置为评估模式

    # 教师模型2
    net3 = T2().to(device)
    net3 = nn.DataParallel(net3)
    net3.load_state_dict(torch.load("/media/pc12/data/LYQ/SAM/COD_LYQ/Result/logT1/MoERSDD-232.pth"), False)
    net3.eval()  # 设置为评估模式

    # ==================== 加载学生模型 ====================
    net1 = S().to(device)
    net1 = nn.DataParallel(net1)

    # 设置参数分组
    hungry_param = []  # 高学习率参数
    full_param = []  # 低学习率参数

    for k, v in net1.named_parameters():
        if "image_encoder" in k:
            if "feature_adaptation" in k or "adapter" in k or "underwater_enhancer" in k or \
                    "norm3" in k or "lora" in k or "prompt_generator" in k or "prompt_tuning" in k or "neck" in k:
                hungry_param.append(v)
            else:
                v.requires_grad = False
        elif "backbone" in k:
            full_param.append(v)
        elif "mask_decoder" in k or "prompt_encoder" in k:
            hungry_param.append(v)
        else:
            hungry_param.append(v)

    # 优化器
    optimizer1 = torch.optim.AdamW(
        [{"params": hungry_param, "lr": args.lr},
         {"params": full_param, "lr": args.lr * 0.1}],
        weight_decay=1e-5)

    # ==================== 加载数据集 ====================
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')
    val_dataset = get_loader(args.test_paths, args.data_root, args.img_size, mode='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=True)

    print(f'''
        Starting training:
            Epochs: {args.epochs}
            Train steps per epoch: {len(train_loader)}
            Batch size: {args.batch_size}
            Learning rate: {args.lr}
            Training size: {len(train_loader.dataset)}
            Validation size: {len(val_loader.dataset)}
        ''')

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    # ==================== 训练循环 ====================
    whole_iter_num = 0

    for epoch in range(args.epochs):
        print(f'\n{"=" * 50}')
        print(f'Starting epoch {epoch + 1}/{args.epochs}')
        print(f'Current learning rate: {optimizer1.param_groups[0]["lr"]:.6f}')
        print(f'{"=" * 50}')

        net1.train()
        epoch_loss = 0.0

        for i, data_batch in enumerate(train_loader):
            # 加载数据
            images, depths, label_224, label_14, label_28, label_56, label_112, \
                contour_224, contour_14, contour_28, contour_56, contour_112, bbox_tensor = data_batch

            images = Variable(images.to(device))
            depths = Variable(depths.to(device))
            label_224 = Variable(label_224.to(device))
            contour_224 = Variable(contour_224.to(device))
            bbox_tensor = Variable(bbox_tensor.to(device))

            # ===== 教师模型前向传播（无梯度） =====
            with torch.no_grad():
                out2 = net2(images)  # 教师1输出
                out3 = net3(images, depths)  # 教师2输出

            # ===== 学生模型前向传播 =====
            outputs_saliency = net1(images)
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11= outputs_saliency[:12]

            # ===== 1. 计算语义蒸馏损失（教师1） =====
            # 浅层特征蒸馏
            loss_semanticsp = (KLD(d0, out2[0]) + KLD(d1, out2[1]) +
                               KLD(d2, out2[2]) + KLD(d3, out2[3]))
            # 深层特征蒸馏
            loss_semanticsb = (KLD(d4, out2[8]) + KLD(d5, out2[9]) +
                               KLD(d6, out2[10]) + KLD(d7, out2[11]))
            loss_semantics = loss_semanticsp + loss_semanticsb   #out2[12])

            # ===== 2. 计算几何蒸馏损失（教师2） =====
            # 浅层特征蒸馏
            loss_geometryp = (KLD(d0, out3[0]) + KLD(d1, out3[1]) +
                              KLD(d2, out3[2]) + KLD(d3, out3[3]))
            # 深层特征蒸馏
            loss_geometryb = (KLD(d4, out3[8]) + KLD(d5, out3[9]) +
                              KLD(d6, out3[10]) + KLD(d7, out3[11]))

            loss_geometry = loss_geometryp + loss_geometryb

            # ===== 3. 计算任务损失（GT监督） =====
            loss_task = (calc_loss(d0, label_224) + iou_loss(d0, label_224) +
                         calc_loss(d1, label_224) + iou_loss(d1, label_224) +
                         calc_loss(d2, label_224) + iou_loss(d2, label_224) +
                         calc_loss(d3, label_224) + iou_loss(d3, label_224))

            #中间特征
            #第一阶段  单独
            # loss1 = dual_distiller64(d8,   out3[8], out2[12])
            # loss2 = dual_distiller128(d9,  out3[9], out2[13])
            # loss3 = dual_distiller256(d10, out3[10],out2[14])
            # loss4 = dual_distiller512(d11, out3[11],out2[15])
            lossb = bl(d10, out3[12])

            # # T1_S
            # #d
            # loss1 = distiller64(d8, out2[12])
            # loss2 = distiller128(d9, out2[13])
            # # r
            # loss3 = distiller256(d10, out2[14])
            # loss4 = distiller512(d11, out2[15])
            # # loss1 = dual_distiller256(d10,out2[14],out3[12])
            #
            #
            # # T2_S
            # loss5 = distiller64(d8, out3[12])
            # loss6 = distiller128(d9, out3[13])
            # loss7 = distiller256(d10, out3[14])
            # loss8 = distiller512(d11, out3[15])
            # ===== 梯度手术：处理多任务冲突 =====
            optimizer1.zero_grad()

            # 步骤1: 计算语义蒸馏梯度
            loss_semantics.backward(retain_graph=True)
            grads_sem = [p.grad.clone() if p.grad is not None else None
                         for p in hungry_param]

            # 步骤2: 清空梯度，计算几何蒸馏梯度
            optimizer1.zero_grad()
            loss_geometry.backward(retain_graph=True)
            grads_geo = [p.grad.clone() if p.grad is not None else None
                         for p in hungry_param]

            # 步骤3: 应用平衡投影
            # 先确保所有 hungry_param 的 grad 已初始化
            for p in hungry_param:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)

            apply_balanced_projection(hungry_param, grads_sem, grads_geo)
            loss_task = loss_task+lossb#+loss5+loss6+loss7+loss8  loss1+loss2loss3loss4+
            # 步骤4: 添加任务损失梯度
            loss_task.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net1.parameters(), max_norm=1.0)

            # 更新参数
            optimizer1.step()

            # 记录损失

            total_loss = loss_task + loss_semantics + loss_geometry
            epoch_loss += total_loss.item()

            # 打印进度
            if (i + 1) % 20 == 0:
                print(f'  Iter [{i + 1}/{len(train_loader)}] '
                      f'Loss_total: {total_loss.item():.4f} | '
                      f'Loss_task: {loss_task.item():.4f} | '
                      f'Loss_sem: {loss_semantics.item():.4f} | '
                      f'Loss_geo: {loss_geometry.item():.4f}')

            whole_iter_num += 1

        # 打印 epoch 平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f'\nEpoch {epoch + 1} average loss: {avg_loss:.4f}')

        # ==================== 保存模型 ====================
        # 保存最终模型
        torch.save(net1.state_dict(),
                   args.save_model_dir + model1_name + '_final.pth')

        # 根据断点调整学习率
        if epoch == args.stepvalue1 or epoch == args.stepvalue2:
            optimizer1 = adjust_learning_rate(optimizer1, decay_rate=args.lr_decay_gamma)
            print(f'Learning rate adjusted to {optimizer1.param_groups[0]["lr"]:.6f}')

        # ==================== 验证阶段 ====================
        print(f'\nStarting validation for epoch {epoch + 1}...')
        val_result = val_usod_(args, epoch, net1, model1_name, val_loader)
        logging.info(f'Epoch {epoch}: ' + val_result["DualTeacher_RGB_Net_log"])

        print(f'Validation SM score: {val_result["sm"]:.4f}')

        # 保存最佳模型（SM > 0.85）
        if val_result['sm'] >= 0.850:
            save_path = args.save_model_dir + model1_name + f"_RSDD_epoch{epoch}_sm{val_result['sm']:.4f}.pth"
            torch.save(net1.state_dict(), save_path)
            print(f'Best model saved: {save_path}')


if __name__ == "__main__":
    from Train_Text_Val.Config_loss.d_training_config_TXT import args

    train_net(args)