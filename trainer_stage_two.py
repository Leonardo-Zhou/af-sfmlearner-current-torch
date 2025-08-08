from __future__ import absolute_import, division, print_function

import time
import json
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 注意：建议使用torch.utils.tensorboard替代tensorboardX
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# 导入项目特定模块
import datasets
import networks
from utils import *
from layers import *


class Trainer(object):
    """
    第二阶段训练器类
    结合光流估计和变换网络进行自监督深度估计
    """
    
    def __init__(self, options):
        """
        初始化训练器
        
        参数:
            options: 训练配置参数对象
        """
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # 检查图像尺寸是否为32的倍数（网络下采样要求）
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        # 初始化模型字典和训练参数列表
        self.models = {}
        self.parameters_to_train = []

        # 设置计算设备（GPU/CPU）
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        # 网络配置参数
        self.num_scales = len(self.opt.scales)  # 多尺度数量，通常为4
        self.num_input_frames = len(self.opt.frame_ids)  # 输入帧数，通常为3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # 确保帧ID从0开始
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # 是否使用位姿网络
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        
        # 如果使用立体训练，添加立体帧标记
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # 初始化深度估计网络
        self._init_depth_networks()
        
        # 初始化位姿网络（如果需要）
        if self.use_pose_net:
            self._init_pose_networks()
            
        # 初始化预测掩码网络（可选）
        if self.opt.predictive_mask:
            self._init_predictive_mask_network()
            
        # 初始化优化器和学习率调度器
        self._init_optimizer()
        
        # 加载预训练权重（如果指定）
        if self.opt.load_weights_folder is not None:
            self.load_model()

        # 打印训练信息
        self._print_training_info()
        
        # 初始化数据集和数据加载器
        self._init_data_loaders()
        
        # 初始化损失函数和相关模块
        self._init_loss_functions()
        
        # 保存训练配置
        self.save_opts()

    def _init_depth_networks(self):
        """初始化深度估计相关的网络"""
        # 深度编码器：使用ResNet提取图像特征
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, 
            self.opt.weights_init == "pretrained"
        )
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # 深度解码器：从特征生成深度图
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, 
            self.opt.scales
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 位置编码器：用于外观流估计（Appearance Flow）
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, 
            self.opt.weights_init == "pretrained", 
            num_input_images=2  # 输入两张图像
        )
        self.models["position_encoder"].to(self.device)

        # 位置解码器：生成外观流场（解决亮度不一致问题）
        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, 
            self.opt.scales
        )
        self.models["position"].to(self.device)

        # 变换编码器：用于图像变换估计
        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, 
            self.opt.weights_init == "pretrained", 
            num_input_images=2
        )
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        # 变换解码器：生成图像变换参数
        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, 
            self.opt.scales
        )
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

    def _init_pose_networks(self):
        """初始化位姿估计网络"""
        if self.opt.pose_model_type == "separate_resnet":
            # 独立的位姿编码器
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames
            )
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            # 位姿解码器
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2
            )

        elif self.opt.pose_model_type == "shared":
            # 共享编码器
            self.models["pose"] = networks.PoseDecoder(
                self.models["encoder"].num_ch_enc, 
                self.num_pose_frames
            )

        elif self.opt.pose_model_type == "posecnn":
            # PoseCNN架构
            self.models["pose"] = networks.PoseCNN(
                self.num_input_frames if self.opt.pose_model_input == "all" else 2
            )

        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

    def _init_predictive_mask_network(self):
        """初始化预测掩码网络（用于处理遮挡）"""
        assert self.opt.disable_automasking, \
            "When using predictive_mask, please disable automasking with --disable_automasking"

        self.models["predictive_mask"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, 
            self.opt.scales,
            num_output_channels=(len(self.opt.frame_ids) - 1)
        )
        self.models["predictive_mask"].to(self.device)
        self.parameters_to_train += list(self.models["predictive_mask"].parameters())

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, 
            self.opt.learning_rate
        )
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, 
            self.opt.scheduler_step_size, 
            gamma=0.1  # 学习率衰减因子
        )

    def _print_training_info(self):
        """打印训练相关信息"""
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

    def _init_data_loaders(self):
        """初始化数据集和数据加载器"""
        # 数据集映射
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # 加载训练和验证文件列表
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'

        # 计算总训练步数
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # 训练数据集和加载器
        train_dataset = self.dataset(
            self.opt.data_path, 
            train_filenames, 
            self.opt.height, 
            self.opt.width,
            self.opt.frame_ids, 
            4,  # 4个尺度
            is_train=True, 
            img_ext=img_ext
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.opt.batch_size, 
            shuffle=True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, 
            drop_last=True
        )
        
        # 验证数据集和加载器
        val_dataset = self.dataset(
            self.opt.data_path, 
            val_filenames, 
            self.opt.height, 
            self.opt.width,
            self.opt.frame_ids, 
            4, 
            is_train=False, 
            img_ext=img_ext
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.opt.batch_size, 
            shuffle=False,
            num_workers=1, 
            pin_memory=True, 
            drop_last=True
        )
        self.val_iter = iter(self.val_loader)

        # 初始化TensorBoard写入器
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

    def _init_loss_functions(self):
        """初始化损失函数和相关模块"""
        # SSIM损失（结构相似性）
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        # 空间变换器（用于图像扭曲）
        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        # 遮挡掩码计算模块
        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        # 深度反投影和3D投影模块
        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        # 深度评估指标名称
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", 
            "da/a1", "da/a2", "da/a3"
        ]

        # 打印数据集信息
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(self.train_loader.dataset), len(self.val_loader.dataset)))

    def set_train(self):
        """将所有网络设置为训练模式"""
        for model_name in self.models:
            self.models[model_name].train()

    def set_eval(self):
        """将所有网络设置为评估模式"""
        for model_name in self.models:
            self.models[model_name].eval()

    def train(self):
        """运行完整的训练流程"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """运行单个训练周期（包含训练和验证）"""
        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # 处理一个批次的数据
            outputs, losses = self.process_batch(inputs)

            # 反向传播和优化
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # 定期记录日志和验证
            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            
        # 更新学习率
        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """处理一个批次的数据，通过网络生成图像和损失"""
        # 将数据移到GPU上
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # 根据位姿模型类型选择处理方式
        if self.opt.pose_model_type == "shared":
            # 如果使用共享编码器（monodepthv1方法）
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # 否则只通过深度编码器处理frame_id=0的图像
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        # 预测掩码（可选）
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # 位姿估计
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs))

        # 生成预测图像
        self.generate_images_pred(inputs, outputs)
        
        # 计算损失
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features, disps):
        """预测单目序列中输入帧之间的位姿
        
        参数:
            inputs: 输入数据字典
            features: 图像特征
            disps: 视差图
            
        返回:
            outputs: 包含位姿预测结果的字典
        """
        outputs = {}
        
        if self.num_pose_frames == 2:
            # 准备位姿估计的输入特征
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            # 处理每个源帧
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # 准备输入对
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # 1. 光流估计（正向）
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    # 处理每个尺度的光流结果
                    for scale in self.opt.scales:
                        # 保存光流结果
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        
                        # 上采样到原始分辨率
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], 
                            [self.opt.height, self.opt.width], 
                            mode="bilinear", 
                            align_corners=False
                        )
                        
                        # 使用光流进行图像配准
                        outputs[("registration", scale, f_i)] = self.spatial_transform(
                            inputs[("color", f_i, 0)], 
                            outputs[("position", "high", scale, f_i)]
                        )

                        # 反向光流
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], 
                            [self.opt.height, self.opt.width], 
                            mode="bilinear", 
                            align_corners=False
                        )
                        
                        # 计算遮挡掩码
                        outputs[("occu_mask_backward", scale, f_i)], outputs[("occu_map_backward", scale, f_i)] = \
                            self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)]
                        )

                    # 2. 变换估计（图像增强）
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    # 处理每个尺度的变换结果
                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], 
                            [self.opt.height, self.opt.width], 
                            mode="bilinear", 
                            align_corners=False
                        )
                        
                        # 应用变换得到增强图像
                        outputs[("refined", scale, f_i)] = (
                            outputs[("transform", "high", scale, f_i)] * 
                            outputs[("occu_mask_backward", 0, f_i)].detach() + 
                            inputs[("color", 0, 0)]
                        )
                        outputs[("refined", scale, f_i)] = torch.clamp(
                            outputs[("refined", scale, f_i)], 
                            min=0.0, 
                            max=1.0
                        )

                    # 3. 位姿估计
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], 
                        translation[:, 0]
                    )
                    
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """生成重投影的彩色图像
        
        使用深度图和位姿将源图像重投影到目标视角
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            
            # 根据配置选择源尺度
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, 
                    [self.opt.height, self.opt.width], 
                    mode="bilinear", 
                    align_corners=False
                )

            # 将视差转换为深度
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            
            # 对每个源帧进行重投影
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # 处理PoseCNN的特殊情况
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], 
                        translation[:, 0] * mean_inv_depth[:, 0], 
                        frame_id < 0
                    )

                # 反投影深度到3D空间
                cam_points = self.backproject_depth[source_scale](
                    depth, 
                    inputs[("inv_K", source_scale)]
                )
                
                # 将3D点投影到源图像
                pix_coords = self.project_3d[source_scale](
                    cam_points, 
                    inputs[("K", source_scale)], 
                    T
                )

                outputs[("sample", frame_id, scale)] = pix_coords

                # 使用网格采样进行图像重投影
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border"
                )

                # 计算位置深度（用于可视化）
                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                    cam_points, 
                    inputs[("K", source_scale)], 
                    T
                )
                
    def compute_reprojection_loss(self, pred, target):
        """计算重投影损失（结合L1和SSIM）"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """计算多尺度损失
        
        包括：
        1. 重投影损失（光度一致性）
        2. 变换一致性损失
        3. 变换平滑损失
        4. 视差平滑损失
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0
            
            # 确定源尺度
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            # 计算每个源帧的损失
            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                
                # 重投影损失：增强图像与重投影图像的差异
                loss_reprojection += (
                    self.compute_reprojection_loss(
                        outputs[("color", frame_id, scale)], 
                        outputs[("refined", scale, frame_id)]
                    ) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()
                
                # 变换一致性损失：变换与配准的差异
                loss_transform += (
                    torch.abs(
                        outputs[("refined", scale, frame_id)] - 
                        outputs[("registration", 0, frame_id)].detach()
                    ).mean(1, True) * occu_mask_backward
                ).sum() / occu_mask_backward.sum()
                
                # 变换平滑损失
                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)], 
                    inputs[("color", 0, 0)], 
                    outputs[("registration", scale, frame_id)].detach(), 
                    occu_mask_backward
                )

            # 视差平滑损失
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            # 组合各种损失
            loss += loss_reprojection / 2.0
            loss += self.opt.transform_constraint * (loss_transform / 2.0)
            loss += self.opt.transform_smoothness * (loss_cvt / 2.0) 
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss

        # 平均多尺度损失
        total_loss /= self.num_scales
        losses["loss"] = total_loss
        
        return losses

    def val(self):
        """在单个批次上验证模型"""
        self.set_eval()
        
        try:
            # 使用Python 3风格的next()调用
            inputs = next(self.val_iter)
        except StopIteration:
            # 如果验证迭代器结束，重新创建
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            # 处理验证批次
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            
            # 验证期间不计算深度评估指标，只记录损失
            
        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:

                    writer.add_image(
                        "brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
            self.models[n].eval()
            for param in self.models[n].parameters():
                param.requires_grad = False

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")
