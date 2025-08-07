from __future__ import absolute_import, division, print_function

import time
import json
import datasets
import networks
import torch.optim as optim

from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


class Trainer:
    # 第一阶段训练光流估计网络。
    def __init__(self, options):
        self.opt = options  # 保存配置参数
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # 检查图像尺寸是否能被32整除（网络下采样要求）
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models = {}  # 存储所有网络模型
        self.parameters_to_train = []  # 需要训练的参数列表
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")  # 设备选择
        self.num_scales = len(self.opt.scales)  # 多尺度数量，默认为4，用于构建特征金字塔网络

        # 创建位置编码器（ResNet18，输入2张图像）
        # OFNet 中使用的位置编码器，用于提取图像特征
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
        self.models["position_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["position_encoder"].parameters())
        # 创建位置解码器（预测光流）
        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales)
        self.models["position"].to(self.device)
        self.parameters_to_train += list(self.models["position"].parameters())

        # Adam优化器配置
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        # 学习率调度器（每scheduler_step_size个epoch降低10倍）
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # 加载预训练权重（如果指定）
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # 数据集映射字典（仅支持SCARED数据集）
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # 读取训练/验证文件列表
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png'  # 图像扩展名

        # 计算总训练步数
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # 创建训练数据加载器
        train_dataset = self.dataset(
            data_path=self.opt.data_path,
            filenames=train_filenames,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.opt.frame_ids,
            num_scales=4,
            is_train=True,
            img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            
        # 创建验证数据加载器
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)  # 验证数据迭代器

        # 创建Tensorboard写入器
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # SSIM损失（结构相似性）
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        # 空间变换器（用于图像扭曲）
        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()  # 保存配置参数到JSON文件

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):  # 遍历所有epoch
            self.run_epoch()  # 运行单个epoch
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()  # 定期保存模型

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()  # 设置为训练模式

        for batch_idx, inputs in enumerate(self.train_loader):  # 遍历数据批次
            before_op_time = time.time()  # 记录开始时间

            outputs, losses = self.process_batch(inputs)  # 前向传播和损失计算

            self.model_optimizer.zero_grad()  # 梯度清零
            losses["loss"].backward()  # 反向传播
            self.model_optimizer.step()  # 更新参数

            duration = time.time() - before_op_time  # 计算批次处理时间

            phase = batch_idx % self.opt.log_frequency == 0  # 是否记录日志

            if phase:  # 定期记录日志和验证
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1  # 全局步数+1
            
        self.model_lr_scheduler.step()  # 更新学习率
        
    def process_batch(self, inputs):
        """
        处理一个批次的数据
        
        将输入数据通过网络前向传播，生成输出图像并计算损失。
        该方法是训练循环的核心部分。
        
        参数:
            inputs: 字典格式的输入数据，包含不同尺度的图像
            
        返回:
            outputs: 网络输出，包含光流和配准后的图像
            losses: 各种损失值
        """
        # 将所有输入数据迁移到指定设备（CPU/GPU）
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # 初始化输出字典
        outputs = {}
        # 预测光流（位置变换）
        outputs.update(self.predict_poses(inputs))
        # 计算损失
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """
        预测光流（位置变换）
        
        使用位置编码器和解码器网络预测相邻帧之间的光流，
        并基于预测的光流进行图像配准（图像扭曲）。
        
        参数:
            inputs: 输入数据字典，包含增强后的图像
            形如
            {
                # 原始图像（不同尺度和帧索引）
                ("color", 0, -1): PIL.Image,      # 参考帧原始图像
                ("color", -1, -1): PIL.Image,   # 前一帧原始图像  
                ("color", 1, -1): PIL.Image,    # 后一帧原始图像
                
                # 经过预处理和增强的张量
                ("color", 0, 0): torch.Tensor,    # 参考帧（目标分辨率）
                ("color", -1, 0): torch.Tensor,   # 前一帧（目标分辨率）
                ("color", 1, 0): torch.Tensor,    # 后一帧（目标分辨率）
                
                # 数据增强版本
                ("color_aug", 0, 0): torch.Tensor,    # 增强后的参考帧
                ("color_aug", -1, 0): torch.Tensor,   # 增强后的前一帧
                ("color_aug", 1, 0): torch.Tensor,    # 增强后的后一帧
                
                # 相机内参矩阵
                ("K", 0): torch.Tensor,           # 目标尺度的相机内参
                ("inv_K", 0): torch.Tensor,      # 目标尺度的逆相机内参
                
                # 可能包含的其他数据
                "stereo_T": torch.Tensor,         # 立体相机外参（如果使用立体数据）
                "depth_gt": torch.Tensor,         # 深度图真值（如果可用）
            }
        返回:
            outputs: 包含光流和配准后图像的字典
        """
        # 初始化输出字典
        outputs = {}
        
        # 提取所有帧的增强图像，用于光流预测
        # pose_feats[0]表示参考帧，pose_feats[-1/1]表示相邻帧
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        # 遍历所有非参考帧（frame_ids[1:]）
        for f_i in self.opt.frame_ids[1:]:
            # 构建输入：将当前帧与参考帧拼接
            # 输入形状：[batch_size, 6, H, W]（2张RGB图像拼接）
            position_input = [pose_feats[f_i], pose_feats[0]]
            
            # 通过位置编码器提取特征
            position_inputs = self.models["position_encoder"](torch.cat(position_input, 1))
            
            # 通过位置解码器预测光流
            outputs_0 = self.models["position"](position_inputs)

            # 处理不同尺度的光流和配准结果
            for scale in self.opt.scales:
                # 保存当前尺度的光流预测结果
                outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                
                # 将光流上采样到原始图像尺寸
                # 用于在原始分辨率下进行精确的图像配准
                outputs[("position", "high", scale, f_i)] = F.interpolate(
                    outputs[("position", scale, f_i)], 
                    [self.opt.height, self.opt.width], 
                    mode="bilinear", 
                    align_corners=False)
                
                # 使用预测的光流进行图像配准
                # 将相邻帧图像扭曲到参考帧视角
                outputs[("registration", scale, f_i)] = self.spatial_transform(
                    inputs[("color", f_i, 0)], 
                    outputs[("position", "high", scale, f_i)])

        return outputs

    def compute_reprojection_loss(self, pred, target):
        """计算重投影损失（L1 + SSIM）
        """
        abs_diff = torch.abs(target - pred)  # L1损失
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)  # SSIM损失
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss  # 加权组合

        return reprojection_loss


    def compute_losses(self, inputs, outputs):
        """计算总损失
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:  # 多尺度损失
            loss = 0
            loss_registration = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]  # 目标图像（参考帧）
            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                # 光流平滑度损失
                loss_registration += (get_smooth_registration(outputs[("position", scale, frame_id)]))
                # 配准损失（扭曲图像与目标图像的差异）
                registration_losses.append(
                    self.compute_reprojection_loss(outputs[("registration", scale, frame_id)], target))

            # 选择最小损失（处理遮挡）
            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            loss += self.opt.position_smoothness * (loss_registration / 2.0) / (2 ** scale)  # 平滑度正则化

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales  # 平均多尺度损失
        losses["loss"] = total_loss
        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()  # 评估模式
        try:
            inputs = next( _iter)  # 修复：使用next()函数而不是.next()方法
        except StopIteration:
            self.val_iter = iter(self.val_loader)  # 重置迭代器
            inputs = next(self.val_iter)  # 修复：使用next()函数

        with torch.no_grad():  # 不计算梯度
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses)  # 记录验证日志
            del inputs, outputs, losses

        self.set_train()  # 恢复训练模式

    def process_batch_val(self, inputs):
        """验证阶段的批次处理（类似训练阶段）
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses(inputs))
        losses = self.compute_losses_val(inputs, outputs)

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """验证阶段的损失计算（使用NCC损失）
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                # 使用NCC（归一化互相关）损失代替SSIM
                registration_losses.append(
                    ncc_loss(outputs[("registration", scale, frame_id)].mean(1, True), target.mean(1, True)))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss  # 验证损失取负值（NCC越大越好）
        return losses

    def log_time(self, batch_idx, duration, loss):
        """打印训练进度信息
        """
        samples_per_sec = self.opt.batch_size / duration  # 每秒处理样本数
        time_sofar = time.time() - self.start_time  # 已用时间
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """记录Tensorboard日志
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # 最多记录4张图像
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)

    def save_opts(self):
        """保存实验配置到JSON文件
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """保存模型权重
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # 保存图像尺寸信息（预测时需要）
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)  # 保存优化器状态

    def load_model(self):
        """从磁盘加载模型权重
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
            # 只加载匹配的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # 加载Adam优化器状态
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")