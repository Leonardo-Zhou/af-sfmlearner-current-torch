from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms

# 允许加载截断的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    """
    PIL图像加载器
    
    使用with语句打开文件，避免资源警告
    将图像转换为RGB格式
    
    参数:
        path: 图像文件路径
        
    返回:
        PIL.Image: RGB格式的图像
    """
    # 以二进制模式打开文件以避免资源警告
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """
    单目数据集基类
    
    为单目深度估计和姿态估计提供基础数据加载功能。
    支持多尺度图像处理、数据增强、相机内参调整等。
    
    参数:
        data_path: 数据集根目录路径
        filenames: 训练/验证文件名列表
        height: 目标图像高度
        width: 目标图像宽度
        frame_idxs: 需要加载的帧索引列表 (如[0, -1, 1])
        num_scales: 多尺度金字塔的层数
        is_train: 是否为训练模式
        img_ext: 图像文件扩展名
    """
    
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoDataset, self).__init__()

        # 基础参数初始化
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        # 帧索引配置
        self.frame_idxs = frame_idxs

        # 训练/验证模式标志
        self.is_train = is_train
        self.img_ext = img_ext

        # 图像加载和转换工具
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # 数据增强参数 ，用于在训练时随机调整图像的亮度、对比度、饱和度和色调，增加数据多样性，提高模型泛化能力。
        # 有效防止过拟合
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.color_jitter = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue)

        # 创建多尺度调整器
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i  # 尺度因子
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                            #    interpolation=self.interp)
                                            interpolation=transforms.InterpolationMode.LANCZOS)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """
        图像预处理和增强
        
        将彩色图像调整到所需尺度，并根据需要进行数据增强。
        确保同一序列中的所有图像使用相同的数据增强。
        
        参数:
            inputs: 输入图像字典
            color_aug: 颜色增强变换
        """
        # 多尺度图像调整
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    # 从上一级尺度下采样得到当前尺度
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        # 转换为张量并应用数据增强
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)  # 原始图像转张量
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))  # 增强图像转张量

    def __len__(self):
        """返回数据集大小"""
        return len(self.filenames)

    def __getitem__(self, index):
        """
        获取单个训练样本
        
        返回包含多尺度图像、相机内参、深度图等信息的字典。
        
        返回格式:
            ("color", <frame_id>, <scale>): 原始彩色图像
            ("color_aug", <frame_id>, <scale>): 增强后的彩色图像
            ("K", scale) 或 ("inv_K", scale): 相机内参及其逆矩阵
            "stereo_T": 立体相机外参
            "depth_gt": 深度图真值
            
        帧索引说明:
            - 整数 (如0, -1, 1): 相对于当前帧的时间步
            - "s": 立体图像对的另一侧
            
        尺度说明:
            - -1: 原始分辨率
            - 0: 目标分辨率 (self.width, self.height)
            - 1: 1/2分辨率
            - 2: 1/4分辨率
            - 3: 1/8分辨率
        """
        inputs = {}

        # 数据增强开关
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # 解析文件名信息
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        # 加载多帧图像
        for i in self.frame_idxs:
            if i == "s":
                # 立体图像对的另一侧
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                # 时序相邻帧
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # 调整相机内参以匹配多尺度金字塔
        for scale in range(self.num_scales):
            # 这里，self.K 是在子类当中定义的
            K = self.K.copy()
            # 根据尺度调整内参
            K[0, :] *= self.width // (2 ** scale)  # 调整fx和cx
            K[1, :] *= self.height // (2 ** scale)  # 调整fy和cy

            # 计算逆内参矩阵
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 设置数据增强
        if do_color_aug:
            color_aug = self.color_jitter
        else:
            color_aug = (lambda x: x)

        # 应用预处理和增强
        self.preprocess(inputs, color_aug)
        
        # 清理临时数据
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # 加载深度图真值（如果可用）
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)  # 增加通道维度
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # 立体图像对的外参矩阵
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)  # 4x4单位矩阵
            baseline_sign = -1 if do_flip else 1  # 考虑翻转的影响
            side_sign = -1 if side == "l" else 1  # 左右相机标识
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1  # 基线长度设为0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
            
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        """
        获取彩色图像（子类必须实现）
        
        参数:
            folder: 数据序列文件夹
            frame_index: 帧索引
            side: 相机视角
            do_flip: 是否翻转
            
        返回:
            PIL.Image: 彩色图像
        """
        raise NotImplementedError

    def check_depth(self):
        """
        检查是否支持深度图真值（子类必须实现）
        
        返回:
            bool: 是否支持深度图
        """
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        获取深度图真值（子类必须实现）
        
        参数:
            folder: 数据序列文件夹
            frame_index: 帧索引
            side: 相机视角
            do_flip: 是否翻转
            
        返回:
            np.ndarray: 深度图
        """
        raise NotImplementedError
