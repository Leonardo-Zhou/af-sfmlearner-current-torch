from __future__ import absolute_import, division, print_function

import os
import numpy as np
from PIL import Image
import torch

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    """
    SCARED数据集基类
    用于处理SCARED数据集中的图像和相机参数
    继承自MonoDataset，实现单目深度估计所需的基本功能
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化SCARED数据集
        
        参数:
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
        """
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        # 相机内参矩阵 (3x4)
        # 这是一个归一化的内参矩阵，假设图像已经归一化到[0,1]范围
        # 实际应用中需要根据真实图像尺寸进行缩放
        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # 原始图像分辨率 (已注释掉)
        # SCARED数据集的原始分辨率为1280x1024
        # self.full_res_shape = (1280, 1024)
        
        # 相机视角映射字典
        # 将字符串标识映射到相机编号
        # "2"或"l"表示左相机，"3"或"r"表示右相机
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        """
        检查数据集是否包含深度真值
        
        返回:
            bool: False表示此数据集版本不包含深度真值
                  需要在子类中重写以支持深度评估
        """
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        """
        获取彩色图像
        
        参数:
            folder: 数据序列文件夹名称
            frame_index: 帧索引号
            side: 相机视角 ("l"或"r")
            do_flip: 是否水平翻转图像
            
        返回:
            PIL.Image: 加载的彩色图像
        """
        # 使用父类的loader方法加载图像
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        # 根据需要进行水平翻转
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color


class SCAREDRAWDataset(SCAREDDataset):
    """
    SCARED数据集RAW版本
    用于加载SCARED数据集的原始图像文件
    继承自SCAREDDataset，实现了具体的文件路径解析
    """
    
    def __init__(self, *args, **kwargs):
        """
        初始化SCARED RAW数据集
        
        参数:
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数
        """
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        """
        获取图像文件的完整路径
        
        参数:
            folder: 数据序列文件夹名称
            frame_index: 帧索引号
            side: 相机视角 ("l"或"r")
            
        返回:
            str: 图像文件的完整路径
        """
        # 格式化文件名，补零到10位数字
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        
        # 构建完整路径
        # 路径格式: data_path/folder/image_0X/data/XXXXXXXXXX.png
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        获取深度图真值
        
        参数:
            folder: 数据序列文件夹名称
            frame_index: 帧索引号（注意：深度图索引比图像索引小1）
            side: 相机视角 ("l"或"r")
            do_flip: 是否水平翻转深度图
            
        返回:
            np.ndarray: 深度图数组，形状为(H, W)
        """
        # 深度图文件名格式，注意索引需要减1
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)

        # 构建深度图路径
        # 路径格式: data_path/folder/image_0X/data/groundtruth/scene_pointsXXXXXX.tiff
        depth_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data/groundtruth".format(self.side_map[side]),
            f_str)

        # 使用PIL读取深度图（现代版本兼容）
        depth_img = Image.open(depth_path)
        depth_gt = np.array(depth_img)
        if len(depth_gt.shape) == 3:
            depth_gt = depth_gt[:, :, 0]  # 提取第一个通道作为深度值
        depth_gt = depth_gt[0:1024, :]
        
        # 根据需要进行水平翻转
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


