from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Tuple, Optional

from torch.distributions.normal import Normal
from collections import OrderedDict
from layers import ConvBlock, upsample


class PositionDecoder(nn.Module):
    """
    光流解码器（PositionDecoder）
    
    基于U-Net架构的解码器，用于从ResNet编码器提取的多尺度特征中
    预测光流场。支持跳跃连接和多尺度输出。
    
    主要功能：
    1. 通过转置卷积/上采样逐步恢复空间分辨率
    2. 利用跳跃连接融合编码器的高分辨率特征
    3. 在多个尺度上预测光流，提供不同精度的输出
    
    输出格式：每个尺度的光流张量，形状为[N, 2, H_i, W_i]
    其中2表示x和y方向的位移
    """
    
    def __init__(self, 
                 num_ch_enc: List[int], 
                 scales: range = range(4),
                 num_output_channels: int = 2, 
                 use_skips: bool = True):
        """
        初始化光流解码器
        
        Args:
            num_ch_enc (List[int]): 编码器各阶段的通道数列表
                例如：[64, 64, 128, 256, 512] 对应ResNet18
            scales (range): 需要输出光流的尺度索引范围
                默认range(4)表示在0-3尺度上输出
            num_output_channels (int): 输出通道数，光流默认为2（x,y位移）
            use_skips (bool): 是否使用跳跃连接融合编码器特征
        """
        super().__init__()

        # 网络配置参数
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'  # 上采样模式：最近邻插值
        self.scales = list(scales)  # 转换为列表确保兼容性

        # 通道数配置
        self.num_ch_enc = num_ch_enc  # 编码器通道数
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # 解码器通道数
        
        # 使用标准Conv2d（现代PyTorch推荐直接使用nn.Conv2d）
        self.conv_layer = nn.Conv2d

        # 构建解码器网络层
        self._build_decoder()
        
        # 注册所有子模块
        self.decoder = nn.ModuleList(list(self.convs.values()))

    def _build_decoder(self) -> None:
        """构建解码器网络结构"""
        self.convs = OrderedDict()  # 保持层顺序的有序字典
        
        # 构建上采样路径（从深层到浅层）
        for i in range(4, -1, -1):
            # 第一层上采样：通道数变换
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # 第二层卷积：融合跳跃连接后的特征
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                # 跳跃连接：加上对应编码器层的特征
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # 为每个输出尺度构建最终的光流预测层
        for scale in self.scales:
            # 光流预测卷积层
            self.convs[("position_conv", scale)] = self.conv_layer(
                self.num_ch_dec[scale], 
                self.num_output_channels, 
                kernel_size=3, 
                padding=1
            )
            
            # 初始化光流层的权重为小随机值，偏置为0
            # 这种初始化有助于训练稳定性，避免初始预测过大
            self.convs[("position_conv", scale)].weight = nn.Parameter(
                Normal(0, 1e-5).sample(self.convs[("position_conv", scale)].weight.shape)
            )
            self.convs[("position_conv", scale)].bias = nn.Parameter(
                torch.zeros_like(self.convs[("position_conv", scale)].bias)
            )

    def forward(self, input_features: List[Tensor]) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            input_features (List[Tensor]): 来自编码器的多尺度特征列表
                按分辨率从高到低排列：
                - input_features[0]: 1/2分辨率特征
                - input_features[1]: 1/4分辨率特征
                - ...
                - input_features[-1]: 最低分辨率特征
        
        Returns:
            Dict[str, Tensor]: 多尺度光流预测结果
                键格式：("position", scale_index)
                值格式：Tensor形状 [batch_size, 2, height, width]
                2表示x和y方向的位移分量
        """
        outputs = {}
        
        # 从最深层的特征开始解码
        x = input_features[-1]  # 最低分辨率特征
        
        # 自底向上的解码过程
        for i in range(4, -1, -1):
            # 第一步：通道数变换和上采样
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)  # 上采样到更高分辨率
            
            # 准备融合的特征列表
            concat_features = [x]
            
            # 第二步：跳跃连接（如果启用）
            if self.use_skips and i > 0:
                # 添加对应编码器层的特征
                skip_connection = input_features[i - 1]
                # 确保尺寸匹配（处理可能的尺寸差异）
                if x.shape[-2:] != skip_connection.shape[-2:]:
                    skip_connection = nn.functional.interpolate(
                        skip_connection, size=x.shape[-2:], mode='nearest'
                    )
                concat_features.append(skip_connection)
            
            # 融合所有特征
            x = torch.cat(concat_features, dim=1)
            
            # 第三步：通过卷积层处理融合后的特征
            x = self.convs[("upconv", i, 1)](x)
            
            # 在当前尺度预测光流（如果该尺度需要输出）
            if i in self.scales:
                outputs[("position", i)] = self.convs[("position_conv", i)](x)
        
        return outputs

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息（用于调试和日志记录）"""
        return {
            "encoder_channels": self.num_ch_enc.tolist(),
            "decoder_channels": self.num_ch_dec.tolist(),
            "output_scales": self.scales,
            "use_skips": self.use_skips,
            "num_parameters": sum(p.numel() for p in self.parameters())
        }
