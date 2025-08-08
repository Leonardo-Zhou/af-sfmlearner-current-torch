# -*- coding: utf-8 -*-
"""
深度解码器模块
用于从编码器特征生成深度图，采用U-Net风格的解码器架构
支持多尺度深度预测和跳跃连接
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    """
    深度解码器类
    
    采用U-Net架构的解码器，从ResNet编码器的多尺度特征生成深度图
    支持：
    - 多尺度深度预测（4个尺度）
    - 跳跃连接（skip connections）
    - 可配置的输出通道数
    """
    
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        """
        初始化深度解码器
        
        参数:
            num_ch_enc: 编码器各层的通道数列表，如[64,64,128,256,512]
            scales: 需要输出深度的尺度列表，默认为[0,1,2,3]
            num_output_channels: 输出通道数，深度估计为1，视差估计为1，其他任务可调整
            use_skips: 是否使用跳跃连接，通常设为True以保留细节信息
        """
        super(DepthDecoder, self).__init__()
        
        # 保存配置参数
        self.num_output_channels = num_output_channels  # 输出通道数（深度图为1）
        self.use_skips = use_skips  # 是否使用跳跃连接
        self.upsample_mode = 'nearest'  # 上采样模式（最近邻插值）
        self.scales = scales  # 输出尺度列表
        
        # 编码器通道配置
        self.num_ch_enc = num_ch_enc  # 编码器各层通道数 [64,64,128,256,512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # 解码器各层通道数
        
        # 构建解码器网络层
        self.convs = OrderedDict()  # 使用有序字典保存卷积层
        
        """
        解码器架构说明：
        - 从最深的特征（i=4）开始逐步上采样
        - 每个尺度包含两个卷积块：upconv_0和upconv_1
        - 使用跳跃连接融合编码器的对应层特征
        - 最后在指定尺度输出深度图
        """
        
        # 构建5个尺度的解码器层（i从4到0）
        for i in range(4, -1, -1):
            # 第一个卷积块：处理输入特征
            # 对于最深层(i=4)，输入来自编码器的最后一层
            # 对于其他层，输入来自上一层的输出
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # 第二个卷积块：融合跳跃连接后的特征
            # 输入包括：上采样后的特征 + 编码器对应层的特征（如果use_skips=True）
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                # 添加跳跃连接：融合编码器对应层的特征
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 为每个指定尺度添加深度预测层
        # 使用1x1卷积将特征图转换为深度图
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(
                self.num_ch_dec[s],  # 输入通道数
                self.num_output_channels  # 输出通道数（深度图为1）
            )
        
        # 将所有卷积层注册为模块列表
        self.decoder = nn.ModuleList(list(self.convs.values()))
        
        # Sigmoid激活函数，将输出限制在[0,1]范围
        # 对于深度估计，输出的是归一化的视差图
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_features):
        """
        前向传播函数
        
        参数:
            input_features: 编码器的多尺度特征列表
                          每个元素是一个特征图，从浅到深排列
                          如：[feat1, feat2, feat3, feat4, feat5]
        
        返回:
            outputs: 字典，包含各尺度的深度预测
                    键为("disp", scale)，值为对应尺度的深度图
        """
        
        # 初始化输出字典
        self.outputs = {}
        
        # 从最深层特征开始解码
        # input_features[-1]是最深层的特征（最小空间尺寸，最多通道）
        x = input_features[-1]
        
        # 逐步上采样和特征融合
        for i in range(4, -1, -1):
            # 第一步：通过第一个卷积块
            x = self.convs[("upconv", i, 0)](x)
            
            # 第二步：上采样到更高分辨率
            # upsample函数使用最近邻插值进行2倍上采样
            x = [upsample(x)]
            
            # 第三步：添加跳跃连接（如果启用）
            if self.use_skips and i > 0:
                # 将编码器对应层的特征添加到当前特征
                # input_features[i-1]是编码器中对应尺度的特征
                x += [input_features[i - 1]]
            
            # 第四步：拼接所有特征
            # torch.cat在通道维度上拼接特征
            x = torch.cat(x, 1)
            
            # 第五步：通过第二个卷积块融合特征
            x = self.convs[("upconv", i, 1)](x)
            
            # 第六步：在指定尺度输出深度图
            if i in self.scales:
                # 使用1x1卷积生成深度图，然后应用sigmoid激活
                disp = self.convs[("dispconv", i)](x)
                self.outputs[("disp", i)] = self.sigmoid(disp)
        
        return self.outputs


"""
使用示例：

# 假设编码器输出5个尺度的特征
encoder = ResnetEncoder(18, pretrained=True)
decoder = DepthDecoder(encoder.num_ch_enc, scales=[0,1,2,3])

# 前向传播
input_image = torch.randn(1, 3, 256, 320)
features = encoder(input_image)  # 5个尺度的特征
outputs = decoder(features)  # 包含4个尺度的深度图

# outputs包含：
# ("disp", 0): 最高分辨率深度图 [1,1,256,320]
# ("disp", 1): 1/2分辨率深度图 [1,1,128,160]
# ("disp", 2): 1/4分辨率深度图 [1,1,64,80]
# ("disp", 3): 1/8分辨率深度图 [1,1,32,40]
"""

"""
网络结构可视化：

输入特征（编码器输出）：
[64,64,128,256,512] ← 各层通道数
[256,128,64,32,16]  ← 空间尺寸（假设输入256×320）

解码器流程：
i=4: 512→256→128 (上采样) [16×20]
i=3: 128+256→256→256 [32×40] → 输出尺度3
i=2: 256+128→128→128 [64×80] → 输出尺度2
i=1: 128+64→64→64 [128×160] → 输出尺度1
i=0: 64+64→32→32 [256×320] → 输出尺度0
"""