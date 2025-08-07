from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """
    支持多图像输入的ResNet模型
    
    该类继承自torchvision.models.ResNet，修改了第一层卷积以支持多个图像的拼接输入。
    主要用于处理时序图像或多视角图像的编码任务。
    
    参数:
        block: ResNet的基本构建块（BasicBlock或Bottleneck）
        layers: 每个阶段的残差块数量列表，如[2,2,2,2]对应ResNet18
        num_classes: 分类任务的类别数（默认为1000，但在此任务中不使用）
        num_input_images: 输入图像的数量，用于时序或多视角融合
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        # 重置输入通道数，支持多图像拼接
        self.inplanes = 64
        
        # 修改第一层卷积：输入通道数从3变为num_input_images*3
        # 例如：2张RGB图像拼接后输入通道数为6
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet的四个主要阶段
        self.layer1 = self._make_layer(block, 64, layers[0])   # 64通道特征图
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 下采样2倍
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 下采样4倍
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 下采样8倍

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化，适合ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """
    构建支持多图像输入的ResNet模型
    
    根据层数创建对应的ResNet模型，支持加载预训练权重，并适配多图像输入。
    
    参数:
        num_layers (int): ResNet层数，支持18或50层
        pretrained (bool): 是否加载ImageNet预训练权重
        num_input_images (int): 输入图像数量，用于时序或多视角输入
        
    返回:
        ResNetMultiImageInput: 配置好的ResNet模型
    """
    # 验证输入参数
    assert num_layers in [18, 50], "仅支持18层或50层的ResNet"
    
    # ResNet配置映射
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    
    # 创建模型实例
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # 加载预训练权重（兼容torchvision 0.22.2）
        model_url = models.resnet.model_urls[f'resnet{num_layers}']
        loaded = model_zoo.load_url(model_url)
        
        # 适配多图像输入：复制并平均化第一个卷积层的权重
        # 将3通道的预训练权重扩展到num_input_images*3通道
        original_weight = loaded['conv1.weight']  # [64, 3, 7, 7]
        expanded_weight = torch.cat([original_weight] * num_input_images, dim=1) / num_input_images
        loaded['conv1.weight'] = expanded_weight
        
        # 加载修改后的权重
        model.load_state_dict(loaded)
    
    return model


class ResnetEncoder(nn.Module):
    """
    ResNet编码器模块
    
    用于深度估计和姿态估计的ResNet特征提取器。
    支持多种ResNet变体（18/34/50/101/152层），可选择预训练权重，
    支持多图像输入（用于时序或多视角学习）。
    
    输出特征层级：
    - 第0层：原始图像的2倍下采样（H/2, W/2）
    - 第1层：4倍下采样（H/4, W/4）
    - 第2层：8倍下采样（H/8, W/8）
    - 第3层：16倍下采样（H/16, W/16）
    - 第4层：32倍下采样（H/32, W/32）
    
    参数:
        num_layers: ResNet层数（18, 34, 50, 101, 152）
        pretrained: 是否使用ImageNet预训练权重
        num_input_images: 输入图像数量（用于时序输入）
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        # 各层输出通道数
        # 注意：这是ResNet18/34的通道配置，ResNet50+会乘以4
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        # 支持的ResNet变体映射（兼容torchvision 0.22.2）
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }

        # 验证输入参数
        if num_layers not in resnets:
            raise ValueError(f"{num_layers}不是有效的ResNet层数，支持：{list(resnets.keys())}")

        # 根据输入图像数量选择编码器
        if num_input_images > 1:
            # 多图像输入：使用自定义的ResNet
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            # 单图像输入：使用标准torchvision ResNet
            self.encoder = resnets[num_layers](pretrained)

        # 调整通道数：ResNet50+使用Bottleneck，通道数扩大4倍
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4  # 变为[64, 256, 512, 1024, 2048]

    def forward(self, input_image):
        """
        前向传播
        
        提取多尺度特征，用于后续的深度解码和姿态估计。
        兼容torch==2.7.1+cu126的自动混合精度训练。
        
        参数:
            input_image: 输入图像张量，形状[N, C*num_input_images, H, W]
            
        返回:
            list: 5个尺度的特征图列表，每个元素形状为：
                - features[0]: [N, 64, H/2, W/2]   (或256对于ResNet50+)
                - features[1]: [N, 64, H/4, W/4]   (或256)
                - features[2]: [N, 128, H/8, W/8]  (或512)
                - features[3]: [N, 256, H/16, W/16] (或1024)
                - features[4]: [N, 512, H/32, W/32] (或2048)
        """
        # 存储各层特征
        self.features = []
        
        # 注意：原始代码中的归一化被注释掉了
        # 如果需要ImageNet预训练的归一化，取消下面这行注释
        # x = (input_image - 0.45) / 0.225  # ImageNet均值方差归一化
        x = input_image
        
        # 第0层：初始卷积和池化
        # 输出：H/2, W/2
        x = self.encoder.conv1(x)        # 7x7卷积，stride=2
        x = self.encoder.bn1(x)          # 批归一化
        x = self.encoder.relu(x)         # ReLU激活
        self.features.append(x)          # 保存第0层特征
        
        # 第1层：最大池化 + ResNet stage1
        # 输出：H/4, W/4
        x = self.encoder.maxpool(x)      # 3x3最大池化，stride=2
        x = self.encoder.layer1(x)       # ResNet stage1
        self.features.append(x)          # 保存第1层特征
        
        # 第2层：ResNet stage2
        # 输出：H/8, W/8
        x = self.encoder.layer2(x)
        self.features.append(x)          # 保存第2层特征
        
        # 第3层：ResNet stage3
        # 输出：H/16, W/16
        x = self.encoder.layer3(x)
        self.features.append(x)          # 保存第3层特征
        
        # 第4层：ResNet stage4
        # 输出：H/32, W/32
        x = self.encoder.layer4(x)
        self.features.append(x)          # 保存第4层特征

        return self.features
