#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# promoter_predictor.py - 核心模型文件

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 非常重要：添加路径设置，让 Python 能找到 RhoFold 库！
# 当前脚本的位置: MyProgamme/promoter_predictor.py
# RhoFold 的位置: ..\RhoFold-main
MYPROGAMME_DIR = os.path.dirname(os.path.abspath(__file__))
RH_OFOLD_DIR = os.path.join(MYPROGAMME_DIR, '..', 'RhoFold-main')
sys.path.append(RH_OFOLD_DIR)

# 现在可以导入 RNA-FM 相关模块了
import rhofold.model.rna_fm as rna_fm
from rhofold.model.rna_fm.pretrained import esm1b_rna_t12


class PromoterPredictor(nn.Module):
    """
    RNA 启动子预测模型，基于 RNA-FM 预训练

    模型原理：
    1. 使用 RNA-FM 预训练模型提取 RNA 序列的表示
    2. 将序列表示通过分类头转换为启动子概率
    3. 支持两个阶段的训练：冻结预训练和微调

    参数:
        freeze_rnafm: 是否冻结 RNA-FM 的预训练权重（默认为 True，建议先冻结训练）
    """
    def __init__(self, freeze_rnafm: bool = True):
        super().__init__()

        print("初始化 PromoterPredictor 模型...")

        # ========== 第一步：加载 RNA-FM 预训练模型 ==========
        print("1. 正在加载 RNA-FM 预训练模型...")
        self.rnafm, self.alphabet = esm1b_rna_t12()
        print(f"RNA-FM 字母表大小: {len(self.alphabet)}")
        print(f"RNA-FM 层数: {self.rnafm.args.layers}")
        print(f"RNA-FM 嵌入维度: {self.rnafm.args.embed_dim}")
        print(f"RNA-FM 注意力头数: {self.rnafm.args.attention_heads}")

        # ========== 第二步：是否冻结 RNA-FM 预训练权重 ==========
        if freeze_rnafm:
            print("2. 冻结 RNA-FM 预训练权重（只训练分类头）")
            for param in self.rnafm.parameters():
                param.requires_grad = False
        else:
            print("2. 允许训练 RNA-FM 预训练权重（微调阶段）")

        # ========== 第三步：添加启动子分类头 ==========
        # RNA-FM 的最后一层输出维度是 640
        # 我们需要一个简单的神经网络来将 640 维特征转换为二分类（是/不是启动子）
        print("3. 构建分类头...")
        self.classifier = nn.Sequential(
            nn.Linear(640, 320),       # 第一层：从 640 维降到 320 维
            nn.ReLU(),                 # 激活函数（引入非线性）
            nn.Dropout(0.1),           # Dropout 层（防止过拟合）
            nn.Linear(320, 160),       # 第二层：从 320 维降到 160 维
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 2)          # 输出层：2 维（对应两个类别）
        )

    def forward(self, rna_fm_tokens):
        """
        前向传播函数（模型预测时调用）

        参数:
            rna_fm_tokens: RNA-FM 格式的输入序列 [batch_size, seq_length]

        返回:
            logits: 分类分数 [batch_size, 2]
            representations: RNA-FM 各层的表示
        """
        # ========== 第一步：RNA-FM 特征提取 ==========
        # 调用 RNA-FM 的 forward 方法
        # repr_layers=[12]: 只取第 12 层的输出（因为有 12 层，0-11）
        # need_head_weights=False: 不需要返回注意力权重（节省计算）
        outputs = self.rnafm(
            rna_fm_tokens,
            repr_layers=[12],
            need_head_weights=False
        )

        # 获得最后一层的表示
        last_repr = outputs['representations'][12]  # 形状: [batch, seq_len, 640]

        # ========== 第二步：序列特征聚合 ==========
        # 我们需要将序列的每个位置的表示合并成一个整体表示
        # 方案：平均池化（对有效位置求平均，忽略填充）
        pad_mask = rna_fm_tokens != self.alphabet.padding_idx  # 找到不是填充的位置
        valid_lengths = pad_mask.sum(dim=1, keepdim=True)      # 每个序列的有效长度

        # 避免除零（虽然代码中不会发生，但作为安全措施）
        valid_lengths = torch.clamp(valid_lengths, min=1)

        # 计算平均表示
        avg_repr = (last_repr * pad_mask.unsqueeze(-1)).sum(dim=1) / valid_lengths

        # ========== 第三步：分类 ==========
        logits = self.classifier(avg_repr)

        return logits, outputs['representations']
