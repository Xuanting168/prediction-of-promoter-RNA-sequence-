#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils.py - 工具函数文件，包含各种有用的辅助函数

import os
import random
import numpy as np
import torch
from Bio import SeqIO
from typing import List, Tuple


def set_seed(seed: int = 42):
    """
    设置随机种子，确保每次运行的结果一致

    为什么要设置随机种子？
    - 深度学习中有很多随机操作（如随机初始化权重、随机丢弃等）
    - 如果不设置种子，每次运行结果会不同
    - 设置种子后，结果是固定的，方便调试和复现

    参数:
        seed: 随机种子数值，默认 42
    """
    random.seed(seed)              # Python 内置随机数生成器
    np.random.seed(seed)           # NumPy 随机数生成器
    torch.manual_seed(seed)        # PyTorch CPU 随机数生成器

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch 所有 GPU 随机数生成器
        torch.backends.cudnn.deterministic = True  # 确保 CuDNN 计算是确定的
        torch.backends.cudnn.benchmark = False    # 禁用自动优化，保证确定性


def load_fasta_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """
    从 FASTA 文件中读取序列

    什么是 FASTA 格式？
    - FASTA 是一种常见的生物序列存储格式
    - 格式示例：
      >序列名称1
      ACGUTCGATCGATCG
      >序列名称2
      CGATCGATCGATCGA

    参数:
        fasta_path: FASTA 文件路径

    返回:
        list: 包含 (序列名, 序列) 的列表
    """
    sequences = []

    # 使用 Biopython 库读取 FASTA 文件
    for record in SeqIO.parse(fasta_path, "fasta"):
        # 转换为 RNA 序列（如果是 DNA 的话，T -> U）
        seq = str(record.seq).upper().replace('T', 'U')
        sequences.append((record.id, seq))

    return sequences


def split_data(data: List, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """
    将数据集划分为训练集、验证集和测试集

    为什么需要划分数据集？
    - 训练集（Train）：用于训练模型
    - 验证集（Val）：用于调整超参数、选择最佳模型
    - 测试集（Test）：最终评估模型性能，只能用一次

    参数:
        data: 原始数据列表
        train_ratio: 训练集比例，默认 0.7（70%）
        val_ratio: 验证集比例，默认 0.15（15%）
        seed: 随机种子

    返回:
        (train_data, val_data, test_data): 三个数据集
    """
    # 先设置随机种子，保证划分结果一致
    random.seed(seed)

    # 随机打乱数据（打乱顺序）
    random.shuffle(data)

    # 计算每个数据集的大小
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # 划分数据
    train_data = data[:n_train]                    # 开头到 n_train
    val_data = data[n_train:n_train + n_val]       # n_train 到 n_train+n_val
    test_data = data[n_train + n_val:]             # 剩下的部分

    return train_data, val_data, test_data


def prepare_data_from_folders(
    promoter_dir: str,
    non_promoter_dir: str,
    max_length: int = 512
) -> List[Tuple[str, int]]:
    """
    从两个文件夹准备数据

    文件夹结构应该是：
        promoter_dir/
            file1.fasta
            file2.fasta
            ...
        non_promoter_dir/
            file1.fasta
            file2.fasta
            ...

    参数:
        promoter_dir: 启动子序列文件夹路径
        non_promoter_dir: 非启动子序列文件夹路径
        max_length: 序列最大长度（超过会被截断）

    返回:
        list: 包含 (序列, 标签) 的列表，标签 1 表示启动子，0 表示非启动子
    """
    data = []

    # ========== 第一步：加载启动子序列（标签为 1）==========
    print(f"正在从 {promoter_dir} 加载启动子序列...")

    # 检查文件夹是否存在
    if not os.path.exists(promoter_dir):
        print(f"警告: 文件夹 {promoter_dir} 不存在！")
    else:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(promoter_dir):
            # 只处理 .fasta 或 .fa 文件
            if filename.endswith(('.fasta', '.fa')):
                filepath = os.path.join(promoter_dir, filename)

                # 读取文件中的所有序列
                seqs = load_fasta_sequences(filepath)

                # 每个序列添加到数据列表，标签为 1（启动子）
                for seq_id, seq in seqs:
                    # 截断过长的序列
                    seq = seq[:max_length]
                    data.append((seq, 1))

    # ========== 第二步：加载非启动子序列（标签为 0）==========
    print(f"正在从 {non_promoter_dir} 加载非启动子序列...")

    if not os.path.exists(non_promoter_dir):
        print(f"警告: 文件夹 {non_promoter_dir} 不存在！")
    else:
        for filename in os.listdir(non_promoter_dir):
            if filename.endswith(('.fasta', '.fa')):
                filepath = os.path.join(non_promoter_dir, filename)
                seqs = load_fasta_sequences(filepath)
                for seq_id, seq in seqs:
                    seq = seq[:max_length]
                    data.append((seq, 0))  # 标签为 0（非启动子）

    print(f"共加载了 {len(data)} 条序列")

    # 统计一下数据分布
    n_promoters = sum(1 for _, label in data if label == 1)
    n_non_promoters = len(data) - n_promoters
    print(f"  - 启动子: {n_promoters} 条")
    print(f"  - 非启动子: {n_non_promoters} 条")

    return data


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    保存模型检查点（包括模型参数和优化器状态）

    为什么要保存检查点？
    - 训练可能需要很长时间，如果中断可以从检查点继续
    - 可以保存不同 epoch 的模型，选择最好的
    - 可以保存优化器状态，继续训练时梯度信息不会丢失

    参数:
        model: 模型对象
        optimizer: 优化器对象
        epoch: 当前轮数
        loss: 当前损失值
        filepath: 保存路径
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    加载模型检查点

    参数:
        filepath: 检查点文件路径
        model: 模型对象（会更新参数）
        optimizer: 优化器对象（可选，会更新状态）

    返回:
        (epoch, loss): 保存时的轮数和损失
    """
    checkpoint = torch.load(filepath)

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态（如果提供了优化器）
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']
