#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data_loader.py - 数据加载和处理

import sys
import os
import re

# 添加路径设置，确保能找到 RhoFold 库
MYPROGAMME_DIR = os.path.dirname(os.path.abspath(__file__))
RHOFOLD_DIR = os.path.join(MYPROGAMME_DIR, '..', 'RhoFold-main')
sys.path.append(RHOFOLD_DIR)

# 导入配置
from config import JSON_DATA_DIR, FASTA_DATA_DIR

from rhofold.model.rna_fm.data import Alphabet, BatchConverter
import torch
from torch.utils.data import Dataset, DataLoader


class PromoterDataset(Dataset):
    """
    启动子数据集类

    为什么需要这个类？
    - PyTorch 的 DataLoader 需要一个 Dataset 类
    - 这个类负责存储数据并提供获取数据的接口
    - 这样 DataLoader 才能在训练时批量加载数据

    参数:
        data: 数据列表，格式为 [(序列字符串, 标签), ...]，标签 0 表示非启动子，1 表示启动子
        alphabet: RNA-FM 的字母表（用于转换序列）
    """
    def __init__(self, data, alphabet):
        self.data = data
        self.alphabet = alphabet

        # 获取批次转换器，用于将字符串序列转换为模型可接受的格式
        self.batch_converter = alphabet.get_batch_converter()

        print(f"数据集大小: {len(data)}")
        print(f"正样本（启动子）数量: {sum(1 for _, label in data if label == 1)}")
        print(f"负样本（非启动子）数量: {sum(1 for _, label in data if label == 0)}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取第 idx 个数据"""
        return self.data[idx]

    def collate_fn(self, batch):
        """
        批次处理函数

        为什么需要这个函数？
        - 输入的序列长度可能不同
        - 需要将它们填充到相同的长度，以便模型处理
        - 需要转换为 PyTorch 张量格式

        参数:
            batch: 一批数据 [ (seq1, label1), (seq2, label2), ... ]

        返回:
            (tokens, labels): 张量格式的输入
        """
        # 1. 分离序列和标签
        seqs, labels = zip(*batch)

        # 2. 准备数据格式
        # 格式是 [ ("序列索引", "序列字符串"), ... ]
        data = [ (str(i), seq) for i, seq in enumerate(seqs) ]

        # 3. 使用 RNA-FM 的批次转换器
        _, _, tokens = self.batch_converter(data)

        # 4. 将标签转换为张量
        labels = torch.tensor(labels)

        return tokens, labels


def create_dataloaders(
    train_data,
    val_data,
    test_data,
    batch_size: int = 8
):
    """
    创建数据加载器

    为什么需要数据加载器？
    - 在训练时不能一次加载所有数据到内存
    - 数据加载器负责批量读取数据
    - 提供随机打乱、并行加载等功能

    参数:
        train_data, val_data, test_data: 划分后的数据集
        batch_size: 每个批次的大小

    返回:
        (train_loader, val_loader, test_loader): 三个数据加载器
    """
    print("创建数据加载器...")

    # 获取 RNA-FM 的字母表
    from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
    _, alphabet = esm1b_rna_t12()

    # 1. 创建数据集实例
    train_dataset = PromoterDataset(train_data, alphabet)
    val_dataset = PromoterDataset(val_data, alphabet)
    test_dataset = PromoterDataset(test_data, alphabet)

    # 2. 创建数据加载器
    # num_workers: 使用多少进程加载数据（Windows 下设为 0，否则会报错）
    num_workers = 0 if sys.platform.startswith('win') else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers
    )

    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    print(f"测试批次数量: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def read_fasta(fasta_file):
    """
    读取单个FASTA文件

    Args:
        fasta_file: FASTA文件路径

    Returns:
        列表，格式为 [(序列ID, 序列字符串), ...]
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # 保存前一条序列
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))

                # 解析新的序列ID
                # 格式: >ID 描述
                match = re.match(r'^>(\S+)', line)
                if match:
                    current_id = match.group(1)
                else:
                    current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    # 保存最后一条序列
    if current_id is not None:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def read_all_fasta(fasta_dir, label=None):
    """
    读取目录下所有FASTA文件

    Args:
        fasta_dir: FASTA文件目录
        label: 标签（0或1，如果提供，所有序列都会使用这个标签）

    Returns:
        列表，格式为 [(序列字符串, 标签), ...]
    """
    import glob

    data = []
    fasta_files = glob.glob(os.path.join(fasta_dir, '*.fasta'))

    if not fasta_files:
        print(f'在 {fasta_dir} 中未找到FASTA文件')
        return data

    print(f'找到 {len(fasta_files)} 个FASTA文件')

    for fasta_file in fasta_files:
        sequences = read_fasta(fasta_file)
        for seq_id, seq_str in sequences:
            # 这里需要确定标签
            # 如果没有提供label，可以根据文件名或序列描述来判断
            if label is not None:
                data.append((seq_str, label))
            else:
                # 尝试从文件名中推断标签（可以根据实际情况修改）
                # 例如：promoter相关的文件是1，其他是0
                filename = os.path.basename(fasta_file).lower()
                if 'promoter' in filename:
                    data.append((seq_str, 1))
                else:
                    data.append((seq_str, 0))

    print(f'共读取 {len(data)} 条序列')
    return data


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True):
    """
    划分训练集、验证集和测试集

    Args:
        data: 数据列表 [(seq, label), ...]
        train_ratio, val_ratio, test_ratio: 划分比例
        shuffle: 是否打乱数据

    Returns:
        (train_data, val_data, test_data)
    """
    import random

    if shuffle:
        random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f'数据划分:')
    print(f'  训练集: {len(train_data)}')
    print(f'  验证集: {len(val_data)}')
    print(f'  测试集: {len(test_data)}')

    return train_data, val_data, test_data
