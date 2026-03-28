#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py - 训练启动子预测模型

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os

# 添加项目路径和 RhoFold 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RhoFold-main'))

from promoter_predictor import PromoterPredictor
from data_loader import create_dataloaders
from utils import set_seed, prepare_data_from_folders, split_data
from config import (
    TRAIN_CONFIG, MODEL_CONFIG, DATA_CONFIG,
    RAW_DATA_DIR, MODEL_DIR
)


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    save_dir: str = MODEL_DIR
):
    """
    训练模型的核心函数

    训练流程：
    1. 初始化损失函数和优化器
    2. 循环训练 num_epochs 轮
    3. 每轮计算训练损失和准确率
    4. 验证模型性能
    5. 保存最佳模型

    参数:
        model: 模型实例
        train_loader, val_loader: 训练和验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        save_dir: 模型保存目录

    返回:
        best_val_acc: 最佳验证集准确率
    """
    device = torch.device(TRAIN_CONFIG['device'])
    model.to(device)

    # ========== 配置训练过程 ==========
    # 1. 损失函数：交叉熵损失（适合二分类）
    criterion = nn.CrossEntropyLoss()

    # 2. 优化器：Adam（常用的自适应优化器）
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )

    best_val_acc = 0.0
    patience_counter = 0
    patience = TRAIN_CONFIG.get('patience', 10)

    # ========== 开始训练 ==========
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")

        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 使用 tqdm 显示进度条
        for tokens, labels in tqdm(train_loader, desc="训练"):
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, _ = model(tokens)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * tokens.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss /= train_total

        # ========== 验证阶段 ==========
        val_acc = evaluate_model(model, val_loader, device)

        print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | "
              f"验证准确率: {val_acc:.2f}%")

        # ========== 保存最佳模型 ==========
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(save_dir, 'best_promoter_predictor.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 保存最佳模型到: {best_model_path} (验证准确率: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1

        # ========== 早停机制 ==========
        if patience_counter >= patience:
            print(f"❌ 验证准确率不再提升，停止训练（耐心值: {patience}）")
            break

    print(f"\n{'='*50}")
    return best_val_acc


def evaluate_model(model, val_loader, device):
    """
    评估模型性能

    参数:
        model: 模型实例
        val_loader: 验证数据加载器
        device: 设备

    返回:
        acc: 准确率
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for tokens, labels in tqdm(val_loader, desc="评估"):
            tokens = tokens.to(device)
            labels = labels.to(device)

            logits, _ = model(tokens)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def main():
    # 1. 初始化随机种子
    set_seed(42)

    print("启动 RNA 启动子预测模型训练...")

    # 2. 准备数据
    print("\n第一步：准备数据")
    promoter_dir = os.path.join(RAW_DATA_DIR, 'promoters')
    non_promoter_dir = os.path.join(RAW_DATA_DIR, 'non_promoters')

    if not os.path.exists(promoter_dir) or not os.path.exists(non_promoter_dir):
        print(f"⚠️  警告: 数据文件夹不存在！")
        print(f"请在 data/raw/ 下创建以下文件夹：")
        print(f"  - promoters/    放置启动子序列的 FASTA 文件")
        print(f"  - non_promoters/ 放置非启动子序列的 FASTA 文件")
        return

    data = prepare_data_from_folders(
        promoter_dir,
        non_promoter_dir,
        max_length=DATA_CONFIG['max_seq_length']
    )

    train_data, val_data, test_data = split_data(
        data,
        train_ratio=DATA_CONFIG['train_ratio'],
        val_ratio=DATA_CONFIG['val_ratio']
    )

    print(f"\n第二步：创建数据加载器")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=TRAIN_CONFIG['batch_size']
    )

    print(f"\n第三步：初始化模型")
    model = PromoterPredictor(freeze_rnafm=MODEL_CONFIG['freeze_rnafm'])

    print(f"\n第四步：开始训练")
    best_acc = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=TRAIN_CONFIG['num_epochs'],
        learning_rate=TRAIN_CONFIG['learning_rate']
    )

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")

    print(f"\n第五步：测试模型")
    device = torch.device(TRAIN_CONFIG['device'])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_promoter_predictor.pt')))
    test_acc = evaluate_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
