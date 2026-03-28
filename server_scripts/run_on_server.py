#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器一键运行脚本 - 完整版
使用步骤：
1. 将整个MyProgamme文件夹放到RhoFold-main文件夹下
2. 运行此脚本
"""

import sys
import os

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)

from config import JSON_DATA_DIR, FASTA_DATA_DIR, is_server_environment


def main():
    print("=" * 60)
    print("RNA 启动子预测 - 服务器运行脚本")
    print("=" * 60)

    # 检查环境
    if not is_server_environment():
        print("\n⚠️  警告: 未检测到服务器环境 (/data目录不存在)")
        print("   这可能会使用本地路径")
        print()

    print(f"\n数据目录: {JSON_DATA_DIR}")
    print(f"输出目录: {FASTA_DATA_DIR}")
    print()

    # 第一步：转换JSON到FASTA
    print("\n[1/3] 将JSON转换为FASTA格式...")
    from json_to_fasta import batch_convert
    if os.path.exists(JSON_DATA_DIR):
        batch_convert(JSON_DATA_DIR, FASTA_DATA_DIR)
    else:
        print(f"   ⚠️  警告: JSON目录不存在: {JSON_DATA_DIR}")
        print("   请确认数据路径是否正确")
        return

    # 第二步：读取FASTA数据
    print("\n[2/3] 读取FASTA数据...")
    from data_loader import read_all_fasta, split_data

    # ==============================================
    # 标签设置说明：
    #
    # 你的情况：所有 JSON 文件都是启动子序列
    #
    # label=1  表示：这个序列是启动子
    # label=0  表示：这个序列不是启动子
    # label=None 表示：根据文件名推断（文件名包含'promoter'就标记为1）
    #
    # 因为你的数据全是启动子，所以设置 label=1！
    # ==============================================

    # 读取所有FASTA数据，设置为启动子（label=1）
    data = read_all_fasta(FASTA_DATA_DIR, label=1)

    if not data:
        print("   没有读取到数据，退出")
        return

    print(f"   成功读取 {len(data)} 条序列")
    print(f"   标签设置：所有序列都是启动子 (label=1)")

    # 划分数据集
    train_data, val_data, test_data = split_data(data)

    # 第三步：训练模型
    print("\n[3/3] 训练模型...")
    from data_loader import create_dataloaders
    from promoter_predictor import PromoterPredictor
    from train import train_model, evaluate_model
    import torch

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size=8
    )

    model = PromoterPredictor(freeze_rnafm=True)
    best_acc = train_model(model, train_loader, val_loader)

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")

    # 测试
    print("\n测试模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(PROJECT_DIR, 'models', 'best_promoter_predictor.pt')))
    test_acc = evaluate_model(model, test_loader, device)
    print(f"测试准确率: {test_acc:.2f}%")

    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
