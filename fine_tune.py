#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fine_tune.py - 微调模型

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RhoFold-main'))

from promoter_predictor import PromoterPredictor
from train import train_model, evaluate_model
from data_loader import create_dataloaders
from utils import set_seed, prepare_data_from_folders, split_data
from config import (
    FINETUNE_CONFIG, TRAIN_CONFIG, DATA_CONFIG,
    RAW_DATA_DIR, MODEL_DIR
)


def finetune_model():
    """
    微调模型

    什么是微调？
    - 先在一般数据上训练（冻结预训练模型，只训练分类头）
    - 然后解冻部分预训练模型，在特定数据上微调
    - 这样既能利用预训练知识，又能适应特定任务

    这个阶段：
    - 加载第一阶段训练好的模型
    - 解冻 RNA-FM 的最后几层
    - 在特定生物环境数据上微调
    """
    set_seed(42)

    # ========== 1. 加载已训练模型 ==========
    print("第一步：加载预训练模型")
    model = PromoterPredictor(freeze_rnafm=False)
    best_model_path = os.path.join(MODEL_DIR, 'best_promoter_predictor.pt')

    if not os.path.exists(best_model_path):
        print(f"❌ 错误: 找不到预训练模型！")
        print(f"请先运行 train.py 进行第一阶段训练")
        print(f"模型路径: {best_model_path}")
        return

    model.load_state_dict(torch.load(best_model_path))
    print("✅ 模型加载成功")

    # ========== 2. 解冻策略 ==========
    print("\n第二步：设置模型训练参数")

    # 先冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 RNA-FM 的最后几层
    print(f"解冻 RNA-FM 第 {FINETUNE_CONFIG['unfreeze_layers']} 层...")
    for layer_idx in FINETUNE_CONFIG['unfreeze_layers']:
        for param in model.rnafm.layers[layer_idx].parameters():
            param.requires_grad = True

    # 确保分类头也可以训练
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")

    # ========== 3. 准备数据 ==========
    print("\n第三步：准备特定生物环境的数据")

    specific_promoter_dir = os.path.join(RAW_DATA_DIR, 'specific_promoters')
    specific_non_promoter_dir = os.path.join(RAW_DATA_DIR, 'specific_non_promoters')

    if not os.path.exists(specific_promoter_dir) or not os.path.exists(specific_non_promoter_dir):
        print("⚠️  找不到特定数据，使用原始数据...")
        specific_promoter_dir = os.path.join(RAW_DATA_DIR, 'promoters')
        specific_non_promoter_dir = os.path.join(RAW_DATA_DIR, 'non_promoters')

    data = prepare_data_from_folders(
        specific_promoter_dir,
        specific_non_promoter_dir,
        max_length=DATA_CONFIG['max_seq_length']
    )

    train_data, val_data, test_data = split_data(data)

    print(f"\n第四步：创建数据加载器")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=FINETUNE_CONFIG['batch_size']
    )

    # ========== 4. 微调 ==========
    print("\n第五步：开始微调")
    best_acc = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=FINETUNE_CONFIG['num_epochs'],
        learning_rate=FINETUNE_CONFIG['learning_rate']
    )

    print(f"✅ 微调完成！最佳验证准确率: {best_acc:.2f}%")

    # ========== 5. 保存微调后的模型 ==========
    finetuned_model_path = os.path.join(MODEL_DIR, 'finetuned_promoter_predictor.pt')
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"微调后的模型保存到: {finetuned_model_path}")

    # ========== 6. 测试 ==========
    print("\n第六步：测试微调后的模型")
    device = torch.device(TRAIN_CONFIG['device'])
    model.load_state_dict(torch.load(finetuned_model_path))
    test_acc = evaluate_model(model, test_loader, device)
    print(f"微调后测试准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    finetune_model()
