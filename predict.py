#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# predict.py - 使用训练好的模型进行预测

import torch
import sys
import os

# 添加项目路径和 RhoFold 路径
# 这样 Python 就能找到我们的代码和 RhoFold 库
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RhoFold-main'))

from promoter_predictor import PromoterPredictor
from utils import set_seed
from config import MODEL_DIR, TRAIN_CONFIG


def predict_single_sequence(sequence: str, model_path: str = None):
    """
    预测单个序列是否为启动子

    参数:
        sequence: RNA 序列字符串（支持 DNA 或 RNA 格式）
        model_path: 模型文件路径，如果不指定则使用默认路径

    返回:
        dict: 包含预测结果的字典
    """
    # 设置随机种子，确保每次预测结果一致
    set_seed(42)

    # 默认使用最佳模型
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_promoter_predictor.pt')

    print(f"使用模型: {model_path}")

    # 创建模型实例
    model = PromoterPredictor(freeze_rnafm=True)

    # 加载模型参数
    try:
        model.load_state_dict(torch.load(model_path, map_location=TRAIN_CONFIG['device']))
    except FileNotFoundError:
        print(f"错误: 找不到模型文件！请先运行 train.py 训练模型")
        return None

    # 将模型移动到指定设备（CPU 或 GPU）
    model.to(TRAIN_CONFIG['device'])
    # 切换到评估模式（不进行梯度计算）
    model.eval()

    # 预处理序列
    # 1. 转换为大写
    # 2. 将 T 替换为 U（因为是 RNA 模型）
    sequence = sequence.upper().replace('T', 'U')
    print(f"预处理后的序列: {sequence}")

    # 转换为 RNA-FM 可以接受的输入格式
    # 1. 获取 RNA-FM 的字母表（字典）
    from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
    _, alphabet = esm1b_rna_t12()

    # 2. 批次转换器：将字符串序列转换为模型可以接受的张量
    batch_converter = alphabet.get_batch_converter()

    # 3. 准备输入：[("序列名", "序列")]，这样能保持格式一致
    data = [("query", sequence)]
    _, _, tokens = batch_converter(data)

    # 4. 将张量移动到指定设备
    tokens = tokens.to(TRAIN_CONFIG['device'])

    # 进行预测
    with torch.no_grad():
        logits, _ = model(tokens)

        # 计算概率（使用 softmax）
        probs = torch.softmax(logits, dim=1)

        # 获取预测结果
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probs[0, prediction].item()

    # 整理结果
    result = {
        'sequence': sequence,
        'is_promoter': bool(prediction == 1),
        'promoter_probability': float(probs[0, 1]),
        'non_promoter_probability': float(probs[0, 0]),
        'confidence': confidence
    }

    return result


def print_prediction_result(result):
    """
    格式化打印预测结果
    """
    if result is None:
        return

    print("=" * 50)
    print(f"序列: {result['sequence']}")
    print("-" * 50)

    if result['is_promoter']:
        print(f"✅  预测结果: 启动子")
        print(f"📊  置信度: {result['confidence']:.2%}")
        print(f"🎯  启动子概率: {result['promoter_probability']:.2%}")
        print(f"📉  非启动子概率: {result['non_promoter_probability']:.2%}")
    else:
        print(f"❌  预测结果: 非启动子")
        print(f"📊  置信度: {result['confidence']:.2%}")
        print(f"🎯  启动子概率: {result['promoter_probability']:.2%}")
        print(f"📉  非启动子概率: {result['non_promoter_probability']:.2%}")
    print("=" * 50)


def main():
    """
    主函数，接受命令行参数
    """
    if len(sys.argv) < 2:
        print("使用方法: python predict.py <RNA序列>")
        print("")
        print("示例:")
        print("  python predict.py 'AUGCGUACGUAUCGAUCG'")
        print("  python predict.py 'ATGCGTACGTATCGATCG'  # DNA 序列会自动转换为 RNA")
        print("")
        print("参数说明:")
        print("  <RNA序列>: 可以是 DNA 序列（T 会自动转换为 U）")
        return

    # 获取命令行参数
    sequence = sys.argv[1]

    # 进行预测
    result = predict_single_sequence(sequence)

    # 打印结果
    if result:
        print_prediction_result(result)


if __name__ == "__main__":
    main()
