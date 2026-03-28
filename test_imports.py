#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_imports.py - 测试依赖和配置是否正确

"""
这个脚本能帮助你验证：
1. 所有依赖库是否正确安装
2. 路径配置是否正确
3. RNA-FM 模型是否能正常加载
4. 设备配置是否正确（CPU/GPU）

运行方法：
    python test_imports.py
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RhoFold-main'))


def test_basic_imports():
    """测试基本依赖库的导入"""
    print("=" * 50)
    print("1. 测试基本依赖库")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")

        import numpy
        print(f"✅ NumPy: {numpy.__version__}")

        import tqdm
        print(f"✅ tqdm: {tqdm.__version__}")

        from Bio import SeqIO
        print(f"✅ Biopython: 已安装")

        from rhofold.model.rna_fm import Alphabet, BatchConverter
        print(f"✅ RNA-FM: 可导入")

        from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
        print(f"✅ RNA-FM 预训练模型: 可导入")

        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False


def test_device_config():
    """测试设备配置"""
    print("\n" + "=" * 50)
    print("2. 测试设备配置")

    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"  - GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        return 'cuda'
    else:
        print("⚠️  CUDA 不可用，将使用 CPU")
        return 'cpu'


def test_rnafm_model():
    """测试 RNA-FM 模型是否能正常加载"""
    print("\n" + "=" * 50)
    print("3. 测试 RNA-FM 模型加载")

    try:
        from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
        model, alphabet = esm1b_rna_t12()

        print("✅ RNA-FM 模型加载成功")
        print(f"  - 字母表大小: {len(alphabet)}")
        print(f"  - 模型层数: {model.args.layers}")
        print(f"  - 嵌入维度: {model.args.embed_dim}")
        print(f"  - 注意力头数: {model.args.attention_heads}")

        return model, alphabet

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_data_handling():
    """测试数据处理功能"""
    print("\n" + "=" * 50)
    print("4. 测试数据处理")

    _, alphabet = test_rnafm_model()

    if alphabet is None:
        print("⚠️  跳过数据处理测试（字母表未加载）")
        return

    batch_converter = alphabet.get_batch_converter()

    # 测试转换功能
    data = [("test1", "AUGCGUACGUAUCGA"), ("test2", "CGUACGUAGCUAGCU")]
    labels, strs, tokens = batch_converter(data)

    print("✅ 批次转换功能正常")
    print(f"  - 输入批次大小: {len(data)}")
    print(f"  - 输出 tokens 形状: {tokens.shape}")
    print(f"  - 第一个序列: {strs[0]}")
    print(f"  - 第一个序列的 tokens: {tokens[0]}")


def test_model_forward_pass():
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("5. 测试模型前向传播")

    device = test_device_config()

    # 初始化模型
    from promoter_predictor import PromoterPredictor
    model = PromoterPredictor(freeze_rnafm=True)
    model.to(device)

    # 创建一个简单的输入
    from rhofold.model.rna_fm.pretrained import esm1b_rna_t12
    _, alphabet = esm1b_rna_t12()
    batch_converter = alphabet.get_batch_converter()

    data = [("test", "AUGCGUACGUAUCGAUCG")]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    # 前向传播
    model.eval()
    import torch
    with torch.no_grad():
        logits, reprs = model(tokens)

    print("✅ 模型前向传播成功")
    print(f"  - 输入形状: {tokens.shape}")
    print(f"  - 输出 logits 形状: {logits.shape}")
    print(f"  - 预测结果: {logits.argmax(dim=1).item()}")
    print(f"  - 概率分布: {logits.softmax(dim=1)}")


def main():
    print("=" * 50)
    print("RNA 启动子预测项目 - 环境测试")
    print("=" * 50)

    all_passed = True

    # 1. 测试导入
    if not test_basic_imports():
        print("\n❌ 请检查依赖库是否正确安装")
        print("运行 pip install -r requirements.txt")
        return

    # 2. 测试设备
    test_device_config()

    # 3. 测试模型
    model, _ = test_rnafm_model()

    if model is not None:
        # 4. 测试数据处理
        test_data_handling()

        # 5. 测试前向传播
        test_model_forward_pass()

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

    print("\n下一步：")
    print("1. 在 data/raw 目录下准备训练数据")
    print("2. 运行 python train.py 进行第一阶段训练")
    print("3. 运行 python fine_tune.py 进行微调")
    print("4. 运行 python predict.py <序列> 进行预测")


if __name__ == "__main__":
    main()
