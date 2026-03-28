#!/usr/bin/env python3
"""
JSON到FASTA格式转换脚本
用于将RNAcentral的JSON数据转换为FASTA格式
"""

import json
import os
import glob
from pathlib import Path


def json_to_fasta(json_file, output_fasta=None):
    """
    将单个JSON文件转换为FASTA文件

    Args:
        json_file: 输入的JSON文件路径
        output_fasta: 输出的FASTA文件路径（可选，默认与JSON同目录同名）
    """
    if output_fasta is None:
        output_fasta = Path(json_file).with_suffix('.fasta')

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取results中的序列
    sequences = data.get('results', [])

    # 写入FASTA文件
    with open(output_fasta, 'w', encoding='utf-8') as f:
        for seq_data in sequences:
            # 获取序列信息
            seq_id = seq_data.get('rnacentral_id', '')
            description = seq_data.get('description', '')
            sequence = seq_data.get('sequence', '')

            if not seq_id or not sequence:
                continue

            # 写入FASTA格式：>ID 描述
            #               序列
            f.write(f'>{seq_id} {description}\n')
            f.write(f'{sequence}\n')

    print(f'转换完成: {json_file} -> {output_fasta}')
    print(f'  共转换 {len(sequences)} 条序列')

    return output_fasta


def batch_convert(json_dir, output_dir=None):
    """
    批量转换目录下的所有JSON文件

    Args:
        json_dir: 包含JSON文件的目录
        output_dir: 输出目录（可选，默认与输入目录相同）
    """
    if output_dir is None:
        output_dir = json_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    if not json_files:
        print(f'在 {json_dir} 中未找到JSON文件')
        return

    print(f'找到 {len(json_files)} 个JSON文件')
    print('-' * 50)

    # 逐个转换
    for json_file in json_files:
        basename = Path(json_file).stem
        output_fasta = os.path.join(output_dir, f'{basename}.fasta')
        json_to_fasta(json_file, output_fasta)


def main():
    """
    主函数 - 支持命令行运行
    """
    import argparse

    parser = argparse.ArgumentParser(description='JSON到FASTA格式转换工具')
    parser.add_argument('input', help='输入的JSON文件或目录')
    parser.add_argument('-o', '--output', help='输出的FASTA文件或目录（可选）')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        # 批量转换
        batch_convert(args.input, args.output)
    elif os.path.isfile(args.input):
        # 单个文件转换
        json_to_fasta(args.input, args.output)
    else:
        print(f'错误: {args.input} 不存在')


if __name__ == '__main__':
    # ==========================================
    # 服务器运行配置 - 可直接运行此脚本
    # ==========================================

    # 服务器上的数据目录
    SERVER_JSON_DIR = '/data/RNAcentral/json/'

    # 输出目录（可以根据需要修改）
    # 假设MyProgamme放在RhoFold-main下面
    SCRIPT_DIR = Path(__file__).parent
    SERVER_OUTPUT_DIR = SCRIPT_DIR / 'fasta_output'

    # 检查是否在服务器上运行（简单判断：检查/data目录是否存在）
    if os.path.exists('/data'):
        print('检测到服务器环境，使用服务器路径...')
        print(f'输入目录: {SERVER_JSON_DIR}')
        print(f'输出目录: {SERVER_OUTPUT_DIR}')
        print('-' * 50)

        batch_convert(SERVER_JSON_DIR, str(SERVER_OUTPUT_DIR))
    else:
        # 本地运行模式 - 可以在这里测试
        print('本地运行模式，请使用命令行参数指定输入文件/目录')
        print('使用示例:')
        print('  python json_to_fasta.py /path/to/input.json')
        print('  python json_to_fasta.py /path/to/json_dir -o /path/to/output_dir')

        # 或者直接运行main函数（如果提供了命令行参数）
        import sys
        if len(sys.argv) > 1:
            main()
