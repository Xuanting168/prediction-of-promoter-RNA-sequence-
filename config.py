# config.py
import os
from pathlib import Path


def is_server_environment():
    """判断是否在服务器环境中运行"""
    return os.path.exists('/data')


# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 服务器路径配置
if is_server_environment():
    print("检测到服务器环境")
    # 服务器数据目录
    JSON_DATA_DIR = '/data/RNAcentral/json/'
    FASTA_DATA_DIR = os.path.join(BASE_DIR, 'fasta_output')
else:
    print("检测到本地环境")
    # 本地数据目录
    JSON_DATA_DIR = os.path.join(Path.home(), 'RNA_progamme')
    FASTA_DATA_DIR = os.path.join(BASE_DIR, 'fasta_output')

# 创建目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
os.makedirs(FASTA_DATA_DIR, exist_ok=True)

# 创建目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    'freeze_rnafm': True,          # 初始冻结 RNA-FM
    'rnafm_layers': 12,             # RNA-FM 层数
    'rnafm_dim': 640,               # RNA-FM 特征维度
    'classifier_hidden_dims': [320, 160],  # 分类头隐藏层
    'dropout': 0.1,                  # Dropout 率
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-3,
    'num_epochs': 50,
    'patience': 10,                  # 早停耐心值
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
}

# 微调配置
FINETUNE_CONFIG = {
    'batch_size': 4,
    'learning_rate': 1e-5,
    'num_epochs': 30,
    'unfreeze_layers': [9, 10, 11],  # 解冻最后几层
}

# 数据配置
DATA_CONFIG = {
    'max_seq_length': 512,           # 最大序列长度
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}