# RNA 启动子预测项目

## 项目简介

这是一个基于 **RNA-FM 预训练模型**的 RNA 启动子预测工具，支持服务器和本地两种运行模式。项目包含完整的训练、微调、预测流程。

---

## 需要借用一部分 RNA-FM 文件夹！

我们的项目默认是：
```
文件结构：
├── RhoFold-main/          <-- RNA-FM 已在这里
│   └── rhofold/
│       └── model/
│           └── rna_fm/  <-- RNA-FM 原文件
└── MyProgamme/            <-- 你的项目
    ├── 所有代码文件...
```

**代码里用 `sys.path` 告诉 Python "去 ../RhoFold-main 找 RNA-FM"，所以不需要复制。**

这样的好处：
1. 不重复代码
2. RhoFold 更新时不需要重新复制
3. 项目更整洁


## 服务器运行模式

### 服务器目录结构

在服务器上，请按以下结构放置文件：

```
RhoFold-main/
└── MyProgamme/
    ├── 所有代码文件...
    ├── fasta_output/        # JSON 转换后的 FASTA 文件（自动生成）
    ├── models/             # 训练好的模型（自动生成）
    └── results/            # 结果保存
```

### 服务器数据路径

- **JSON 数据位置**：`/data/RNAcentral/json/`（所有 JSON 文件放这里）
- **FASTA 输出位置**：`MyProgamme/fasta_output/`（自动生成）
- **模型保存位置**：`MyProgamme/models/`

### 服务器使用步骤

#### 方法 1：一键运行（推荐）

```bash
cd /path/to/RhoFold-main/MyProgamme
python server_scripts/run_on_server.py
```

这个脚本会自动：
1. 检测服务器环境
2. 将 JSON 转换为 FASTA
3. 读取 FASTA 数据
4. 训练模型

#### 方法 2：分步运行

##### 第一步：转换 JSON 到 FASTA

```bash
python json_to_fasta.py
# 或手动指定路径
python json_to_fasta.py /data/RNAcentral/json/ -o ./fasta_output
```

##### 第二步：准备训练数据

**重要：标签设置**

你需要根据实际数据确定哪些是启动子（1），哪些不是（0）。有几种方式：

**方式 1：按文件名推断**（已实现）
- 在 `data_loader.py` 中的 `read_all_fasta()` 函数里实现
- 包含 'promoter' 的文件标记为 1，其他为 0

**方式 2：分成两个文件夹**（推荐）
```python
# 修改 server_scripts/run_on_server.py 中的数据读取部分
promoter_data = read_all_fasta('/path/to/promoters', label=1)
non_promoter_data = read_all_fasta('/path/to/non_promoters', label=0)
data = promoter_data + non_promoter_data
```

**方式 3：在代码中直接设置**
```python
# 在 server_scripts/run_on_server.py 中修改
data = read_all_fasta(FASTA_DATA_DIR, label=1)  # 所有序列都是启动子
# 或
data = read_all_fasta(FASTA_DATA_DIR, label=0)  # 所有序列都是非启动子
```

##### 第三步：训练模型

```bash
python train.py
python fine_tune.py  # 可选，第二阶段微调
```

---

## 本地运行模式

### 本地配置

在 `config.py` 中会自动检测是否在服务器上运行（检查 `/data` 目录是否存在）。

### 本地使用步骤

#### 第一步：安装依赖

```bash
cd MyProgamme
pip install -r requirements.txt
```

#### 第二步：准备数据

在 `data/raw/` 目录下创建：
- `promoters/`：放启动子 FASTA 文件
- `non_promoters/`：放非启动子 FASTA 文件

#### 第三步：测试环境

```bash
python test_imports.py
```

#### 第四步：训练

```bash
python train.py
python fine_tune.py  # 可选
```

#### 第五步：预测

```bash
python predict.py 'AUGCGUACGUAUCGA'
```

---

## 详细代码解释

### promoter_predictor.py - 核心模型

**这个文件做了什么：**
1. 加载 RNA-FM 预训练模型
2. 添加一个简单的分类头
3. 处理前向传播

**关键参数：**
- `freeze_rnafm`: 是否冻结 RNA-FM 权重
  - 训练阶段设为 `True`（只训练分类头）
  - 微调阶段设为 `False`（允许训练部分层）

### json_to_fasta.py - JSON 转 FASTA 工具

**这个文件做了什么：**
1. 读取 RNAcentral 的 JSON 文件
2. 提取序列信息
3. 转换为 FASTA 格式

**特点：**
- 支持批量转换
- 自动检测服务器/本地环境
- 输出到 fasta_output 文件夹

### server_scripts/run_on_server.py - 服务器一键运行

**这个文件做了什么：**
1. 检测服务器/本地环境
2. 自动转换 JSON 到 FASTA
3. 调用训练流程

**需要修改的部分：**
```python
# 在 run_on_server.py 中需要修改标签设置
print("⚠️  注意: 你需要确认如何给序列标注标签")
```

---

## 配置说明

在 `config.py` 中可以修改：

### 模型配置

```python
MODEL_CONFIG = {
    'freeze_rnafm': True,        # 是否冻结 RNA-FM
    'rnafm_layers': 12,           # RNA-FM 层数
    'rnafm_dim': 640,             # 特征维度
    'dropout': 0.1,                # Dropout 率
}
```

### 训练配置

```python
TRAIN_CONFIG = {
    'batch_size': 8,            # 批次大小（显存小就调小）
    'learning_rate': 1e-3,       # 学习率
    'num_epochs': 50,            # 最大训练轮数
    'patience': 10,              # 早停耐心值
}
```

### 服务器配置

```python
# 服务器环境
SERVER_JSON_DIR = '/data/RNAcentral/json/'
SERVER_FASTA_DIR = '/data/RNAcentral/fasta/'

# 本地环境
LOCAL_JSON_DIR = './json_data/'
LOCAL_FASTA_DIR = './fasta_output/'

# 自动检测
if os.path.exists('/data'):
    JSON_DATA_DIR = SERVER_JSON_DIR
    FASTA_DATA_DIR = SERVER_FASTA_DIR
else:
    JSON_DATA_DIR = LOCAL_JSON_DIR
    FASTA_DATA_DIR = LOCAL_FASTA_DIR
```

---

## 使用流程图

### 服务器运行流程

```
开始
  ↓
运行 server_scripts/run_on_server.py
  ↓
[1/3] 检测服务器环境
  ↓
[2/3] JSON 转 FASTA
  ↓
[3/3] 训练模型
  ↓
得到 best_promoter_predictor.pt
  ↓
可选：运行 fine_tune.py
  ↓
得到 finetuned_promoter_predictor.pt
  ↓
运行 predict.py
  ↓
完成！
```

### 本地运行流程

```
开始
  ↓
运行 test_imports.py
  ↓
准备数据（data/raw/）
  ↓
运行 train.py（第一阶段）
  ↓
得到 best_promoter_predictor.pt
  ↓
可选：运行 fine_tune.py（第二阶段）
  ↓
得到 finetuned_promoter_predictor.pt
  ↓
运行 predict.py
  ↓
完成！
```

---

## 安装依赖

### 基础依赖

```bash
pip install torch tqdm biopython ml_collections scikit-learn pandas numpy
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

---

## 常见问题

### 问题 1: 找不到 RhoFold 模块

**错误信息：** `ModuleNotFoundError: No module named 'rhofold'`

**解决方法：**
1. 确保 MyProgamme 在 RhoFold-main 文件夹下
2. 检查路径设置是否正确

### 问题 2: CUDA 显存不足

**错误信息：** `RuntimeError: CUDA out of memory`

**解决方法：**
1. 在 `config.py` 中减小 `batch_size`（从 8 改到 4 或 2）
2. 使用 CPU（训练会比较慢，但一定能运行）

### 问题 3: JSON 文件未找到

**错误信息：** `在 /data/RNAcentral/json/ 中未找到JSON文件`

**解决方法：**
1. 确认服务器上有 `/data/RNAcentral/json/` 目录
2. 检查该目录下是否有 .json 文件
3. 如果在本地运行，确认 config.py 中的 LOCAL_JSON_DIR 路径正确

---

## 标签设置示例

**重要提示：标签设置是项目成功的关键！**

```python
# 在 server_scripts/run_on_server.py 中的示例
from data_loader import read_all_fasta, split_data

# 方法 1：读取启动子和非启动子两个文件夹
promoter_data = read_all_fasta('/data/promoter_fasta', label=1)
non_promoter_data = read_all_fasta('/data/non_promoter_fasta', label=0)
data = promoter_data + non_promoter_data

# 方法 2：使用文件名推断
data = read_all_fasta('/data/RNAcentral/fasta', label=None)

# 方法 3：所有序列都是启动子
data = read_all_fasta('/data/RNAcentral/fasta', label=1)
```

---

## 总结

### RNA-FM 的使用方式

**你不需要复制 RNA-FM 文件夹！**

- **RhoFold-main/rhofold/model/rna_fm/** 是源文件位置
- **代码中的路径设置** 告诉 Python 去哪里找它
- 这种方式保持了代码整洁，避免重复

### 训练阶段

1. **第一阶段（train.py）**：只训练分类头，学习率大
2. **第二阶段（fine_tune.py）**：解冻最后几层，学习率小
3. **预测阶段（predict.py）**：使用训练好的模型

### 环境要求

- **服务器**：要求有 `/data` 目录
- **GPU**：推荐 NVIDIA Tesla V100 或更好
- **本地**：16GB RAM 可运行，有 GPU 更好

---

## 版本说明

| 版本 | 变更 | 日期 |
|------|------|------|
| 1.0  | 初始版本，本地运行 | 2026-3-27 |
| 1.1  | 添加服务器支持 | 2026-3-27 |
| 1.2  | 整合所有文档 | 2026-03-28 |
