# DeepGOPlus - 蛋白质功能预测工具

DeepGOPlus 是一个用于蛋白质功能预测的深度学习工具，结合了卷积神经网络（CNN）和序列相似性搜索（DIAMOND）来预测蛋白质的 Gene Ontology (GO) 注释。

## 功能特性

- 支持多个 GO 本体：**MF** (分子功能)、**BP** (生物过程)、**CC** (细胞组分)、**PF** (蛋白质家族)
- 结合深度学习模型和序列相似性搜索
- 支持大规模蛋白质序列预测
- 自动处理超长序列（自动分段）
- 完整的评估和分析工具

## 环境要求

- Python 3.6+
- TensorFlow 2.x / Keras
- DIAMOND (用于序列相似性搜索)
- 其他依赖：pandas, numpy, click, tqdm, scikit-learn

## 安装

### 1. 安装 Python 依赖
ssh -p 46454 root@connect.westc.gpuhub.com
```bash
pip install tensorflow pandas numpy click tqdm scikit-learn
```

### 2. 安装 DIAMOND

**Linux:**
```bash
wget http://github.com/bbuchfink/diamond/releases/download/v2.0.2/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
sudo mv diamond /usr/local/bin/
```

**macOS:**
```bash
brew install diamond
```

**或使用 Docker:**
```bash
docker build -t deepgoplus docker/
```

## 快速开始

### 使用完整流程脚本（推荐）

最简单的方式是使用完整流程脚本，支持参数化配置：

```bash
# 运行完整流程（MF本体，galaxy数据集，run 1）
python run_complete_pipeline.py \
    --ont mf \
    --dataset galaxy \
    --run 1 \
    --data-root galaxy/ \
    --epochs 1 \
    --batch-size 32 \
    --device gpu:0 \
    --save-separate

# 运行完整流程（CC本体，cafa数据集，run 1）
python run_complete_pipeline.py \
    --ont cc \
    --dataset galaxy \
    --run 1 \
    --data-root galaxy/ \
    --epochs 1 \
    --batch-size 64 \
    --device gpu:0 \
    --save-separate
    
python run_complete_pipeline.py \
    --ont bp \
    --dataset galaxy \
    --run 1 \
    --data-root galaxy/ \
    --epochs 1 \
    --batch-size 64 \
    --device gpu:0 \
    --save-separate
    
python run_complete_pipeline.py \
    --ont pf \
    --dataset galaxy \
    --run 1 \
    --data-root galaxy/ \
    --epochs 1 \
    --batch-size 64 \
    --device gpu:0 \
    --save-separate

# 跳过训练步骤（使用已有模型）
## galaxy
echo "EVALUATION START" 
echo "GALAXY" 
python run_complete_pipeline.py \
    --ont mf \
    --dataset galaxy \
    --run 3 \
    --data-root galaxy/ \
    --alpha 0.05 \
    --skip-train \
    --save-separate
python run_complete_pipeline.py \
    --ont cc \
    --dataset galaxy \
    --run 3 \
    --data-root galaxy/ \
    --alpha 0.05 \
    --skip-train \
    --save-separate
python run_complete_pipeline.py \
    --ont bp \
    --dataset galaxy \
    --run 3 \
    --data-root galaxy/ \
    --alpha 0.05 \
    --skip-train \
    --save-separate
    
python run_complete_pipeline.py \
    --ont pf \
    --dataset galaxy \
    --run 3 \
    --data-root galaxy/ \
    --alpha 0.05 \
    --skip-train \
    --save-separate

echo "CAFA"
python run_complete_pipeline.py \
    --ont mf \
    --dataset cafa \
    --run 3 \
    --data-root cafa/ \
    --alpha 0.9 \
    --skip-train \
    --save-separate
python run_complete_pipeline.py \
    --ont cc \
    --dataset cafa \
    --run 3 \
    --data-root cafa/ \
    --alpha 0.5 \
    --skip-train \
    --save-separate
python run_complete_pipeline.py \
    --ont bp \
    --dataset cafa \
    --run 3 \
    --data-root cafa/ \
    --alpha 0.5 \
    --skip-train \
    --save-separate
    
python run_complete_pipeline.py \
    --ont pf \
    --dataset cafa \
    --run 3 \
    --data-root cafa/ \
    --alpha 0.05 \
    --skip-train \
    --save-separate
echo "EVALUATION COMPLETE" 
```

**完整流程脚本参数：**
- `--ont`: 本体类型（mf/cc/bp/pf，必需）
- `--dataset`: 数据集类型（galaxy/cafa，必需）
- `--run`: 运行次数编号（默认：1）
- `--data-root`: 数据根目录（默认：galaxy/）
- `--epochs`: 训练轮数（默认：12）
- `--batch-size`: 批次大小（默认：32）
- `--learning-rate`: 学习率（默认：0.001）
- `--device`: 计算设备（默认：gpu:0）
- `--threshold`: 预测阈值（默认：0.0，保存所有预测）
- `--alpha`: DIAMOND和DeepGO权重（默认：0.5）
- `--skip-train`: 跳过训练步骤
- `--skip-pred`: 跳过预测步骤
- `--skip-eval`: 跳过评估步骤
- `--save-separate`: 保存DIAMOND和DeepGO的单独预测结果

**完整流程脚本会自动执行：**
1. 准备训练/验证/测试数据（PKL格式）
2. 生成术语文件
3. 训练模型
4. 创建DIAMOND数据库
5. 进行预测（结合DIAMOND和DeepGO）
6. 评估预测结果（使用修复后的评估脚本）
7. 分析预测结果（如果使用--save-separate）
8. 诊断DeepGO模型（如果使用--save-separate）

## 数据准备

### 数据格式要求

1. **GO 本体文件**: `go.obo` - Gene Ontology 标准格式文件
2. **训练数据**: TSV 格式的标注文件 + FASTA 格式的序列文件

### TSV 文件格式

支持两种格式：

**格式 1: Pairs 格式** (两列)
```
protein_id    GO_id
P12345        GO:0003674
P12345        GO:0008150
```

**格式 2: Wide 格式** (多列，包含 GO 标签列)
```
acc    sequence    GO_MF_labels              GO_BP_labels
P12345 ATGCG...    GO:0003674;GO:0005524    GO:0008150
```

### 数据目录结构

推荐的数据目录结构：

```
galaxy/  (或 cafa/)
├── go.obo                          # GO 本体文件
├── MF_train_data.tsv              # 训练数据（TSV）
├── MF_train_sequences.fasta       # 训练序列（FASTA）
├── MF_validation_data.tsv         # 验证数据（TSV）
├── MF_validation_sequences.fasta  # 验证序列（FASTA）
├── MF_test_data.tsv               # 测试数据（TSV）
├── MF_test_sequences.fasta        # 测试序列（FASTA）
├── results/                        # 结果目录（自动创建）
│   ├── mf_test_preds_galaxy_run1.tsv
│   ├── mf_evaluation_results_run1.txt
│   └── ...
```

## 手动运行步骤

如果不使用完整流程脚本，可以手动执行以下步骤：

### 1. 准备数据

```bash
# 准备训练数据
python src/data/prepare_data.py \
  --go-file galaxy/go.obo \
  --data-file galaxy/MF_train_data.tsv \
  --sequences-file galaxy/MF_train_sequences.fasta \
  --out-file galaxy/MF_train_data.pkl \
  --ont mf

# 准备验证数据
python src/data/prepare_data.py \
  --go-file galaxy/go.obo \
  --data-file galaxy/MF_validation_data.tsv \
  --sequences-file galaxy/MF_validation_sequences.fasta \
  --out-file galaxy/MF_validation_data.pkl \
  --ont mf

# 准备测试数据
python src/data/prepare_data.py \
  --go-file galaxy/go.obo \
  --data-file galaxy/MF_test_data.tsv \
  --sequences-file galaxy/MF_test_sequences.fasta \
  --out-file galaxy/MF_test_data.pkl \
  --ont mf

# 生成术语文件
python src/data/get_terms.py \
  --train-data-file galaxy/MF_train_data.pkl \
  --out-file galaxy/terms_mf.pkl
```

### 2. 训练模型

```bash
python deepgoplus/train.py \
  --go-file galaxy/go.obo \
  --train-data-file galaxy/MF_train_data.pkl \
  --valid-data-file galaxy/MF_validation_data.pkl \
  --test-data-file galaxy/MF_test_data.pkl \
  --terms-file galaxy/terms_mf.pkl \
  --model-file galaxy/model_mf.h5 \
  --out-file galaxy/MF_predictions.pkl \
  --epochs 12 \
  --batch-size 32 \
  --device gpu:0 \
  --learning-rate 0.001
```

### 3. 创建DIAMOND数据库

```bash
diamond makedb --in galaxy/MF_train_sequences.fasta -d galaxy/MF_train_data
```

### 4. 进行预测

```bash
python deepgoplus/main.py \
  --data-root galaxy/ \
  --in-file MF_test_sequences.fasta \
  --out-file results/mf_test_preds_galaxy_run1.tsv \
  --go-file go.obo \
  --model-file model_mf.h5 \
  --terms-file terms_mf.pkl \
  --annotations-file MF_train_data.pkl \
  --diamond-db MF_train_data.dmnd \
  --threshold 0.0 \
  --batch-size 32 \
  --alpha 0.5 \
  --save-separate
```

### 5. 评估预测结果

```bash
# 使用评估脚本（已修复）
python src/evaluation/evaluate_predictions.py \
    --pred-file galaxy/results/mf_test_preds_galaxy_run1.tsv \
    --true-file galaxy/MF_test_data.pkl \
    --out-file galaxy/results/mf_evaluation_results_run1.txt \
    --go-file galaxy/go.obo \
    --ont mf
```

### 6. 分析预测结果（可选）

```bash
# 分析DIAMOND和DeepGO的贡献
python src/evaluation/analyze_predictions.py \
    --pred-file galaxy/results/mf_test_preds_galaxy_run1.tsv \
    --diamond-file galaxy/results/mf_test_preds_galaxy_run1_diamond_only.tsv \
    --deep-file galaxy/results/mf_test_preds_galaxy_run1_deep_only.tsv \
    --true-file galaxy/MF_test_data.pkl \
    --train-file galaxy/MF_train_data.pkl \
    --go-file galaxy/go.obo \
    --ont mf \
    --out-file galaxy/results/analysis_report.txt

# 诊断DeepGO模型
python src/evaluation/diagnose_model.py \
    --deep-file galaxy/results/mf_test_preds_galaxy_run1_deep_only.tsv \
    --true-file galaxy/MF_test_data.pkl \
    --terms-file galaxy/terms_mf.pkl \
    --go-file galaxy/go.obo \
    --ont mf \
    --out-file galaxy/results/diagnosis_report.txt
```

## 输出格式

### TSV 格式输出（预测结果）

预测结果以长格式保存，每行一个预测：

```
protein_id	go_id	score
AF-Q9Y3D9-F1-model_v4	GO:0003723	1.0
AF-Q9Y3D9-F1-model_v4	GO:0003735	1.0
AF-P46597-F1-model_v4	GO:0003723	0.191
```

### 评估结果输出（TXT 格式）

评估脚本会生成包含以下信息的文本报告：

```
================================================================================
DeepGOPlus 预测结果评估报告（修复版本）
================================================================================

本体 (Ontology): MF

数据集信息:
  蛋白质数量: 2059
  GO术语数量: 4714
  总预测数: 1903
  总真实标签数: 2986
  真正例数: 1678

主要指标:
  Fmax: 0.9132 (阈值: 0.380)
  AUPR: 0.7118

详细指标 (最佳阈值 0.380):
  总体精确率: 0.8816
  总体召回率: 0.5613
  平均精确率: 0.9199
  平均召回率: 0.9066
  平均F1: 0.9092
================================================================================
```

## 参数说明

### 训练参数

- `--go-file`: GO 本体文件路径
- `--train-data-file`: 训练数据 PKL 文件
- `--valid-data-file`: 验证数据 PKL 文件
- `--test-data-file`: 测试数据 PKL 文件
- `--terms-file`: GO 术语列表 PKL 文件
- `--model-file`: 输出模型文件路径
- `--out-file`: 输出预测结果文件路径
- `--epochs`: 训练轮数（默认：12）
- `--batch-size`: 批次大小（默认：32）
- `--device`: 计算设备，`gpu:0` 或 `cpu:0`（默认：gpu:0）
- `--learning-rate`: 学习率（默认：0.001）

### 预测参数

- `--data-root`: 数据根目录（必需）
- `--in-file`: 输入 FASTA 文件（必需）
- `--out-file`: 输出结果文件（TSV 格式）
- `--go-file`: GO 本体文件（默认：go.obo）
- `--model-file`: 模型文件（默认：model.h5）
- `--terms-file`: GO 术语列表文件（默认：terms.pkl）
- `--annotations-file`: 训练标注文件（用于 DIAMOND 预测）
- `--diamond-db`: DIAMOND 数据库文件
- `--threshold`: 预测阈值（默认：0.1，使用 0.0 保存所有预测）
- `--alpha`: DIAMOND 和深度学习模型的权重（默认：0.5）
- `--batch-size`: 批次大小（默认：32）
- `--save-separate`: 保存DIAMOND和DeepGO的单独预测结果

### 评估参数

- `--pred-file`: 预测结果TSV文件
- `--true-file`: 真实标签PKL文件
- `--out-file`: 输出评估结果TXT文件
- `--go-file`: GO本体文件
- `--ont`: 本体类型（mf/cc/bp/pf，可选）

## 注意事项

1. **序列长度限制**: 模型支持最大 2000 个氨基酸的序列，超长序列会自动分段处理
2. **GPU 内存**: 如果遇到 GPU 内存不足，可以减小 `batch-size`
3. **DIAMOND 数据库**: 使用主预测脚本时需要先创建 DIAMOND 数据库
4. **数据格式**: 确保 TSV 文件中的 GO 术语使用分号或逗号分隔
5. **文件路径**: 
   - 如果输入文件不在当前目录，脚本会自动在 `data-root` 目录中查找
   - 输出文件如果不是绝对路径，会保存到 `data-root` 目录
6. **GO 术语格式**: 代码会自动处理包含分号分隔的多个 GO ID 的情况
7. **评估阈值**: 预测时使用 `--threshold 0.0` 可以保存所有预测结果，然后在评估时自动选择最佳阈值
8. **评估方法**: `evaluate_predictions.py` 已修复，现在可以正确计算评估指标（修复了召回率计算错误和平均值计算问题）

## 故障排除

### 常见问题

1. **找不到 train.py**: 确保文件在 `deepgoplus/` 目录中
2. **DIAMOND 命令未找到**: 确保 DIAMOND 已安装并在 PATH 中
3. **GPU 内存不足**: 减小 batch-size 或使用 CPU
4. **数据文件不存在**: 检查文件路径和数据目录结构
5. **ModuleNotFoundError: No module named 'utils'**: 
   - 已修复：代码现在使用 `from deepgoplus.utils import ...`
   - 确保从项目根目录运行脚本
6. **KeyError: 'GO:0004930;GO:0004984'**: 
   - 已修复：代码现在会自动处理包含分号分隔的多个 GO ID
7. **Error opening file test.fasta**: 
   - 已修复：脚本现在会自动在 `data-root` 目录中查找输入文件
8. **评估指标虚高**: 
   - `evaluate_predictions.py` 已修复，现在可以正确计算评估指标
   - 修复了召回率计算错误（没有真实标签时召回率应该是0，不是1.0）
   - 修复了平均值计算（只对有真实标签的蛋白质计算平均值）

## 引用

如果使用 DeepGOPlus，请引用相关论文。

## 许可证

请查看项目许可证文件。
