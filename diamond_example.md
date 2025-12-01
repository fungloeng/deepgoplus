# DIAMOND 集成使用指南

本指南介绍如何将DIAMOND序列相似性搜索集成到蛋白质功能预测模型中，参考DeepGOPlus的实现方式。

## 目录

1. [DIAMOND简介](#diamond简介)
2. [安装DIAMOND](#安装diamond)
3. [核心步骤](#核心步骤)
4. [代码实现](#代码实现)
5. [与深度学习模型结合](#与深度学习模型结合)
6. [完整示例](#完整示例)
7. [参数调优](#参数调优)

---

## DIAMOND简介

DIAMOND是一个快速的蛋白质序列相似性搜索工具，类似于BLAST但速度更快。在蛋白质功能预测中，DIAMOND用于：
- 在训练集中找到与查询序列相似的蛋白质
- 基于相似蛋白质的已知功能注释，预测查询序列的功能

**核心思想**：如果两个蛋白质序列相似，它们很可能具有相似的功能。

---

## 安装DIAMOND

### Linux
```bash
wget http://github.com/bbuchfink/diamond/releases/download/v2.0.2/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
sudo mv diamond /usr/local/bin/
```

### macOS
```bash
brew install diamond
```

### 验证安装
```bash
diamond --version
```

---

## 核心步骤

### 步骤1: 创建DIAMOND数据库

从训练集的FASTA文件创建DIAMOND数据库：

```bash
diamond makedb --in train_sequences.fasta -d train_db
```

**参数说明**：
- `--in`: 输入FASTA文件（训练集的蛋白质序列）
- `-d`: 输出数据库名称（会生成 `.dmnd` 文件）

**输出**：`train_db.dmnd`

### 步骤2: 运行DIAMOND搜索

对查询序列（测试集）进行搜索：

```bash
diamond blastp \
    -d train_db \
    --more-sensitive \
    -t /tmp \
    -q test_sequences.fasta \
    --outfmt 6 qseqid sseqid bitscore \
    -o diamond_results.txt
```

**参数说明**：
- `-d`: DIAMOND数据库文件
- `--more-sensitive`: 使用更敏感的模式（提高准确性）
- `-t`: 临时文件目录
- `-q`: 查询序列文件（要预测的序列）
- `--outfmt 6`: 输出格式6（制表符分隔）
  - `qseqid`: 查询序列ID
  - `sseqid`: 数据库序列ID
  - `bitscore`: BLAST bitscore（相似性分数）
- `-o`: 输出文件

**输出格式**（`diamond_results.txt`）：
```
query_protein_1    train_protein_A    450.2
query_protein_1    train_protein_B    380.5
query_protein_2    train_protein_C    520.1
...
```

### 步骤3: 解析DIAMOND结果并转换为GO预测

将DIAMOND的序列相似性结果转换为GO功能预测。

---

## 代码实现

### 完整Python实现

```python
import subprocess
import numpy as np
from collections import defaultdict
import pandas as pd

def create_diamond_database(fasta_file, db_name):
    """
    创建DIAMOND数据库
    
    Args:
        fasta_file: 训练集FASTA文件路径
        db_name: 数据库名称（不含.dmnd扩展名）
    """
    cmd = [
        "diamond", "makedb",
        "--in", fasta_file,
        "-d", db_name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to create DIAMOND database: {result.stderr}")
    print(f"DIAMOND database created: {db_name}.dmnd")


def run_diamond_search(query_file, db_file, output_file, temp_dir="/tmp"):
    """
    运行DIAMOND搜索
    
    Args:
        query_file: 查询序列文件（测试集FASTA）
        db_file: DIAMOND数据库文件（.dmnd）
        output_file: 输出结果文件
        temp_dir: 临时文件目录
    """
    cmd = [
        "diamond", "blastp",
        "-d", db_file,
        "--more-sensitive",
        "-t", temp_dir,
        "-q", query_file,
        "--outfmt", "6", "qseqid", "sseqid", "bitscore",
        "-o", output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"DIAMOND search failed: {result.stderr}")
    print(f"DIAMOND search completed: {output_file}")


def load_annotations(annotations_file):
    """
    加载训练集的GO注释
    
    Args:
        annotations_file: 注释文件（PKL格式，包含proteins和annotations列）
    
    Returns:
        dict: {protein_id: set(go_terms)}
    """
    df = pd.read_pickle(annotations_file)
    annotations = {}
    for row in df.itertuples():
        # 假设DataFrame有proteins和prop_annotations列
        annotations[row.proteins] = set(row.prop_annotations)
    return annotations


def parse_diamond_results(diamond_file):
    """
    解析DIAMOND搜索结果
    
    Args:
        diamond_file: DIAMOND输出文件
    
    Returns:
        dict: {query_protein_id: {similar_protein_id: bitscore}}
    """
    mapping = {}
    with open(diamond_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[0]
                similar_id = parts[1]
                bitscore = float(parts[2])
                
                if query_id not in mapping:
                    mapping[query_id] = {}
                mapping[query_id][similar_id] = bitscore
    return mapping


def diamond_to_go_predictions(mapping, annotations):
    """
    将DIAMOND相似性结果转换为GO功能预测
    
    算法：
    1. 对于每个查询蛋白质，找到所有相似蛋白质
    2. 收集所有相似蛋白质的GO注释
    3. 对每个GO术语，计算加权分数：
       score(go_id) = sum(bitscore of proteins with go_id) / sum(all bitscores)
    
    Args:
        mapping: DIAMOND结果映射 {query_id: {similar_id: bitscore}}
        annotations: 训练集注释 {protein_id: set(go_terms)}
    
    Returns:
        dict: {query_protein_id: {go_id: score}}
    """
    diamond_predictions = {}
    
    for query_id, similar_proteins in mapping.items():
        # 收集所有相似蛋白质的GO术语
        all_go_terms = set()
        total_bitscore = 0.0
        
        for similar_id, bitscore in similar_proteins.items():
            if similar_id in annotations:
                all_go_terms |= annotations[similar_id]
                total_bitscore += bitscore
        
        if total_bitscore == 0:
            continue
        
        # 计算每个GO术语的分数
        go_scores = {}
        all_go_terms = sorted(list(all_go_terms))
        
        for go_id in all_go_terms:
            score_sum = 0.0
            for similar_id, bitscore in similar_proteins.items():
                if similar_id in annotations and go_id in annotations[similar_id]:
                    score_sum += bitscore
            
            # 归一化分数
            go_scores[go_id] = score_sum / total_bitscore
        
        diamond_predictions[query_id] = go_scores
    
    return diamond_predictions


# 完整使用示例
def example_usage():
    """完整使用示例"""
    
    # 1. 创建DIAMOND数据库（只需运行一次）
    create_diamond_database(
        fasta_file="train_sequences.fasta",
        db_name="train_db"
    )
    
    # 2. 运行DIAMOND搜索
    run_diamond_search(
        query_file="test_sequences.fasta",
        db_file="train_db.dmnd",
        output_file="diamond_results.txt"
    )
    
    # 3. 加载训练集注释
    annotations = load_annotations("train_annotations.pkl")
    
    # 4. 解析DIAMOND结果
    mapping = parse_diamond_results("diamond_results.txt")
    
    # 5. 转换为GO预测
    diamond_predictions = diamond_to_go_predictions(mapping, annotations)
    
    # 6. 使用预测结果
    for protein_id, go_scores in diamond_predictions.items():
        print(f"\n{protein_id}:")
        for go_id, score in sorted(go_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {go_id}: {score:.4f}")


if __name__ == "__main__":
    example_usage()
```

---

## 与深度学习模型结合

### 方法1: 加权组合（推荐）

将DIAMOND预测和深度学习模型预测按权重组合：

```python
def combine_predictions(diamond_preds, deep_preds, alpha=0.5):
    """
    组合DIAMOND和深度学习模型的预测
    
    Args:
        diamond_preds: DIAMOND预测 {protein_id: {go_id: score}}
        deep_preds: 深度学习预测 {protein_id: {go_id: score}}
        alpha: DIAMOND权重（0-1），深度学习权重为(1-alpha)
    
    Returns:
        dict: 组合后的预测 {protein_id: {go_id: score}}
    """
    combined = {}
    
    for protein_id in set(list(diamond_preds.keys()) + list(deep_preds.keys())):
        combined[protein_id] = {}
        
        # DIAMOND预测（权重alpha）
        if protein_id in diamond_preds:
            for go_id, score in diamond_preds[protein_id].items():
                combined[protein_id][go_id] = alpha * score
        
        # 深度学习预测（权重1-alpha）
        if protein_id in deep_preds:
            for go_id, score in deep_preds[protein_id].items():
                if go_id in combined[protein_id]:
                    combined[protein_id][go_id] += (1 - alpha) * score
                else:
                    combined[protein_id][go_id] = (1 - alpha) * score
    
    return combined
```

### 方法2: 按GO命名空间使用不同权重

不同GO本体（MF/BP/CC）可以使用不同的权重：

```python
from deepgoplus.utils import NAMESPACES  # 或自己实现

def combine_by_namespace(diamond_preds, deep_preds, go_ontology, alphas):
    """
    按GO命名空间使用不同权重组合预测
    
    Args:
        diamond_preds: DIAMOND预测
        deep_preds: 深度学习预测
        go_ontology: GO本体对象
        alphas: {namespace: alpha_value}
                例如: {'molecular_function': 0.63, 'biological_process': 0.68, 'cellular_component': 0.48}
    
    Returns:
        dict: 组合后的预测
    """
    combined = {}
    
    for protein_id in set(list(diamond_preds.keys()) + list(deep_preds.keys())):
        combined[protein_id] = {}
        
        # 处理DIAMOND预测
        if protein_id in diamond_preds:
            for go_id, score in diamond_preds[protein_id].items():
                namespace = go_ontology.get_namespace(go_id)
                if namespace in alphas:
                    alpha = alphas[namespace]
                    combined[protein_id][go_id] = alpha * score
        
        # 处理深度学习预测
        if protein_id in deep_preds:
            for go_id, score in deep_preds[protein_id].items():
                namespace = go_ontology.get_namespace(go_id)
                if namespace in alphas:
                    alpha = alphas[namespace]
                    if go_id in combined[protein_id]:
                        combined[protein_id][go_id] += (1 - alpha) * score
                    else:
                        combined[protein_id][go_id] = (1 - alpha) * score
    
    return combined
```

---

## 完整示例

### 示例：集成到现有模型

```python
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict

class DiamondPredictor:
    """DIAMOND预测器类"""
    
    def __init__(self, db_file, annotations_file):
        """
        初始化DIAMOND预测器
        
        Args:
            db_file: DIAMOND数据库文件（.dmnd）
            annotations_file: 训练集注释文件（PKL格式）
        """
        self.db_file = db_file
        self.annotations = self._load_annotations(annotations_file)
    
    def _load_annotations(self, annotations_file):
        """加载注释"""
        df = pd.read_pickle(annotations_file)
        annotations = {}
        for row in df.itertuples():
            annotations[row.proteins] = set(row.prop_annotations)
        return annotations
    
    def predict(self, query_file, output_file=None, temp_dir="/tmp"):
        """
        对查询序列进行预测
        
        Args:
            query_file: 查询序列FASTA文件
            output_file: DIAMOND输出文件（可选，自动生成）
            temp_dir: 临时文件目录
        
        Returns:
            dict: {protein_id: {go_id: score}}
        """
        if output_file is None:
            output_file = query_file.replace('.fasta', '_diamond.txt')
        
        # 运行DIAMOND搜索
        self._run_search(query_file, output_file, temp_dir)
        
        # 解析结果
        mapping = self._parse_results(output_file)
        
        # 转换为GO预测
        predictions = self._to_go_predictions(mapping)
        
        return predictions
    
    def _run_search(self, query_file, output_file, temp_dir):
        """运行DIAMOND搜索"""
        cmd = [
            "diamond", "blastp",
            "-d", self.db_file,
            "--more-sensitive",
            "-t", temp_dir,
            "-q", query_file,
            "--outfmt", "6", "qseqid", "sseqid", "bitscore",
            "-o", output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"DIAMOND search failed: {result.stderr}")
    
    def _parse_results(self, diamond_file):
        """解析DIAMOND结果"""
        mapping = {}
        with open(diamond_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    similar_id = parts[1]
                    bitscore = float(parts[2])
                    
                    if query_id not in mapping:
                        mapping[query_id] = {}
                    mapping[query_id][similar_id] = bitscore
        return mapping
    
    def _to_go_predictions(self, mapping):
        """转换为GO预测"""
        predictions = {}
        
        for query_id, similar_proteins in mapping.items():
            all_go_terms = set()
            total_bitscore = 0.0
            
            for similar_id, bitscore in similar_proteins.items():
                if similar_id in self.annotations:
                    all_go_terms |= self.annotations[similar_id]
                    total_bitscore += bitscore
            
            if total_bitscore == 0:
                continue
            
            go_scores = {}
            for go_id in all_go_terms:
                score_sum = 0.0
                for similar_id, bitscore in similar_proteins.items():
                    if similar_id in self.annotations and go_id in self.annotations[similar_id]:
                        score_sum += bitscore
                go_scores[go_id] = score_sum / total_bitscore
            
            predictions[query_id] = go_scores
        
        return predictions


# 使用示例
def main():
    # 1. 创建DIAMOND数据库（只需一次）
    subprocess.run([
        "diamond", "makedb",
        "--in", "train_sequences.fasta",
        "-d", "train_db"
    ])
    
    # 2. 初始化DIAMOND预测器
    diamond = DiamondPredictor(
        db_file="train_db.dmnd",
        annotations_file="train_annotations.pkl"
    )
    
    # 3. 进行预测
    diamond_preds = diamond.predict("test_sequences.fasta")
    
    # 4. 与深度学习模型结合
    # 假设你已经有了深度学习模型的预测 deep_preds
    # combined = combine_predictions(diamond_preds, deep_preds, alpha=0.5)
    
    # 5. 保存结果
    with open("diamond_predictions.tsv", "w") as f:
        f.write("protein_id\tgo_id\tscore\n")
        for protein_id, go_scores in diamond_preds.items():
            for go_id, score in go_scores.items():
                f.write(f"{protein_id}\t{go_id}\t{score}\n")


if __name__ == "__main__":
    main()
```

---

## 参数调优

### DIAMOND搜索参数

1. **敏感度模式**：
   ```bash
   # 默认模式（快速）
   diamond blastp -d db -q query.fasta
   
   # 更敏感模式（推荐，更准确）
   diamond blastp -d db -q query.fasta --more-sensitive
   
   # 超敏感模式（最准确，但较慢）
   diamond blastp -d db -q query.fasta --ultra-sensitive
   ```

2. **E值阈值**：
   ```bash
   # 只返回E值 < 0.001的结果
   diamond blastp -d db -q query.fasta -e 0.001
   ```

3. **最大目标序列数**：
   ```bash
   # 每个查询序列最多返回1000个相似序列
   diamond blastp -d db -q query.fasta --max-target-seqs 1000
   ```

### 组合权重调优

根据验证集性能调整alpha值：

```python
# 测试不同的alpha值
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    combined = combine_predictions(diamond_preds, deep_preds, alpha=alpha)
    # 评估性能
    metrics = evaluate(combined, true_labels)
    print(f"Alpha={alpha}: Fmax={metrics['fmax']:.4f}")
```

**经验值**（来自DeepGOPlus）：
- MF (分子功能): alpha ≈ 0.55-0.63
- BP (生物过程): alpha ≈ 0.59-0.68
- CC (细胞组分): alpha ≈ 0.46-0.48

---

## 注意事项

1. **内存使用**：DIAMOND数据库可能很大，确保有足够内存
2. **临时文件**：使用`-t`参数指定临时目录，避免磁盘空间不足
3. **并行处理**：DIAMOND自动使用多线程，可通过`--threads`参数控制
4. **结果过滤**：可以只保留bitscore较高的结果，减少计算量
5. **GO术语传播**：考虑使用GO本体结构传播预测分数（参考DeepGOPlus实现）

---

## 参考

- DIAMOND官方文档: https://github.com/bbuchfink/diamond
- DeepGOPlus实现: `deepgoplus/main.py`
- GO本体: http://geneontology.org/

---

## 快速开始检查清单

- [ ] 安装DIAMOND
- [ ] 准备训练集FASTA文件
- [ ] 准备训练集注释文件（PKL格式，包含proteins和annotations列）
- [ ] 创建DIAMOND数据库：`diamond makedb --in train.fasta -d train_db`
- [ ] 运行DIAMOND搜索：`diamond blastp -d train_db -q test.fasta --outfmt 6 qseqid sseqid bitscore -o results.txt`
- [ ] 实现解析和转换代码
- [ ] 与深度学习模型结合
- [ ] 调优alpha权重参数

