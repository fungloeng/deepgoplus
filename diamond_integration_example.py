#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIAMOND集成示例代码
参考DeepGOPlus的实现，可以直接集成到你的模型中
"""

import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_diamond_database(fasta_file, db_name):
    """
    创建DIAMOND数据库
    
    Args:
        fasta_file: 训练集FASTA文件路径
        db_name: 数据库名称（不含.dmnd扩展名）
    
    Returns:
        str: 数据库文件路径
    """
    db_file = f"{db_name}.dmnd"
    
    if os.path.exists(db_file):
        logging.info(f"DIAMOND database already exists: {db_file}")
        return db_file
    
    logging.info(f"Creating DIAMOND database from {fasta_file}...")
    cmd = [
        "diamond", "makedb",
        "--in", fasta_file,
        "-d", db_name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Failed to create DIAMOND database: {result.stderr}")
    
    logging.info(f"DIAMOND database created: {db_file}")
    return db_file


def run_diamond_search(query_file, db_file, output_file, temp_dir="/tmp", max_target_seqs=None):
    """
    运行DIAMOND搜索
    
    Args:
        query_file: 查询序列文件（测试集FASTA）
        db_file: DIAMOND数据库文件（.dmnd）
        output_file: 输出结果文件
        temp_dir: 临时文件目录
        max_target_seqs: 每个查询序列最多返回的相似序列数（None表示不限制）
    
    Returns:
        str: 输出文件路径
    """
    logging.info(f"Running DIAMOND search: {query_file} -> {db_file}")
    
    cmd = [
        "diamond", "blastp",
        "-d", db_file,
        "--more-sensitive",  # 使用更敏感的模式
        "-t", temp_dir,
        "-q", query_file,
        "--outfmt", "6", "qseqid", "sseqid", "bitscore",
        "-o", output_file
    ]
    
    if max_target_seqs is not None:
        cmd.extend(["--max-target-seqs", str(max_target_seqs)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"DIAMOND search failed: {result.stderr}")
    
    logging.info(f"DIAMOND search completed: {output_file}")
    return output_file


def load_annotations(annotations_file):
    """
    加载训练集的GO注释
    
    Args:
        annotations_file: 注释文件（PKL格式）
                         DataFrame应包含'proteins'和'prop_annotations'列
                         'prop_annotations'可以是set或list
    
    Returns:
        dict: {protein_id: set(go_terms)}
    """
    logging.info(f"Loading annotations from {annotations_file}...")
    df = pd.read_pickle(annotations_file)
    
    annotations = {}
    for row in df.itertuples():
        protein_id = row.proteins
        # 处理不同的注释格式
        if hasattr(row, 'prop_annotations'):
            annots = row.prop_annotations
        elif hasattr(row, 'annotations'):
            annots = row.annotations
        else:
            raise ValueError("Annotations file must contain 'prop_annotations' or 'annotations' column")
        
        # 转换为set
        if isinstance(annots, set):
            annotations[protein_id] = annots
        elif isinstance(annots, list):
            annotations[protein_id] = set(annots)
        else:
            annotations[protein_id] = set([annots])
    
    logging.info(f"Loaded annotations for {len(annotations)} proteins")
    return annotations


def parse_diamond_results(diamond_file):
    """
    解析DIAMOND搜索结果
    
    Args:
        diamond_file: DIAMOND输出文件
    
    Returns:
        dict: {query_protein_id: {similar_protein_id: bitscore}}
    """
    logging.info(f"Parsing DIAMOND results from {diamond_file}...")
    
    mapping = defaultdict(dict)
    
    with open(diamond_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                query_id = parts[0]
                similar_id = parts[1]
                bitscore = float(parts[2])
                
                mapping[query_id][similar_id] = bitscore
    
    logging.info(f"Parsed results for {len(mapping)} query proteins")
    return dict(mapping)


def diamond_to_go_predictions(mapping, annotations, min_bitscore=None):
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
        min_bitscore: 最小bitscore阈值（可选，过滤低质量匹配）
    
    Returns:
        dict: {query_protein_id: {go_id: score}}
    """
    logging.info("Converting DIAMOND results to GO predictions...")
    
    diamond_predictions = {}
    
    for query_id, similar_proteins in mapping.items():
        # 收集所有相似蛋白质的GO术语
        all_go_terms = set()
        total_bitscore = 0.0
        
        for similar_id, bitscore in similar_proteins.items():
            # 可选：过滤低质量匹配
            if min_bitscore is not None and bitscore < min_bitscore:
                continue
            
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
                if min_bitscore is not None and bitscore < min_bitscore:
                    continue
                if similar_id in annotations and go_id in annotations[similar_id]:
                    score_sum += bitscore
            
            # 归一化分数
            go_scores[go_id] = score_sum / total_bitscore
        
        diamond_predictions[query_id] = go_scores
    
    logging.info(f"Generated GO predictions for {len(diamond_predictions)} proteins")
    return diamond_predictions


def combine_predictions(diamond_preds, deep_preds, alpha=0.5):
    """
    组合DIAMOND和深度学习模型的预测
    
    Args:
        diamond_preds: DIAMOND预测 {protein_id: {go_id: score}}
        deep_preds: 深度学习预测 {protein_id: {go_id: score}}
        alpha: DIAMOND权重（0-1），深度学习权重为(1-alpha)
               alpha=0.5 表示两者权重相等
               alpha=1.0 表示只用DIAMOND
               alpha=0.0 表示只用深度学习
    
    Returns:
        dict: 组合后的预测 {protein_id: {go_id: score}}
    """
    logging.info(f"Combining predictions with alpha={alpha}...")
    
    combined = {}
    all_proteins = set(list(diamond_preds.keys()) + list(deep_preds.keys()))
    
    for protein_id in all_proteins:
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
    
    logging.info(f"Combined predictions for {len(combined)} proteins")
    return combined


def save_predictions(predictions, output_file):
    """
    保存预测结果到TSV文件
    
    Args:
        predictions: 预测结果 {protein_id: {go_id: score}}
        output_file: 输出文件路径
    """
    logging.info(f"Saving predictions to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("protein_id\tgo_id\tscore\n")
        for protein_id, go_scores in predictions.items():
            for go_id, score in go_scores.items():
                f.write(f"{protein_id}\t{go_id}\t{score}\n")
    
    logging.info(f"Saved {sum(len(scores) for scores in predictions.values())} predictions")


# ============================================================================
# 完整使用示例
# ============================================================================

def example_full_workflow():
    """完整工作流程示例"""
    
    # 配置路径
    train_fasta = "train_sequences.fasta"
    train_annotations = "train_annotations.pkl"
    test_fasta = "test_sequences.fasta"
    db_name = "train_db"
    diamond_output = "diamond_results.txt"
    final_output = "combined_predictions.tsv"
    
    # 步骤1: 创建DIAMOND数据库（只需运行一次）
    db_file = create_diamond_database(train_fasta, db_name)
    
    # 步骤2: 运行DIAMOND搜索
    run_diamond_search(test_fasta, db_file, diamond_output)
    
    # 步骤3: 加载训练集注释
    annotations = load_annotations(train_annotations)
    
    # 步骤4: 解析DIAMOND结果
    mapping = parse_diamond_results(diamond_output)
    
    # 步骤5: 转换为GO预测
    diamond_preds = diamond_to_go_predictions(mapping, annotations)
    
    # 步骤6: 与深度学习模型结合（假设你已经有deep_preds）
    # deep_preds = your_deep_learning_model.predict(test_fasta)
    # combined = combine_predictions(diamond_preds, deep_preds, alpha=0.5)
    
    # 如果只用DIAMOND，直接保存
    save_predictions(diamond_preds, final_output)
    
    logging.info("Workflow completed!")


class DiamondPredictor:
    """DIAMOND预测器类（面向对象接口）"""
    
    def __init__(self, db_file, annotations_file):
        """
        初始化DIAMOND预测器
        
        Args:
            db_file: DIAMOND数据库文件（.dmnd）
            annotations_file: 训练集注释文件（PKL格式）
        """
        self.db_file = db_file
        self.annotations = load_annotations(annotations_file)
        logging.info(f"DIAMOND predictor initialized with {len(self.annotations)} annotated proteins")
    
    def predict(self, query_file, output_file=None, temp_dir="/tmp", min_bitscore=None):
        """
        对查询序列进行预测
        
        Args:
            query_file: 查询序列FASTA文件
            output_file: DIAMOND输出文件（可选，自动生成）
            temp_dir: 临时文件目录
            min_bitscore: 最小bitscore阈值
        
        Returns:
            dict: {protein_id: {go_id: score}}
        """
        if output_file is None:
            output_file = query_file.replace('.fasta', '_diamond.txt')
            if output_file == query_file:
                output_file = query_file + "_diamond.txt"
        
        # 运行DIAMOND搜索
        run_diamond_search(query_file, self.db_file, output_file, temp_dir)
        
        # 解析结果
        mapping = parse_diamond_results(output_file)
        
        # 转换为GO预测
        predictions = diamond_to_go_predictions(mapping, self.annotations, min_bitscore)
        
        return predictions


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例1: 使用函数式接口
    print("=" * 60)
    print("示例1: 函数式接口")
    print("=" * 60)
    
    # 配置（根据实际情况修改）
    train_fasta = "galaxy/MF_train_sequences.fasta"
    train_annotations = "galaxy/MF_train_data.pkl"
    test_fasta = "galaxy/MF_test_sequences.fasta"
    
    # 创建数据库
    db_file = create_diamond_database(train_fasta, "train_db")
    
    # 运行搜索
    diamond_output = "diamond_results.txt"
    run_diamond_search(test_fasta, db_file, diamond_output)
    
    # 加载注释和解析结果
    annotations = load_annotations(train_annotations)
    mapping = parse_diamond_results(diamond_output)
    
    # 转换为GO预测
    diamond_preds = diamond_to_go_predictions(mapping, annotations)
    
    # 保存结果
    save_predictions(diamond_preds, "diamond_predictions.tsv")
    
    print("\n" + "=" * 60)
    print("示例2: 面向对象接口")
    print("=" * 60)
    
    # 示例2: 使用类接口
    predictor = DiamondPredictor(db_file, train_annotations)
    predictions = predictor.predict(test_fasta)
    save_predictions(predictions, "diamond_predictions_oo.tsv")
    
    print("\n完成！")

