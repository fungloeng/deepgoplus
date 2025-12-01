#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估预测结果脚本
计算 F1, Precision, Recall, Fmax, AUPR 等指标

用法:
python src/evaluation/evaluate_predictions.py \
    --pred-file galaxy/mf_test_preds_galaxy_deepgoplus_run1.tsv \
    --true-file galaxy/MF_test_data.pkl \
    --out-file galaxy/mf_evaluation_results.txt \
    --go-file galaxy/go.obo \
    --ont mf
"""

import click as ck
import numpy as np
import pandas as pd
import sys
import os
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import logging
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Add src directory to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from deepgoplus.utils import Ontology, NAMESPACES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_prediction_file(pred_file):
    """解析预测结果文件，格式：protein_id\tgo_id\tscore (长格式，每行一个预测)"""
    predictions = defaultdict(dict)  # {prot_id: {go_id: score}}
    
    with open(pred_file, 'r') as f:
        # Skip header if present
        first_line = f.readline().strip()
        if first_line.lower().startswith('protein_id') or first_line.lower().startswith('protein'):
            # Header line, skip it
            pass
        else:
            # No header, process first line
            parts = first_line.split('\t')
            if len(parts) >= 3:
                prot_id = parts[0].strip()
                go_id = parts[1].strip()
                try:
                    score = float(parts[2].strip())
                    predictions[prot_id][go_id] = max(predictions[prot_id].get(go_id, 0), score)
                except (ValueError, IndexError):
                    logging.warning(f"Invalid line format: {first_line}")
        
        # Process remaining lines
        for line in tqdm(f, desc="Parsing predictions"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            prot_id = parts[0].strip()
            go_id = parts[1].strip()
            try:
                score = float(parts[2].strip())
                # If same protein and GO term appears multiple times, take max score
                predictions[prot_id][go_id] = max(predictions[prot_id].get(go_id, 0), score)
            except ValueError:
                logging.warning(f"Invalid score format in line: {line}")
    
    logging.info(f"Loaded predictions for {len(predictions)} proteins")
    return predictions


def load_true_labels(true_file, ont=None, go=None):
    """从PKL文件加载真实标签"""
    logging.info(f"Loading true labels from {true_file}...")
    df = pd.read_pickle(true_file)
    
    true_labels = {}
    
    # 对于 mf/cc/bp，需要加载 GO 并根据本体过滤；对于 pf 则不依赖 GO
    use_ontology_filter = ont is not None and ont != 'pf'

    # 如果需要基于 GO 的本体过滤但还没有传入 go，则尝试从 true_file 目录加载
    if use_ontology_filter and go is None:
        go_file = os.path.join(os.path.dirname(true_file), 'go.obo')
        if os.path.exists(go_file):
            logging.info("Loading GO ontology for filtering...")
            go = Ontology(go_file, with_rels=True)
        else:
            logging.warning(f"GO file not found at {go_file}, skipping ontology filtering")
            ont = None
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Loading true labels"):
        prot_id = row.proteins
        # prop_annotations 是 set
        annots = set(row.prop_annotations)
        if use_ontology_filter and go:
            # 如果指定了 mf/cc/bp，本体，只保留该本体的标注
            namespace = NAMESPACES.get(ont)
            filtered_annots = set()
            for go_id in annots:
                try:
                    if go.has_term(go_id):
                        if go.get_namespace(go_id) == namespace:
                            filtered_annots.add(go_id)
                except:
                    pass
            annots = filtered_annots
        true_labels[prot_id] = annots
    
    logging.info(f"Loaded true labels for {len(true_labels)} proteins")
    return true_labels


def calculate_metrics(predictions, true_labels, go, ont=None):
    """计算各种评估指标"""
    # 收集所有蛋白质和GO术语
    # 重要：只评估有真实标签的蛋白质，避免虚高指标
    all_proteins = set(true_labels.keys())  # 只使用有真实标签的蛋白质
    all_go_terms = set()
    
    # 只收集有真实标签的蛋白质的预测和真实标签中的GO术语
    for prot_id in all_proteins:
        if prot_id in predictions:
            all_go_terms.update(predictions[prot_id].keys())
        if prot_id in true_labels:
            all_go_terms.update(true_labels[prot_id])
    
    # 过滤：只保留指定本体的GO术语；对于 pf 不做 GO 本体过滤
    if ont and ont != 'pf':
        namespace = NAMESPACES.get(ont)
        filtered_terms = set()
        logging.info(f"Filtering GO terms for ontology {ont}...")
        for go_id in tqdm(all_go_terms, desc="Filtering GO terms"):
            try:
                if go.has_term(go_id):
                    if go.get_namespace(go_id) == namespace:
                        filtered_terms.add(go_id)
            except:
                pass
        all_go_terms = filtered_terms
    
    all_go_terms = sorted(list(all_go_terms))
    all_proteins = sorted(list(all_proteins))
    
    logging.info(f"Evaluating {len(all_proteins)} proteins with {len(all_go_terms)} GO terms")
    
    # 构建真实标签矩阵
    logging.info("Building true labels matrix...")
    y_true = np.zeros((len(all_proteins), len(all_go_terms)), dtype=np.int32)
    go_id_to_idx = {go_id: j for j, go_id in enumerate(all_go_terms)}
    
    for i, prot_id in enumerate(tqdm(all_proteins, desc="Building true labels matrix")):
        if prot_id in true_labels:
            for go_id in true_labels[prot_id]:
                if go_id in go_id_to_idx:
                    y_true[i, go_id_to_idx[go_id]] = 1
    
    # 构建预测分数矩阵
    logging.info("Building prediction scores matrix...")
    y_pred_scores = np.zeros((len(all_proteins), len(all_go_terms)), dtype=np.float32)
    for i, prot_id in enumerate(tqdm(all_proteins, desc="Building prediction matrix")):
        if prot_id in predictions:
            for go_id, score in predictions[prot_id].items():
                if go_id in go_id_to_idx:
                    j = go_id_to_idx[go_id]
                    y_pred_scores[i, j] = max(y_pred_scores[i, j], score)  # 取最大值如果有重复
    
    # 计算Fmax（在不同阈值下）
    logging.info("Calculating Fmax across thresholds...")
    thresholds = np.arange(0.0, 1.01, 0.01)
    fmax_scores = []
    best_threshold = 0.0
    best_fmax = 0.0
    
    # 预计算实际正样本数（避免重复计算）
    actual_pos_counts = np.sum(y_true, axis=1)
    
    for threshold in tqdm(thresholds, desc="Computing Fmax"):
        y_pred = (y_pred_scores >= threshold).astype(np.int32)
        
        # 计算每个蛋白质的precision和recall（向量化计算）
        pred_pos_counts = np.sum(y_pred, axis=1)
        true_pos_counts = np.sum((y_true == 1) & (y_pred == 1), axis=1)
        
        # 计算precision和recall
        precisions = np.where(
            pred_pos_counts > 0,
            true_pos_counts / pred_pos_counts,
            np.where(actual_pos_counts == 0, 1.0, 0.0)
        )
        
        recalls = np.where(
            actual_pos_counts > 0,
            true_pos_counts / actual_pos_counts,
            0.0  # 如果没有真实标签，召回率应该是0，而不是1.0（避免虚高）
        )
        
        # 计算平均precision和recall（只对有真实标签的蛋白质）
        mask = actual_pos_counts > 0
        if np.sum(mask) > 0:
            avg_precision = np.mean(precisions[mask])
            avg_recall = np.mean(recalls[mask])
        else:
            avg_precision = 0.0
            avg_recall = 0.0
        
        # 计算F1
        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        fmax_scores.append(f1)
        
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = threshold
    
    # 计算AUPR（使用所有分数）
    logging.info("Calculating AUPR...")
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_scores.flatten()
    
    # 只计算有真实标签的样本
    mask = y_true_flat >= 0  # 所有样本
    if np.sum(y_true_flat[mask] == 1) > 0:  # 确保有正样本
        precision, recall, _ = precision_recall_curve(y_true_flat[mask], y_pred_flat[mask])
        aupr = auc(recall, precision)
    else:
        aupr = 0.0
    logging.info(f"AUPR calculated: {aupr:.4f}")
    
    # 计算最佳阈值下的详细指标
    y_pred_best = (y_pred_scores >= best_threshold).astype(np.int32)
    
    # 总体统计
    total_true_pos = np.sum((y_true == 1) & (y_pred_best == 1))
    total_pred_pos = np.sum(y_pred_best == 1)
    total_actual_pos = np.sum(y_true == 1)
    
    overall_precision = total_true_pos / total_pred_pos if total_pred_pos > 0 else 0.0
    overall_recall = total_true_pos / total_actual_pos if total_actual_pos > 0 else 0.0
    
    # 计算每个蛋白质的平均指标（向量化计算）
    logging.info("Calculating per-protein metrics...")
    pred_pos_counts = np.sum(y_pred_best, axis=1)
    true_pos_counts = np.sum((y_true == 1) & (y_pred_best == 1), axis=1)
    
    protein_precisions = np.where(
        pred_pos_counts > 0,
        true_pos_counts / pred_pos_counts,
        0.0  # 如果没有预测，精确率是0
    )
    
    protein_recalls = np.where(
        actual_pos_counts > 0,
        true_pos_counts / actual_pos_counts,
        0.0  # 如果没有真实标签，召回率应该是0，而不是1.0（避免虚高）
    )
    
    # 计算F1
    protein_f1s = np.where(
        protein_precisions + protein_recalls > 0,
        2 * protein_precisions * protein_recalls / (protein_precisions + protein_recalls),
        0.0
    )
    
    # 只对有真实标签的蛋白质计算平均值（修复：避免虚高）
    mask = actual_pos_counts > 0
    if np.sum(mask) > 0:
        avg_precision = np.mean(protein_precisions[mask])
        avg_recall = np.mean(protein_recalls[mask])
        avg_f1 = np.mean(protein_f1s[mask])
    else:
        avg_precision = 0.0
        avg_recall = 0.0
        avg_f1 = 0.0
    
    return {
        'fmax': best_fmax,
        'fmax_threshold': best_threshold,
        'aupr': aupr,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'num_proteins': len(all_proteins),
        'num_go_terms': len(all_go_terms),
        'total_predictions': total_pred_pos,
        'total_true_labels': total_actual_pos,
        'total_true_positives': total_true_pos
    }


@ck.command()
@ck.option('--pred-file', '-pf', required=True, help='预测结果TSV文件')
@ck.option('--true-file', '-tf', required=True, help='真实标签PKL文件')
@ck.option('--out-file', '-of', required=True, help='输出评估结果TXT文件')
@ck.option('--go-file', '-gf', default='go.obo', help='GO本体文件')
@ck.option('--ont', '-o', default=None, type=ck.Choice(['mf', 'cc', 'bp', 'pf']), 
           help='要评估的本体（可选，如果不指定则评估所有）')
def main(pred_file, true_file, out_file, go_file, ont):
    """评估预测结果并计算各项指标"""
    
    # 解析文件路径
    if not os.path.isabs(pred_file):
        pred_file = os.path.abspath(pred_file)
    if not os.path.isabs(true_file):
        true_file = os.path.abspath(true_file)
    if not os.path.isabs(go_file):
        # 尝试在true_file的目录中查找
        true_dir = os.path.dirname(true_file)
        go_file_path = os.path.join(true_dir, go_file)
        if os.path.exists(go_file_path):
            go_file = go_file_path
        else:
            go_file = os.path.abspath(go_file)
    
    if not os.path.exists(pred_file):
        logging.error(f"预测文件不存在: {pred_file}")
        sys.exit(1)
    if not os.path.exists(true_file):
        logging.error(f"真实标签文件不存在: {true_file}")
        sys.exit(1)
    if not os.path.exists(go_file):
        logging.error(f"GO文件不存在: {go_file}")
        sys.exit(1)
    
    logging.info(f"加载GO本体: {go_file}")
    go = Ontology(go_file, with_rels=True)
    
    logging.info(f"解析预测文件: {pred_file}")
    predictions = parse_prediction_file(pred_file)
    
    logging.info(f"加载真实标签: {true_file}")
    true_labels = load_true_labels(true_file, ont, go)
    
    logging.info("计算评估指标...")
    metrics = calculate_metrics(predictions, true_labels, go, ont)
    
    # 写入结果文件
    logging.info(f"保存评估结果到: {out_file}")
    with open(out_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DeepGOPlus 预测结果评估报告\n")
        f.write("=" * 60 + "\n\n")
        
        if ont:
            f.write(f"本体 (Ontology): {ont.upper()}\n")
        else:
            f.write(f"本体 (Ontology): 全部\n")
        
        f.write(f"\n数据集信息:\n")
        f.write(f"  蛋白质数量: {metrics['num_proteins']}\n")
        f.write(f"  GO术语数量: {metrics['num_go_terms']}\n")
        f.write(f"  总预测数: {metrics['total_predictions']}\n")
        f.write(f"  总真实标签数: {metrics['total_true_labels']}\n")
        f.write(f"  真正例数: {metrics['total_true_positives']}\n")
        
        f.write(f"\n主要指标:\n")
        f.write(f"  Fmax: {metrics['fmax']:.4f} (阈值: {metrics['fmax_threshold']:.3f})\n")
        f.write(f"  AUPR: {metrics['aupr']:.4f}\n")
        
        f.write(f"\n详细指标 (最佳阈值 {metrics['fmax_threshold']:.3f}):\n")
        f.write(f"  总体精确率 (Overall Precision): {metrics['overall_precision']:.4f}\n")
        f.write(f"  总体召回率 (Overall Recall): {metrics['overall_recall']:.4f}\n")
        f.write(f"  平均精确率 (Average Precision): {metrics['avg_precision']:.4f}\n")
        f.write(f"  平均召回率 (Average Recall): {metrics['avg_recall']:.4f}\n")
        f.write(f"  平均F1分数 (Average F1): {metrics['avg_f1']:.4f}\n")
        
        f.write(f"\n" + "=" * 60 + "\n")
    
    logging.info("评估完成！")
    logging.info(f"Fmax: {metrics['fmax']:.4f}")
    logging.info(f"AUPR: {metrics['aupr']:.4f}")


if __name__ == '__main__':
    main()

