#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析预测结果，检查数据泄露并分别评估DIAMOND和深度学习模型

用法:
python src/evaluation/analyze_predictions.py \
    --pred-file galaxy/results/mf_test_preds_galaxy_deepgoplus_run1.tsv \
    --diamond-file galaxy/results/mf_test_preds_galaxy_deepgoplus_run1_diamond_only.tsv \
    --deep-file galaxy/results/mf_test_preds_galaxy_deepgoplus_run1_deep_only.tsv \
    --true-file galaxy/MF_test_data.pkl \
    --train-file galaxy/MF_train_data.pkl \
    --out-file galaxy/results/analysis_report.txt
"""

import click as ck
import pandas as pd
import sys
import os
from collections import defaultdict
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Add src directory to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from evaluate_predictions import parse_prediction_file, load_true_labels, calculate_metrics
from deepgoplus.utils import Ontology

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_data_leakage(test_file, train_file):
    """检查测试集和训练集是否有重叠"""
    logging.info("Checking for data leakage...")
    
    test_df = pd.read_pickle(test_file)
    train_df = pd.read_pickle(train_file)
    
    test_proteins = set(test_df['proteins'].values)
    train_proteins = set(train_df['proteins'].values)
    
    overlap = test_proteins & train_proteins
    overlap_ratio = len(overlap) / len(test_proteins) if len(test_proteins) > 0 else 0
    
    return {
        'test_count': len(test_proteins),
        'train_count': len(train_proteins),
        'overlap_count': len(overlap),
        'overlap_ratio': overlap_ratio,
        'overlap_proteins': list(overlap)[:10]  # First 10 for display
    }


def analyze_model_contribution(combined_preds, diamond_preds, deep_preds, true_labels, go, ont):
    """分析DIAMOND和深度学习模型的贡献"""
    logging.info("Analyzing model contributions...")
    
    # Count predictions from each model
    diamond_only = 0
    deep_only = 0
    both = 0
    diamond_total = 0
    deep_total = 0
    
    for prot_id in combined_preds:
        combined_gos = set(combined_preds[prot_id].keys())
        diamond_gos = set(diamond_preds.get(prot_id, {}).keys())
        deep_gos = set(deep_preds.get(prot_id, {}).keys())
        
        diamond_total += len(diamond_gos)
        deep_total += len(deep_gos)
        
        diamond_only_gos = diamond_gos - deep_gos
        deep_only_gos = deep_gos - diamond_gos
        both_gos = diamond_gos & deep_gos
        
        diamond_only += len(diamond_only_gos)
        deep_only += len(deep_only_gos)
        both += len(both_gos)
    
    # Calculate average scores
    diamond_avg_score = 0.0
    deep_avg_score = 0.0
    diamond_count = 0
    deep_count = 0
    
    for prot_id in diamond_preds:
        for go_id, score in diamond_preds[prot_id].items():
            diamond_avg_score += score
            diamond_count += 1
    
    for prot_id in deep_preds:
        for go_id, score in deep_preds[prot_id].items():
            deep_avg_score += score
            deep_count += 1
    
    diamond_avg_score = diamond_avg_score / diamond_count if diamond_count > 0 else 0
    deep_avg_score = deep_avg_score / deep_count if deep_count > 0 else 0
    
    return {
        'diamond_only_count': diamond_only,
        'deep_only_count': deep_only,
        'both_count': both,
        'diamond_total': diamond_total,
        'deep_total': deep_total,
        'diamond_avg_score': diamond_avg_score,
        'deep_avg_score': deep_avg_score,
        'diamond_ratio': diamond_total / (diamond_total + deep_total) if (diamond_total + deep_total) > 0 else 0,
        'deep_ratio': deep_total / (diamond_total + deep_total) if (diamond_total + deep_total) > 0 else 0
    }


@ck.command()
@ck.option('--pred-file', '-pf', required=True, help='组合预测结果文件')
@ck.option('--diamond-file', '-df', default=None, help='DIAMOND单独预测文件（可选）')
@ck.option('--deep-file', '-deep', default=None, help='深度学习模型单独预测文件（可选）')
@ck.option('--true-file', '-tf', required=True, help='真实标签PKL文件')
@ck.option('--train-file', '-trf', default=None, help='训练数据PKL文件（用于检查数据泄露）')
@ck.option('--go-file', '-gf', default='go.obo', help='GO本体文件')
@ck.option('--ont', '-o', default=None, type=ck.Choice(['mf', 'cc', 'bp', 'pf']), 
           help='本体类型')
@ck.option('--out-file', '-of', required=True, help='输出分析报告文件')
def main(pred_file, diamond_file, deep_file, true_file, train_file, go_file, ont, out_file):
    """分析预测结果，检查数据泄露并评估各模型"""
    
    # Resolve paths
    if not os.path.isabs(go_file):
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
    
    logging.info("Loading GO ontology...")
    go = Ontology(go_file, with_rels=True)
    
    # Load predictions
    logging.info("Loading combined predictions...")
    combined_preds = parse_prediction_file(pred_file)
    
    diamond_preds = {}
    deep_preds = {}
    
    if diamond_file and os.path.exists(diamond_file):
        logging.info("Loading DIAMOND predictions...")
        diamond_preds = parse_prediction_file(diamond_file)
    
    if deep_file and os.path.exists(deep_file):
        logging.info("Loading DeepGO predictions...")
        deep_preds = parse_prediction_file(deep_file)
    
    # Load true labels
    logging.info("Loading true labels...")
    true_labels = load_true_labels(true_file, ont, go)
    
    # Check data leakage
    leakage_info = None
    if train_file and os.path.exists(train_file):
        leakage_info = check_data_leakage(true_file, train_file)
    
    # Calculate metrics for combined predictions
    logging.info("Calculating metrics for combined predictions...")
    combined_metrics = calculate_metrics(combined_preds, true_labels, go, ont)
    
    # Calculate metrics for separate models if available
    diamond_metrics = None
    deep_metrics = None
    if diamond_preds:
        logging.info("Calculating metrics for DIAMOND predictions...")
        diamond_metrics = calculate_metrics(diamond_preds, true_labels, go, ont)
    
    if deep_preds:
        logging.info("Calculating metrics for DeepGO predictions...")
        deep_metrics = calculate_metrics(deep_preds, true_labels, go, ont)
    
    # Analyze model contribution
    contribution_info = None
    if diamond_preds and deep_preds:
        contribution_info = analyze_model_contribution(
            combined_preds, diamond_preds, deep_preds, true_labels, go, ont
        )
    
    # Write report
    logging.info(f"Writing analysis report to {out_file}...")
    with open(out_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DeepGOPlus 预测结果分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        if ont:
            f.write(f"本体 (Ontology): {ont.upper()}\n\n")
        else:
            f.write(f"本体 (Ontology): 全部\n\n")
        
        # Data leakage check
        if leakage_info:
            f.write("=" * 80 + "\n")
            f.write("数据泄露检查 (Data Leakage Check)\n")
            f.write("=" * 80 + "\n")
            f.write(f"测试集蛋白质数量: {leakage_info['test_count']}\n")
            f.write(f"训练集蛋白质数量: {leakage_info['train_count']}\n")
            f.write(f"重叠蛋白质数量: {leakage_info['overlap_count']}\n")
            f.write(f"重叠比例: {leakage_info['overlap_ratio']:.2%}\n")
            if leakage_info['overlap_count'] > 0:
                f.write(f"\n⚠️  警告: 发现 {leakage_info['overlap_count']} 个重叠蛋白质！\n")
                f.write(f"这可能导致评估结果虚高。\n")
                if leakage_info['overlap_proteins']:
                    f.write(f"示例重叠蛋白质: {', '.join(leakage_info['overlap_proteins'])}\n")
            else:
                f.write(f"\n✓ 未发现数据泄露\n")
            f.write("\n")
        
        # Model contribution analysis
        if contribution_info:
            f.write("=" * 80 + "\n")
            f.write("模型贡献分析 (Model Contribution Analysis)\n")
            f.write("=" * 80 + "\n")
            f.write(f"DIAMOND 单独预测数: {contribution_info['diamond_only_count']}\n")
            f.write(f"DeepGO 单独预测数: {contribution_info['deep_only_count']}\n")
            f.write(f"两者都预测的数量: {contribution_info['both_count']}\n")
            f.write(f"\nDIAMOND 总预测数: {contribution_info['diamond_total']}\n")
            f.write(f"DeepGO 总预测数: {contribution_info['deep_total']}\n")
            f.write(f"DIAMOND 预测比例: {contribution_info['diamond_ratio']:.2%}\n")
            f.write(f"DeepGO 预测比例: {contribution_info['deep_ratio']:.2%}\n")
            f.write(f"\nDIAMOND 平均分数: {contribution_info['diamond_avg_score']:.4f}\n")
            f.write(f"DeepGO 平均分数: {contribution_info['deep_avg_score']:.4f}\n")
            f.write("\n")
        
        # Combined metrics
        f.write("=" * 80 + "\n")
        f.write("组合模型评估指标 (Combined Model Metrics)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Fmax: {combined_metrics['fmax']:.4f} (阈值: {combined_metrics['fmax_threshold']:.3f})\n")
        f.write(f"AUPR: {combined_metrics['aupr']:.4f}\n")
        f.write(f"总体精确率: {combined_metrics['overall_precision']:.4f}\n")
        f.write(f"总体召回率: {combined_metrics['overall_recall']:.4f}\n")
        f.write(f"平均精确率: {combined_metrics['avg_precision']:.4f}\n")
        f.write(f"平均召回率: {combined_metrics['avg_recall']:.4f}\n")
        f.write(f"平均F1: {combined_metrics['avg_f1']:.4f}\n")
        f.write("\n")
        
        # DIAMOND metrics
        if diamond_metrics:
            f.write("=" * 80 + "\n")
            f.write("DIAMOND 模型评估指标 (DIAMOND Model Metrics)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fmax: {diamond_metrics['fmax']:.4f} (阈值: {diamond_metrics['fmax_threshold']:.3f})\n")
            f.write(f"AUPR: {diamond_metrics['aupr']:.4f}\n")
            f.write(f"总体精确率: {diamond_metrics['overall_precision']:.4f}\n")
            f.write(f"总体召回率: {diamond_metrics['overall_recall']:.4f}\n")
            f.write(f"平均精确率: {diamond_metrics['avg_precision']:.4f}\n")
            f.write(f"平均召回率: {diamond_metrics['avg_recall']:.4f}\n")
            f.write(f"平均F1: {diamond_metrics['avg_f1']:.4f}\n")
            f.write("\n")
        
        # DeepGO metrics
        if deep_metrics:
            f.write("=" * 80 + "\n")
            f.write("DeepGO 模型评估指标 (DeepGO Model Metrics)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Fmax: {deep_metrics['fmax']:.4f} (阈值: {deep_metrics['fmax_threshold']:.3f})\n")
            f.write(f"AUPR: {deep_metrics['aupr']:.4f}\n")
            f.write(f"总体精确率: {deep_metrics['overall_precision']:.4f}\n")
            f.write(f"总体召回率: {deep_metrics['overall_recall']:.4f}\n")
            f.write(f"平均精确率: {deep_metrics['avg_precision']:.4f}\n")
            f.write(f"平均召回率: {deep_metrics['avg_recall']:.4f}\n")
            f.write(f"平均F1: {deep_metrics['avg_f1']:.4f}\n")
            f.write("\n")
        
        # Summary and recommendations
        f.write("=" * 80 + "\n")
        f.write("分析总结 (Summary)\n")
        f.write("=" * 80 + "\n")
        
        if leakage_info and leakage_info['overlap_ratio'] > 0.01:
            f.write("⚠️  发现数据泄露问题！测试集和训练集有重叠。\n")
            f.write("   建议：重新划分数据集，确保测试集和训练集完全分离。\n\n")
        
        if diamond_metrics and deep_metrics:
            if diamond_metrics['fmax'] > deep_metrics['fmax'] * 1.1:
                f.write("📊 DIAMOND 模型表现明显优于 DeepGO 模型。\n")
                f.write("   这可能表明：\n")
                f.write("   - 测试集与训练集序列相似度较高\n")
                f.write("   - 深度学习模型需要更多训练或调优\n\n")
            elif deep_metrics['fmax'] > diamond_metrics['fmax'] * 1.1:
                f.write("📊 DeepGO 模型表现明显优于 DIAMOND 模型。\n")
                f.write("   这表明深度学习模型学到了序列相似性之外的特征。\n\n")
            else:
                f.write("📊 两个模型表现相近，组合使用可以互补。\n\n")
        
        f.write("=" * 80 + "\n")
    
    logging.info("Analysis complete!")


if __name__ == '__main__':
    main()

