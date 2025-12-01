#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断DeepGO模型问题

用法:
python src/evaluation/diagnose_model.py \
    --deep-file galaxy/results/mf_test_preds_galaxy_deepgoplus_run1_deep_only.tsv \
    --true-file galaxy/MF_test_data.pkl \
    --terms-file galaxy/terms_mf.pkl \
    --go-file galaxy/go.obo \
    --ont mf \
    --out-file galaxy/results/diagnosis_report.txt
"""

import click as ck
import pandas as pd
import numpy as np
import sys
import os
from collections import defaultdict, Counter
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from evaluate_predictions import parse_prediction_file, load_true_labels
from deepgoplus.utils import Ontology, NAMESPACES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@ck.command()
@ck.option('--deep-file', '-df', required=True, help='DeepGO预测文件')
@ck.option('--true-file', '-tf', required=True, help='真实标签文件')
@ck.option('--terms-file', '-terms', required=True, help='训练时的术语文件')
@ck.option('--go-file', '-gf', default='go.obo', help='GO本体文件')
@ck.option('--ont', '-o', default=None, type=ck.Choice(['mf', 'cc', 'bp', 'pf']), help='本体类型')
@ck.option('--out-file', '-of', required=True, help='输出诊断报告')
def main(deep_file, true_file, terms_file, go_file, ont, out_file):
    """诊断DeepGO模型问题"""
    
    # Load data
    logging.info("Loading data...")
    go = Ontology(go_file, with_rels=True)
    deep_preds = parse_prediction_file(deep_file)
    true_labels = load_true_labels(true_file, ont, go)
    
    # Load terms used in training
    terms_df = pd.read_pickle(terms_file)
    train_terms = set(terms_df['terms'].values.flatten())
    
    # Filter train terms by ontology if specified (仅对 mf/cc/bp 使用 GO 本体过滤)
    if ont and ont != 'pf':
        namespace = NAMESPACES.get(ont)
        filtered_train_terms = set()
        for term in train_terms:
            try:
                if go.has_term(term) and go.get_namespace(term) == namespace:
                    filtered_train_terms.add(term)
            except:
                pass
        train_terms = filtered_train_terms
    
    # Collect all predicted GO terms
    all_pred_terms = set()
    for prot_id, pred_dict in deep_preds.items():
        all_pred_terms.update(pred_dict.keys())
    
    # Collect all true GO terms
    all_true_terms = set()
    for prot_id, true_set in true_labels.items():
        all_true_terms.update(true_set)
    
    # Analyze score distribution
    all_scores = []
    correct_scores = []
    incorrect_scores = []
    
    for prot_id in deep_preds:
        if prot_id not in true_labels:
            continue
        true_set = true_labels[prot_id]
        for go_id, score in deep_preds[prot_id].items():
            all_scores.append(score)
            if go_id in true_set:
                correct_scores.append(score)
            else:
                incorrect_scores.append(score)
    
    # Analyze term overlap
    pred_terms_in_train = all_pred_terms & train_terms
    pred_terms_not_in_train = all_pred_terms - train_terms
    true_terms_in_train = all_true_terms & train_terms
    true_terms_not_in_train = all_true_terms - train_terms
    
    # Analyze per-protein statistics
    proteins_with_predictions = len(deep_preds)
    proteins_with_true_labels = len(true_labels)
    proteins_with_both = len(set(deep_preds.keys()) & set(true_labels.keys()))
    
    # Count predictions per protein
    pred_counts = [len(pred_dict) for pred_dict in deep_preds.values()]
    
    # Analyze coverage
    proteins_with_correct_preds = 0
    total_correct_preds = 0
    total_preds = 0
    
    for prot_id in deep_preds:
        if prot_id not in true_labels:
            continue
        true_set = true_labels[prot_id]
        pred_dict = deep_preds[prot_id]
        total_preds += len(pred_dict)
        
        correct = sum(1 for go_id in pred_dict.keys() if go_id in true_set)
        total_correct_preds += correct
        if correct > 0:
            proteins_with_correct_preds += 1
    
    # Write report
    with open(out_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DeepGO 模型诊断报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 基本统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"有预测的蛋白质数量: {proteins_with_predictions}\n")
        f.write(f"有真实标签的蛋白质数量: {proteins_with_true_labels}\n")
        f.write(f"两者都有的蛋白质数量: {proteins_with_both}\n")
        f.write(f"总预测数: {total_preds}\n")
        f.write(f"总正确预测数: {total_correct_preds}\n")
        f.write(f"有正确预测的蛋白质数量: {proteins_with_correct_preds}\n")
        f.write(f"平均每个蛋白质的预测数: {np.mean(pred_counts):.2f}\n")
        f.write(f"中位数每个蛋白质的预测数: {np.median(pred_counts):.2f}\n")
        f.write("\n")
        
        f.write("2. GO术语覆盖分析\n")
        f.write("-" * 80 + "\n")
        f.write(f"预测的GO术语总数: {len(all_pred_terms)}\n")
        f.write(f"真实标签的GO术语总数: {len(all_true_terms)}\n")
        f.write(f"训练时的GO术语总数: {len(train_terms)}\n")

        # 避免在没有任何预测/真实术语时出现除零错误
        if len(all_pred_terms) > 0:
            pred_in_train_pct = len(pred_terms_in_train) / len(all_pred_terms) * 100.0
            pred_not_in_train_pct = len(pred_terms_not_in_train) / len(all_pred_terms) * 100.0
        else:
            pred_in_train_pct = 0.0
            pred_not_in_train_pct = 0.0

        if len(all_true_terms) > 0:
            true_in_train_pct = len(true_terms_in_train) / len(all_true_terms) * 100.0
            true_not_in_train_pct = len(true_terms_not_in_train) / len(all_true_terms) * 100.0
        else:
            true_in_train_pct = 0.0
            true_not_in_train_pct = 0.0

        f.write(f"\n预测术语中在训练集中的: {len(pred_terms_in_train)} ({pred_in_train_pct:.1f}%)\n")
        f.write(f"预测术语中不在训练集的: {len(pred_terms_not_in_train)} ({pred_not_in_train_pct:.1f}%)\n")
        f.write(f"真实标签中在训练集的: {len(true_terms_in_train)} ({true_in_train_pct:.1f}%)\n")
        f.write(f"真实标签中不在训练集的: {len(true_terms_not_in_train)} ({true_not_in_train_pct:.1f}%)\n")
        f.write("\n")
        
        f.write("3. 分数分布分析\n")
        f.write("-" * 80 + "\n")
        if all_scores:
            f.write(f"所有预测分数的统计:\n")
            f.write(f"  最小值: {np.min(all_scores):.6f}\n")
            f.write(f"  最大值: {np.max(all_scores):.6f}\n")
            f.write(f"  平均值: {np.mean(all_scores):.6f}\n")
            f.write(f"  中位数: {np.median(all_scores):.6f}\n")
            f.write(f"  标准差: {np.std(all_scores):.6f}\n")
            f.write(f"  25%分位数: {np.percentile(all_scores, 25):.6f}\n")
            f.write(f"  75%分位数: {np.percentile(all_scores, 75):.6f}\n")
            f.write(f"  95%分位数: {np.percentile(all_scores, 95):.6f}\n")
            f.write(f"  99%分位数: {np.percentile(all_scores, 99):.6f}\n")
            f.write("\n")
        
        if correct_scores:
            f.write(f"正确预测的分数统计:\n")
            f.write(f"  数量: {len(correct_scores)}\n")
            f.write(f"  平均值: {np.mean(correct_scores):.6f}\n")
            f.write(f"  中位数: {np.median(correct_scores):.6f}\n")
            f.write(f"  最小值: {np.min(correct_scores):.6f}\n")
            f.write(f"  最大值: {np.max(correct_scores):.6f}\n")
            f.write("\n")
        
        if incorrect_scores:
            f.write(f"错误预测的分数统计:\n")
            f.write(f"  数量: {len(incorrect_scores)}\n")
            f.write(f"  平均值: {np.mean(incorrect_scores):.6f}\n")
            f.write(f"  中位数: {np.median(incorrect_scores):.6f}\n")
            f.write(f"  最小值: {np.min(incorrect_scores):.6f}\n")
            f.write(f"  最大值: {np.max(incorrect_scores):.6f}\n")
            f.write("\n")
        
        # Score distribution by bins
        f.write("4. 分数区间分布\n")
        f.write("-" * 80 + "\n")
        bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = sum(1 for s in all_scores if bins[i] <= s < bins[i+1])
            correct_count = sum(1 for s in correct_scores if bins[i] <= s < bins[i+1])
            incorrect_count = sum(1 for s in incorrect_scores if bins[i] <= s < bins[i+1])
            precision = correct_count / count if count > 0 else 0
            f.write(f"[{bins[i]:.2f}, {bins[i+1]:.2f}): {count} 预测, {correct_count} 正确, {incorrect_count} 错误, 精确率: {precision:.4f}\n")
        f.write("\n")
        
        # Top predictions analysis
        f.write("5. 最常见预测的GO术语（前20）\n")
        f.write("-" * 80 + "\n")
        term_counts = Counter()
        for pred_dict in deep_preds.values():
            term_counts.update(pred_dict.keys())
        for term, count in term_counts.most_common(20):
            in_train = "✓" if term in train_terms else "✗"
            in_true = "✓" if term in all_true_terms else "✗"
            f.write(f"{term}: {count} 次预测 {in_train}训练 {in_true}真实\n")
        f.write("\n")
        
        # Diagnosis
        f.write("6. 问题诊断\n")
        f.write("-" * 80 + "\n")
        
        if len(pred_terms_not_in_train) > len(pred_terms_in_train) * 0.1:
            f.write("⚠️  警告: 大量预测的GO术语不在训练集中！\n")
            f.write(f"   这可能表明模型预测了未训练过的术语。\n\n")
        
        if np.mean(all_scores) < 0.1:
            f.write("⚠️  警告: 预测分数普遍很低！\n")
            f.write(f"   平均分数只有 {np.mean(all_scores):.6f}，模型可能没有学到有用的特征。\n\n")
        
        if len(correct_scores) == 0:
            f.write("❌ 严重问题: 没有任何正确预测！\n")
            f.write("   模型完全无法预测正确的GO术语。\n\n")
        elif len(correct_scores) / len(all_scores) < 0.01:
            f.write("❌ 严重问题: 正确预测比例极低！\n")
            f.write(f"   只有 {len(correct_scores)/len(all_scores)*100:.2f}% 的预测是正确的。\n\n")
        
        if np.mean(correct_scores) < np.mean(incorrect_scores) if incorrect_scores else False:
            f.write("⚠️  警告: 正确预测的分数反而低于错误预测！\n")
            f.write("   这表明模型无法区分正确和错误的预测。\n\n")
        
        if proteins_with_correct_preds / proteins_with_both < 0.1:
            f.write("⚠️  警告: 只有很少的蛋白质有正确预测！\n")
            f.write(f"   只有 {proteins_with_correct_preds/proteins_with_both*100:.1f}% 的蛋白质有至少一个正确预测。\n\n")
        
        f.write("7. 建议\n")
        f.write("-" * 80 + "\n")
        f.write("1. 检查模型训练是否充分（epochs是否足够）\n")
        f.write("2. 检查训练数据质量（标注是否准确）\n")
        f.write("3. 检查模型架构和超参数\n")
        f.write("4. 检查预测时的阈值设置（当前可能过滤掉了太多预测）\n")
        f.write("5. 检查terms.pkl文件是否与训练时使用的术语一致\n")
        f.write("6. 考虑增加训练数据或使用预训练模型\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    logging.info(f"Diagnosis report saved to {out_file}")


if __name__ == '__main__':
    main()

