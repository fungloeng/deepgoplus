#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的DeepGOPlus工作流程脚本
支持参数化配置，方便运行不同本体、数据集和运行次数

用法:
python run_complete_pipeline.py \
    --ont mf \
    --dataset galaxy \
    --run 1 \
    --data-root galaxy/ \
    --epochs 12 \
    --batch-size 32 \
    --device gpu:0
"""

import click as ck
import subprocess
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@ck.command()
@ck.option('--ont', '-o', required=True, type=ck.Choice(['mf', 'cc', 'bp', 'pf']), 
           help='本体类型: mf (分子功能), cc (细胞组分), bp (生物过程), pf (蛋白质家族)')
@ck.option('--dataset', '-d', required=True, type=ck.Choice(['galaxy', 'cafa']),
           help='数据集类型: galaxy 或 cafa')
@ck.option('--run', '-r', default=1, type=int, help='运行次数编号 (默认: 1)')
@ck.option('--data-root', '-dr', default='galaxy/', help='数据根目录 (默认: galaxy/)')
@ck.option('--go-file', '-gf', default='go.obo', help='GO本体文件名 (默认: go.obo)')
@ck.option('--epochs', '-e', default=12, type=int, help='训练轮数 (默认: 12)')
@ck.option('--batch-size', '-bs', default=32, type=int, help='批次大小 (默认: 32)')
@ck.option('--learning-rate', '-lr', default=0.001, type=float, help='学习率 (默认: 0.001)')
@ck.option('--device', '-dev', default='gpu:0', help='计算设备 (默认: gpu:0)')
@ck.option('--threshold', '-t', default=0.0, type=float, help='预测阈值 (默认: 0.0，保存所有预测)')
@ck.option('--alpha', '-a', default=0.1, type=float, help='DIAMOND和DeepGO权重 (默认: 0.5)')
@ck.option('--skip-train', '-st', is_flag=True, help='跳过训练步骤（使用已有模型）')
@ck.option('--skip-pred', '-sp', is_flag=True, help='跳过预测步骤')
@ck.option('--skip-eval', '-se', is_flag=True, help='跳过评估步骤')
@ck.option('--save-separate', '-ss', is_flag=True, help='保存DIAMOND和DeepGO的单独预测结果')
def main(ont, dataset, run, data_root, go_file, epochs, batch_size, learning_rate, 
         device, threshold, alpha, skip_train, skip_pred, skip_eval, save_separate):
    """运行完整的DeepGOPlus工作流程"""
    
    # 设置文件路径
    ont_upper = ont.upper()
    # Galaxy数据集使用 MF/CC/BP 前缀，CAFA数据集可能使用不同前缀
    if dataset == 'galaxy':
        prefix = ont_upper  # MF, CC, BP, PF
    else:  # cafa
        prefix = ont_upper  # 可以根据实际情况调整
    
    # 文件路径配置
    paths = {
        'go_file': os.path.join(data_root, go_file),
        'train_tsv': os.path.join(data_root, f'{prefix}_train_data.tsv'),
        'train_fasta': os.path.join(data_root, f'{prefix}_train_sequences.fasta'),
        'train_pkl': os.path.join(data_root, f'{prefix}_train_data.pkl'),
        'val_tsv': os.path.join(data_root, f'{prefix}_validation_data.tsv'),
        'val_fasta': os.path.join(data_root, f'{prefix}_validation_sequences.fasta'),
        'val_pkl': os.path.join(data_root, f'{prefix}_validation_data.pkl'),
        'test_tsv': os.path.join(data_root, f'{prefix}_test_data.tsv'),
        'test_fasta': os.path.join(data_root, f'{prefix}_test_sequences.fasta'),
        'test_pkl': os.path.join(data_root, f'{prefix}_test_data.pkl'),
        'terms_pkl': os.path.join(data_root, f'terms_{ont}.pkl'),
        'model_h5': os.path.join(data_root, f'model_{ont}.h5'),
        'diamond_db': os.path.join(data_root, f'{prefix}_train_data'),
        'diamond_db_file': os.path.join(data_root, f'{prefix}_train_data.dmnd'),
        'results_dir': os.path.join(data_root, 'results'),
        'pred_tsv': os.path.join(data_root, 'results', f'{ont}_test_preds_{dataset}_deepgoplus_run{run}.tsv'),
        'eval_txt': os.path.join(data_root, 'results', f'{ont}_evaluation_results_run{run}.txt'),
        'analysis_txt': os.path.join(data_root, 'results', f'{ont}_analysis_report_run{run}.txt'),
        'diagnosis_txt': os.path.join(data_root, 'results', f'{ont}_diagnosis_report_run{run}.txt'),
    }
    
    # 创建结果目录
    os.makedirs(paths['results_dir'], exist_ok=True)
    
    logging.info("=" * 80)
    logging.info(f"DeepGOPlus 完整工作流程")
    logging.info(f"本体: {ont.upper()}, 数据集: {dataset}, 运行: {run}")
    logging.info("=" * 80)
    
    # Step 1: 准备训练数据
    logging.info("\n[Step 1/10] 准备训练数据...")
    if not os.path.exists(paths['train_pkl']):
        cmd = [
            'python', 'src/data/prepare_data.py',
            '--go-file', paths['go_file'],
            '--data-file', paths['train_tsv'],
            '--sequences-file', paths['train_fasta'],
            '--out-file', paths['train_pkl'],
            '--ont', ont
        ]
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("准备训练数据失败！")
            sys.exit(1)
    else:
        logging.info(f"训练数据已存在: {paths['train_pkl']}")
    
    # Step 2: 准备验证数据
    logging.info("\n[Step 2/10] 准备验证数据...")
    if not os.path.exists(paths['val_pkl']):
        cmd = [
            'python', 'src/data/prepare_data.py',
            '--go-file', paths['go_file'],
            '--data-file', paths['val_tsv'],
            '--sequences-file', paths['val_fasta'],
            '--out-file', paths['val_pkl'],
            '--ont', ont
        ]
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("准备验证数据失败！")
            sys.exit(1)
    else:
        logging.info(f"验证数据已存在: {paths['val_pkl']}")
    
    # Step 3: 准备测试数据
    logging.info("\n[Step 3/10] 准备测试数据...")
    if not os.path.exists(paths['test_pkl']):
        cmd = [
            'python', 'src/data/prepare_data.py',
            '--go-file', paths['go_file'],
            '--data-file', paths['test_tsv'],
            '--sequences-file', paths['test_fasta'],
            '--out-file', paths['test_pkl'],
            '--ont', ont
        ]
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("准备测试数据失败！")
            sys.exit(1)
    else:
        logging.info(f"测试数据已存在: {paths['test_pkl']}")
    
    # Step 4: 生成术语文件
    logging.info("\n[Step 4/10] 生成术语文件...")
    if not os.path.exists(paths['terms_pkl']):
        cmd = [
            'python', 'src/data/get_terms.py',
            '--train-data-file', paths['train_pkl'],
            '--out-file', paths['terms_pkl']
        ]
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("生成术语文件失败！")
            sys.exit(1)
    else:
        logging.info(f"术语文件已存在: {paths['terms_pkl']}")
    
    # Step 5: 训练模型
    if not skip_train:
        logging.info("\n[Step 5/10] 训练模型...")
        if not os.path.exists(paths['model_h5']):
            cmd = [
                'python', 'deepgoplus/train.py',
                '--go-file', paths['go_file'],
                '--train-data-file', paths['train_pkl'],
                '--valid-data-file', paths['val_pkl'],
                '--test-data-file', paths['test_pkl'],
                '--terms-file', paths['terms_pkl'],
                '--model-file', paths['model_h5'],
                '--out-file', os.path.join(data_root, f'{prefix}_predictions.pkl'),
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--device', device,
                '--learning-rate', str(learning_rate)
            ]
            logging.info(f"运行: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                logging.error("训练模型失败！")
                sys.exit(1)
        else:
            logging.info(f"模型已存在: {paths['model_h5']}")
    else:
        logging.info("\n[Step 5/10] 跳过训练步骤（使用已有模型）")
    
    # Step 6: 创建DIAMOND数据库
    logging.info("\n[Step 6/10] 创建DIAMOND数据库...")
    if not os.path.exists(paths['diamond_db_file']):
        cmd = [
            'diamond', 'makedb',
            '--in', paths['train_fasta'],
            '-d', paths['diamond_db']
        ]
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("创建DIAMOND数据库失败！")
            sys.exit(1)
    else:
        logging.info(f"DIAMOND数据库已存在: {paths['diamond_db_file']}")
    
    # Step 7: 进行预测
    if not skip_pred:
        logging.info("\n[Step 7/10] 进行预测...")
        cmd = [
            'python', 'deepgoplus/main.py',
            '--data-root', data_root,
            '--in-file', os.path.basename(paths['test_fasta']),
            '--out-file', os.path.join('results', os.path.basename(paths['pred_tsv'])),
            '--go-file', go_file,
            '--model-file', os.path.basename(paths['model_h5']),
            '--terms-file', os.path.basename(paths['terms_pkl']),
            '--annotations-file', os.path.basename(paths['train_pkl']),
            '--diamond-db', os.path.basename(paths['diamond_db_file']),
            '--threshold', str(threshold),
            '--batch-size', str(batch_size),
            '--alpha', str(alpha)
        ]
        if save_separate:
            cmd.append('--save-separate')
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("预测失败！")
            sys.exit(1)
    else:
        logging.info("\n[Step 7/10] 跳过预测步骤")
    
    # Step 8: 评估预测结果
    if not skip_eval:
        logging.info("\n[Step 8/10] 评估预测结果...")
        cmd = [
            'python', 'src/evaluation/evaluate_predictions.py',
            '--pred-file', paths['pred_tsv'],
            '--true-file', paths['test_pkl'],
            '--out-file', paths['eval_txt'],
            '--go-file', paths['go_file'],
            '--ont', ont
        ]
        logging.info(f"运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logging.error("评估失败！")
            sys.exit(1)
    else:
        logging.info("\n[Step 8/10] 跳过评估步骤")
    
    # Step 9: 分析预测结果（如果保存了单独预测）
    if save_separate and not skip_eval:
        logging.info("\n[Step 9/10] 分析预测结果...")
        diamond_file = paths['pred_tsv'].replace('.tsv', '_diamond_only.tsv')
        deep_file = paths['pred_tsv'].replace('.tsv', '_deep_only.tsv')
        
        if os.path.exists(diamond_file) and os.path.exists(deep_file):
            cmd = [
                'python', 'src/evaluation/analyze_predictions.py',
                '--pred-file', paths['pred_tsv'],
                '--diamond-file', diamond_file,
                '--deep-file', deep_file,
                '--true-file', paths['test_pkl'],
                '--train-file', paths['train_pkl'],
                '--go-file', paths['go_file'],
                '--ont', ont,
                '--out-file', paths['analysis_txt']
            ]
            logging.info(f"运行: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                logging.warning("分析失败，但继续执行...")
        else:
            logging.warning("未找到单独预测文件，跳过分析步骤")
    else:
        logging.info("\n[Step 9/10] 跳过分析步骤（需要 --save-separate 选项）")
    
    # Step 10: 诊断DeepGO模型（如果保存了单独预测）
    if save_separate and not skip_eval:
        logging.info("\n[Step 10/10] 诊断DeepGO模型...")
        deep_file = paths['pred_tsv'].replace('.tsv', '_deep_only.tsv')
        
        if os.path.exists(deep_file):
            cmd = [
                'python', 'src/evaluation/diagnose_model.py',
                '--deep-file', deep_file,
                '--true-file', paths['test_pkl'],
                '--terms-file', paths['terms_pkl'],
                '--go-file', paths['go_file'],
                '--ont', ont,
                '--out-file', paths['diagnosis_txt']
            ]
            logging.info(f"运行: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                logging.warning("诊断失败，但继续执行...")
        else:
            logging.warning("未找到DeepGO单独预测文件，跳过诊断步骤")
    else:
        logging.info("\n[Step 10/10] 跳过诊断步骤（需要 --save-separate 选项）")
    
    logging.info("\n" + "=" * 80)
    logging.info("工作流程完成！")
    logging.info("=" * 80)
    logging.info(f"\n输出文件:")
    logging.info(f"  预测结果: {paths['pred_tsv']}")
    if not skip_eval:
        logging.info(f"  评估结果: {paths['eval_txt']}")
        if save_separate:
            logging.info(f"  分析报告: {paths['analysis_txt']}")
            logging.info(f"  诊断报告: {paths['diagnosis_txt']}")


if __name__ == '__main__':
    main()

