#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
000_eval_base.py - DeepGOPlus 测试流水线
支持自定义本体、测试集和模型

用法示例：
python 000_eval_base.py -o mf -m data/model.h5 -n vanilla \
    -t data/MF_test_data.tsv -tf data/MF_test_sequences.fasta
"""

import argparse
import os
import csv
import subprocess
import sys
import logging
import re

logging.basicConfig(level=logging.INFO)

def run_cmd(cmd_list):
    """Run a command and print it."""
    print("Running:", " ".join(cmd_list))
    subprocess.run(cmd_list, check=True)

def to_pairs(in_tsv, out_pairs, go_col):
    """Convert wide TSV to two-column (acc <tab> GO_ID) format.
    
    Supports multiple TSV formats:
    - cafa format: has GO_MF_labels, GO_CC_labels, GO_BP_labels columns
    - galaxy format: has only GO_labels column (all ontologies combined)
    """
    if not os.path.exists(in_tsv):
        print(f"Warning: Input file {in_tsv} does not exist. Skipping...")
        return
    
    wrote = 0
    # First, detect the actual column name in the file by reading the header
    detected_go_col = None
    with open(in_tsv, newline='', encoding='utf-8') as fin:
        # Read just the first line to get column names
        header_line = fin.readline().strip()
        if not header_line:
            print(f"Warning: File {in_tsv} is empty. Skipping...")
            return
        
        available_columns = header_line.split('\t')
        
        # Try the specified column first (e.g., GO_MF_labels for cafa format)
        if go_col in available_columns:
            detected_go_col = go_col
        # Fallback to GO_labels (galaxy format)
        elif 'GO_labels' in available_columns:
            detected_go_col = 'GO_labels'
        # Try GO_MF_propagated, GO_CC_propagated, GO_BP_propagated (alternative format)
        elif go_col.replace('_labels', '_propagated') in available_columns:
            detected_go_col = go_col.replace('_labels', '_propagated')
        
        if detected_go_col:
            print(f"Detected GO column: {detected_go_col} in {in_tsv}")
        else:
            print(f"Warning: Could not detect GO column in {in_tsv}")
            print(f"  Expected: {go_col} or GO_labels")
            print(f"  Available columns: {available_columns}")
            return
    
    # Now process the file with the detected column
    with open(in_tsv, newline='', encoding='utf-8') as fin, \
         open(out_pairs, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin, delimiter='\t')
        for row in reader:
            acc = row.get('acc') or row.get('Entry') or row.get('Entry_clean')
            if not acc:
                continue
            
            # Use the detected column
            gos = (row.get(detected_go_col) or '').strip()
            if not gos:
                continue
            
            # 支持分号和逗号作为分隔符，使用正则表达式分割
            # 按分号或逗号分割，支持连续的分隔符
            go_list = re.split(r'[;,]+', gos)
            
            for go in go_list:
                go = go.strip()
                if go:  # 确保不是空字符串，且不包含分隔符
                    # 再次清理，确保没有残留的分隔符
                    go = go.strip(';,').strip()
                    if go:
                        fout.write(f"{acc}\t{go}\n")
                        wrote += 1
    
    print(f"Wrote {wrote} pairs to {out_pairs}")

def main():
    parser = argparse.ArgumentParser(description="DeepGOPlus evaluation pipeline")
    # 支持 --data-dir 和 -dd，但使用完整的名称以避免冲突
    parser.add_argument('--data-dir', '-dd', dest='data_dir', default='data', help='Data directory (default: data)')
    parser.add_argument('-o', '--ont', default='mf', choices=['mf','cc','bp','pf'], help='Ontology to evaluate')
    parser.add_argument('-m', '--model', default=None, help='Model H5 file (default: {data_dir}/model.h5)')
    parser.add_argument('-n', '--name', default='custom', help='Model name for output files')
    parser.add_argument('-t', '--test-tsv', required=True, help='Test TSV file')
    parser.add_argument('-tf', '--test-fasta', required=True, help='Test FASTA file')
    
    # 调试：打印原始命令行参数
    print(f"Debug: Raw command line arguments: {sys.argv}")
    
    args = parser.parse_args()
    
    # 调试：打印解析后的参数
    print(f"Debug: Parsed arguments:")
    print(f"  data_dir: {args.data_dir}")
    print(f"  ont: {args.ont}")
    print(f"  model: {args.model}")
    print(f"  test_tsv: {args.test_tsv}")
    print(f"  test_fasta: {args.test_fasta}")

    data_dir = args.data_dir.rstrip('/') if args.data_dir else 'data'  # 移除末尾的斜杠，避免路径问题
    ont = args.ont
    model_file = args.model
    model_name = args.name
    test_tsv = args.test_tsv
    test_fasta = args.test_fasta

    # 调试信息：打印接收到的参数
    print(f"Arguments received:")
    print(f"  Data directory (-dd): {data_dir}")
    print(f"  Ontology (-o): {ont}")
    print(f"  Test TSV (-t): {test_tsv}")
    print(f"  Test FASTA (-tf): {test_fasta}")

    # 如果模型文件未指定，使用默认路径
    if model_file is None:
        model_file = os.path.join(data_dir, 'model.h5')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)  # 为了向后兼容，仍然创建 data 目录
    os.makedirs("results", exist_ok=True)
    
    # 检查测试文件是否存在
    if not os.path.exists(test_tsv):
        print(f"Error: Test TSV file does not exist: {test_tsv}")
        print(f"  Current working directory: {os.getcwd()}")
        return
    if not os.path.exists(test_fasta):
        print(f"Error: Test FASTA file does not exist: {test_fasta}")
        print(f"  Current working directory: {os.getcwd()}")
        return

    # 设置 GO 列和训练集前缀
    go_cols = {
        'mf': 'GO_MF_labels', 
        'cc': 'GO_CC_labels', 
        'bp': 'GO_BP_labels',
        'pf': 'GO_PF_labels'
    }
    prefixes = {'mf': 'MF', 'cc': 'CC', 'bp': 'BP', 'pf': 'PF'}

    go_col = go_cols.get(ont, f'GO_{ont.upper()}_labels')
    prefix = prefixes.get(ont, ont.upper())

    print(f"[Step 1] Preparing train/validation/test pairs for {ont}...")
    print(f"Data directory: {data_dir}")

    # 生成训练/验证/测试 pairs
    # 使用数据目录中的文件，如果不存在则跳过（兼容galaxy和cafa）
    train_tsv = os.path.join(data_dir, f"{prefix}_train_data.tsv")
    val_tsv = os.path.join(data_dir, f"{prefix}_validation_data.tsv")
    
    to_pairs(train_tsv, os.path.join(data_dir, f"{prefix}_train_pairs.tsv"), go_col)
    to_pairs(val_tsv, os.path.join(data_dir, f"{prefix}_validation_pairs.tsv"), go_col)
    to_pairs(test_tsv, os.path.join(data_dir, f"{prefix}_test_pairs.tsv"), go_col)

    print(f"[Step 2] Creating DIAMOND database from {prefix} training set...")
    train_fasta = os.path.join(data_dir, f"{prefix}_train_sequences.fasta")
    diamond_db = os.path.join(data_dir, f"{prefix}_train_data")
    
    # 检查训练fasta文件是否存在
    if not os.path.exists(train_fasta):
        print(f"Warning: Training fasta file {train_fasta} does not exist. Skipping DIAMOND database creation.")
        print(f"  Please ensure the file exists or check the data directory path.")
    else:
        run_cmd(['diamond', 'makedb', '--in', train_fasta, '-d', diamond_db])

    print(f"[Step 3] Running DIAMOND blastp on custom test set...")
    diamond_output = os.path.join(data_dir, f"{prefix}_test_diamond.res")
    diamond_db_path = f'{diamond_db}.dmnd'
    
    # 检查DIAMOND数据库文件是否存在
    if not os.path.exists(diamond_db_path):
        print(f"Warning: DIAMOND database {diamond_db_path} does not exist. Skipping DIAMOND blastp.")
        print(f"  This usually means Step 2 was skipped due to missing training fasta file.")
    elif not os.path.exists(test_fasta):
        print(f"Warning: Test fasta file {test_fasta} does not exist. Skipping DIAMOND blastp.")
    else:
        run_cmd([
            'diamond', 'blastp', '-d', diamond_db_path, '--more-sensitive', '-t', '.',
            '-q', test_fasta,
            '--outfmt', '6',  # 正确参数
            '-o', diamond_output
        ])

    # Step 4: Prepare PKL files (if needed)
    print(f"[Step 4] Preparing PKL files for {ont}...")
    
    # Check if PKL files already exist
    train_pkl = os.path.join(data_dir, f"{prefix}_train_data.pkl")
    test_pkl = os.path.join(data_dir, f"{prefix}_test_data.pkl")
    val_pkl = os.path.join(data_dir, f"{prefix}_validation_data.pkl")
    
    go_file_path = os.path.join(data_dir, 'go.obo')
    if not os.path.exists(go_file_path):
        go_file_path = 'data/go.obo'  # fallback
    
    # Prepare train PKL if it doesn't exist
    if not os.path.exists(train_pkl):
        train_pairs = os.path.join(data_dir, f"{prefix}_train_pairs.tsv")
        train_fasta = os.path.join(data_dir, f"{prefix}_train_sequences.fasta")
        if os.path.exists(train_pairs) and os.path.exists(train_fasta):
            logging.info(f"Creating train PKL: {train_pkl}")
            run_cmd([
                'python', os.path.join(os.path.dirname(__file__), '..', 'data', 'prepare_data.py'),
                '--go-file', go_file_path,
                '--data-file', train_pairs,
                '--sequences-file', train_fasta,
                '--out-file', train_pkl,
                '--ont', ont
            ])
        else:
            logging.warning(f"Train pairs or FASTA not found, skipping train PKL creation")
    else:
        logging.info(f"Train PKL already exists: {train_pkl}")
    
    # Prepare test PKL if it doesn't exist
    if not os.path.exists(test_pkl):
        test_pairs = os.path.join(data_dir, f"{prefix}_test_pairs.tsv")
        if os.path.exists(test_pairs):
            logging.info(f"Creating test PKL: {test_pkl}")
            run_cmd([
                'python', os.path.join(os.path.dirname(__file__), '..', 'data', 'prepare_data.py'),
                '--go-file', go_file_path,
                '--data-file', test_pairs,
                '--sequences-file', test_fasta,
                '--out-file', test_pkl,
                '--ont', ont
            ])
        else:
            # Try using TSV directly if pairs don't exist
            if os.path.exists(test_tsv) and os.path.exists(test_fasta):
                logging.info(f"Creating test PKL directly from TSV: {test_pkl}")
                run_cmd([
                    'python', os.path.join(os.path.dirname(__file__), '..', 'data', 'prepare_data.py'),
                    '--go-file', go_file_path,
                    '--data-file', test_tsv,
                    '--sequences-file', test_fasta,
                    '--out-file', test_pkl,
                    '--ont', ont
                ])
            else:
                logging.warning(f"Test pairs/TSV or FASTA not found, skipping test PKL creation")
    else:
        logging.info(f"Test PKL already exists: {test_pkl}")
    
    # Prepare validation PKL if it doesn't exist
    if not os.path.exists(val_pkl):
        val_pairs = os.path.join(data_dir, f"{prefix}_validation_pairs.tsv")
        val_fasta = os.path.join(data_dir, f"{prefix}_validation_sequences.fasta")
        if os.path.exists(val_pairs) and os.path.exists(val_fasta):
            logging.info(f"Creating validation PKL: {val_pkl}")
            run_cmd([
                'python', os.path.join(os.path.dirname(__file__), '..', 'data', 'prepare_data.py'),
                '--go-file', go_file_path,
                '--data-file', val_pairs,
                '--sequences-file', val_fasta,
                '--out-file', val_pkl,
                '--ont', ont
            ])
        else:
            logging.warning(f"Validation pairs or FASTA not found, skipping validation PKL creation")
    else:
        logging.info(f"Validation PKL already exists: {val_pkl}")
    
    print(f"PKL files prepared. Train: {train_pkl}, Test: {test_pkl}, Validation: {val_pkl}")
    print(f"Next steps:")
    print(f"  1. Run prediction: python src/prediction/predict.py ...")
    print(f"  2. Run evaluation: python evaluate_deepgoplus2.py ...")

if __name__ == "__main__":
    main()
