#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
import csv
import os
import re
import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Add src directory to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from deepgoplus.utils import Ontology, FUNC_DICT, read_fasta
import logging

logging.basicConfig(level=logging.INFO)

def detect_tsv_format(data_file):
    """检测TSV文件的格式：pairs格式或wide格式"""
    with open(data_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            return None, None
        
        columns = first_line.split('\t')
        
        # 检查是否是pairs格式（只有两列，且第二列看起来像GO ID）
        if len(columns) == 2:
            # 读取第二行确认
            second_line = f.readline().strip()
            if second_line:
                parts = second_line.split('\t')
                if len(parts) == 2 and (parts[1].startswith('GO:') or parts[1].startswith('PF')):
                    return 'pairs', None
        
        # 检查是否是wide格式（包含GO标签列）
        go_cols = {
            'GO_MF_labels', 'GO_CC_labels', 'GO_BP_labels', 'GO_PF_labels',
            'GO_MF_propagated', 'GO_CC_propagated', 'GO_BP_propagated',
            'GO_labels'  # galaxy格式
        }
        
        detected_cols = {}
        for col in columns:
            if col in go_cols:
                detected_cols[col] = col
        
        if detected_cols:
            return 'wide', detected_cols
        
        # 如果都不匹配，检查是否有sequence列
        if 'sequence' in columns:
            return 'wide', {}  # 可能是wide格式但没有GO列（用于预测）
        
        return None, None

def load_annotations_from_pairs(data_file):
    """从pairs格式TSV加载标注"""
    annotations = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 2:
                continue
            prot_id = row[0].strip()
            go_id = row[1].strip()
            if not prot_id or not go_id:
                continue
            if prot_id not in annotations:
                annotations[prot_id] = set()
            annotations[prot_id].add(go_id)
    return annotations

def load_annotations_from_wide(data_file, go_columns=None, ont=None):
    """从wide格式TSV加载标注
    
    Args:
        data_file: TSV文件路径
        go_columns: GO列名字典，如果为None则自动检测
        ont: 本体（mf/cc/bp/pf），用于选择特定的GO列
    """
    annotations = {}
    
    # 确定要使用的GO列
    go_cols_to_use = []
    
    if ont:
        # 如果指定了本体，优先使用特定本体的列
        ont_specific_col = f'GO_{ont.upper()}_labels'
        ont_propagated_col = f'GO_{ont.upper()}_propagated'
        
        if go_columns and ont_specific_col in go_columns:
            go_cols_to_use.append(ont_specific_col)
        elif go_columns and ont_propagated_col in go_columns:
            go_cols_to_use.append(ont_propagated_col)
        elif go_columns and 'GO_labels' in go_columns:
            # galaxy格式：使用GO_labels（包含所有本体）
            go_cols_to_use.append('GO_labels')
    else:
        # 如果没有指定本体，使用所有可用的GO列
        if go_columns:
            go_cols_to_use = list(go_columns.keys())
        else:
            # 如果没有提供列信息，尝试检测
            go_cols_to_use = ['GO_MF_labels', 'GO_CC_labels', 'GO_BP_labels', 'GO_PF_labels', 'GO_labels']
    
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            acc = row.get('acc') or row.get('Entry') or row.get('Entry_clean')
            if not acc:
                continue
            
            annots = set()
            for go_col in go_cols_to_use:
                if go_col in row:
                    gos = (row.get(go_col) or '').strip()
                    if gos:
                        # 支持分号和逗号作为分隔符，使用正则表达式分割
                        # 按分号或逗号分割，支持连续的分隔符
                        go_list = re.split(r'[;,]+', gos)
                        for go in go_list:
                            go = go.strip()
                            if go:  # 确保不是空字符串，且不包含分隔符
                                # 再次清理，确保没有残留的分隔符
                                go = go.strip(';,').strip()
                                if go:
                                    annots.add(go)
            
            if annots:
                annotations[acc] = annots
    
    return annotations

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', required=True,
    help='Data file with sequences and annotations (TSV format: pairs or wide)')
@ck.option(
    '--sequences-file', '-sf', required=True,
    help='Sequences fasta file')
@ck.option(
    '--out-file', '-of', required=True,
    help='Output pkl file')
@ck.option(
    '--ont', '-o', default=None,
    type=ck.Choice(['mf', 'cc', 'bp', 'pf']),
    help='Ontology to filter annotations (optional). If not specified, use all annotations')
@ck.option(
    '--use-tsv-sequences', '-uts', is_flag=True,
    help='Use sequences from TSV file instead of FASTA file (if TSV has sequence column)')
def main(go_file, data_file, sequences_file, out_file, ont, use_tsv_sequences):
    """准备数据文件，从TSV和FASTA文件创建pkl格式的数据文件。
    
    支持两种TSV格式：
    1. Pairs格式：两列 (prot_id \t GO_id)
    2. Wide格式：多列，包含GO_MF_labels, GO_CC_labels等列
    
    如果TSV文件包含sequence列，可以使用--use-tsv-sequences直接从TSV读取序列。
    """
    logging.info('Loading GO')
    go = Ontology(go_file, with_rels=True)
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.exists(sequences_file):
        raise FileNotFoundError(f"Sequences file not found: {sequences_file}")
    
    # 检测TSV格式
    logging.info('Detecting TSV format...')
    tsv_format, go_columns = detect_tsv_format(data_file)
    
    if tsv_format is None:
        raise ValueError(f"无法识别TSV文件格式: {data_file}")
    
    logging.info(f'Detected TSV format: {tsv_format}')
    if go_columns:
        logging.info(f'Detected GO columns: {list(go_columns.keys())}')
    
    # 加载标注
    logging.info('Loading annotations...')
    if tsv_format == 'pairs':
        annotations = load_annotations_from_pairs(data_file)
    else:  # wide format
        annotations = load_annotations_from_wide(data_file, go_columns, ont)
    
    logging.info(f'Loaded annotations for {len(annotations)} proteins')
    
    # 统计标注数量
    total_annots = sum(len(annots) for annots in annotations.values())
    logging.info(f'Total annotations: {total_annots}')
    
    # 加载序列
    if use_tsv_sequences and tsv_format == 'wide':
        # 从TSV读取序列
        logging.info('Loading sequences from TSV file...')
        tsv_sequences = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                acc = row.get('acc') or row.get('Entry') or row.get('Entry_clean')
                seq = row.get('sequence') or row.get('Sequence')
                if acc and seq:
                    tsv_sequences[acc] = seq
        
        logging.info(f'Loaded {len(tsv_sequences)} sequences from TSV')
        
        # 使用TSV中的序列
        info, seqs = [], []
        for acc, seq in tsv_sequences.items():
            info.append(acc)
            seqs.append(seq)
    else:
        # 从FASTA文件读取序列
        logging.info('Loading sequences from FASTA file...')
        info, seqs = read_fasta(sequences_file)
    
    # 合并序列和标注
    proteins = []
    sequences = []
    annots = []
    
    # 创建序列字典以便快速查找
    seq_dict = {}
    for prot_info, sequence in zip(info, seqs):
        prot_id = prot_info.split()[0]
        seq_dict[prot_id] = sequence
    
    # 匹配序列和标注
    matched_count = 0
    unmatched_count = 0
    
    for prot_id, annot_set in annotations.items():
        if prot_id in seq_dict:
            proteins.append(prot_id)
            sequences.append(seq_dict[prot_id])
            annots.append(annot_set)
            matched_count += 1
        else:
            unmatched_count += 1
    
    if unmatched_count > 0:
        logging.warning(f'{unmatched_count} proteins have annotations but no matching sequence')
    
    # 如果有一些序列在FASTA中但没有标注，也添加它们（标注为空集）
    for prot_id, sequence in seq_dict.items():
        if prot_id not in annotations:
            proteins.append(prot_id)
            sequences.append(sequence)
            annots.append(set())
    
    logging.info(f'Matched {matched_count} proteins with annotations and sequences')
    logging.info(f'Total proteins in output: {len(proteins)}')
    
    # 传播标注（添加祖先术语）
    logging.info('Propagating annotations...')
    prop_annotations = []
    for annot_set in annots:
        propagated_set = set(annot_set)  # 包含原始术语
        for go_id in annot_set:
            try:
                propagated_set |= go.get_anchestors(go_id)
            except:
                # 如果术语不在GO中（如PF），保持原样
                pass
        prop_annotations.append(propagated_set)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'proteins': proteins,
        'sequences': sequences,
        'prop_annotations': prop_annotations,
    })
    
    # 统计信息
    non_empty_annots = sum(1 for a in prop_annotations if len(a) > 0)
    logging.info(f'Proteins: {len(df)}')
    logging.info(f'Proteins with annotations: {non_empty_annots}/{len(df)}')
    logging.info(f'Saving data to {out_file}')
    
    df.to_pickle(out_file)
    logging.info('Done!')

if __name__ == '__main__':
    main()