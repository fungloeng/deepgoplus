#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 terms.pkl 文件
从训练数据 pkl 文件中提取所有唯一的术语
"""

import pandas as pd
import click as ck
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option('--train-data-file', '-tdf', required=True, 
           help='训练数据 pkl 文件（如 PF_train_data.pkl）')
@ck.option('--out-file', '-o', default='terms.pkl', 
           help='输出的 terms.pkl 文件路径')
def main(train_data_file, out_file):
    """从训练数据中提取所有唯一的术语并保存为 terms.pkl"""
    
    logging.info(f'Loading training data from {train_data_file}...')
    df = pd.read_pickle(train_data_file)
    
    # 收集所有术语
    all_terms = set()
    for prop_annots in df['prop_annotations']:
        all_terms |= prop_annots
    
    # 转换为排序列表
    terms_list = sorted(list(all_terms))
    
    logging.info(f'Found {len(terms_list)} unique terms')
    
    # 创建 DataFrame
    terms_df = pd.DataFrame({'terms': terms_list})
    
    # 保存
    terms_df.to_pickle(out_file)
    logging.info(f'Terms saved to {out_file}')
    
    # 显示前几个术语
    logging.info(f'First 10 terms: {terms_list[:10]}')

if __name__ == '__main__':
    main()