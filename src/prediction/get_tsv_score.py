#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import run
import time
import sys
import os
# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Add src directory to path for imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from deepgoplus.utils import Ontology, NAMESPACES
from deepgoplus.aminoacids import to_onehot
import os
import sys
import logging
import json

MAXLEN = 2000

@ck.command()
@ck.option('--data-root', '-dr', default='/autodl-tmp/deepgoplus/data', help='Data root folder', required=True)
@ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option('--out-file', '-of', default='/autodl-tmp/deepgoplus/results/results.tsv', help='Output result file')
@ck.option('--go-file', '-gf', default='go.obo', help='Gene Ontology')
@ck.option('--model-file', '-mf', default='model.h5', help='Tensorflow model file')
@ck.option('--terms-file', '-tf', default='terms.pkl', help='List of predicted terms')
@ck.option('--annotations-file', '-af', default='train_data.pkl', help='Experimental annotations')
@ck.option('--diamond-db', '-dd', default='train_data.dmnd', help='Diamond Database file')
@ck.option('--diamond-file', '-df', default='diamond.res', help='Diamond Mapping file')
@ck.option('--chunk-size', '-cs', default=1000, help='Number of sequences to read at a time')
@ck.option('--threshold', '-t', default=0.0, help='Prediction threshold (0.0 = save all predictions)')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option('--alpha', '-a', default=0.5, help='Alpha weight parameter')
def main(data_root, in_file, out_file, go_file, model_file, terms_file, annotations_file,
         diamond_db, diamond_file, chunk_size, threshold, batch_size, alpha):

    # 拼接完整路径
    data_root = os.path.abspath(data_root)
    go_file = os.path.join(data_root, go_file)
    model_file = os.path.join(data_root, model_file)
    terms_file = os.path.join(data_root, terms_file)
    annotations_file = os.path.join(data_root, annotations_file)
    diamond_db = os.path.join(data_root, diamond_db)
    diamond_file = os.path.join(data_root, diamond_file)
    in_file = os.path.abspath(in_file)
    out_file = os.path.abspath(out_file)

    # 设置日志级别
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 检查必需文件
    missing_files = []
    for f in [go_file, model_file, terms_file, annotations_file, diamond_db]:
        if not os.path.exists(f):
            logging.warning(f"警告: 文件缺失 {f}, 部分功能可能会受影响")
            missing_files.append(f)
    
    if not os.path.exists(in_file):
        logging.error(f"错误: 输入文件不存在: {in_file}")
        sys.exit(1)
    
    logging.info(f"输入文件: {in_file}")
    logging.info(f"输出文件: {out_file}")
    logging.info(f"GO文件: {go_file}")
    logging.info(f"模型文件: {model_file}")
    logging.info(f"Terms文件: {terms_file}")

    # Load GO
    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()

    # Load alphas
    metadata_path = os.path.join(data_root, 'metadata/last_release.json')
    if not os.path.exists(metadata_path):
        # 尝试在当前目录或父目录查找
        possible_paths = [
            'metadata/last_release.json',
            '../metadata/last_release.json',
            '../../metadata/last_release.json'
        ]
        metadata_path = None
        for path in possible_paths:
            if os.path.exists(path):
                metadata_path = path
                break
    
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                alphas_dict = metadata.get('alphas', {})
                # 支持所有本体，包括pf
                alphas = {}
                for ont_key in ['mf', 'bp', 'cc', 'pf']:
                    if ont_key in alphas_dict:
                        if ont_key in NAMESPACES:
                            alphas[NAMESPACES[ont_key]] = alphas_dict[ont_key]
                        else:
                            # 对于pf等不在NAMESPACES中的，使用默认值
                            alphas[ont_key] = alphas_dict[ont_key]
                    else:
                        # 如果元数据中没有，使用默认值0.5
                        default_alpha = 0.5
                        if ont_key in NAMESPACES:
                            alphas[NAMESPACES[ont_key]] = default_alpha
                        else:
                            alphas[ont_key] = default_alpha
        except Exception as e:
            logging.warning(f"Failed to load metadata from {metadata_path}: {e}. Using default alpha=0.5")
            # 使用默认值
            alphas = {ns: 0.5 for ns in NAMESPACES.values()}
            alphas['pf'] = 0.5  # PF 不在 NAMESPACES 中
    else:
        logging.warning(f"Metadata file not found. Using default alpha=0.5 for all ontologies")
        alphas = {ns: 0.5 for ns in NAMESPACES.values()}
        alphas['pf'] = 0.5  # PF 不在 NAMESPACES 中

    # Load experimental annotations
    annotations = {}
    if os.path.exists(annotations_file):
        df = pd.read_pickle(annotations_file)
        for row in df.itertuples():
            annotations[row.proteins] = set(row.prop_annotations)
    else:
        logging.warning(f"Annotations file {annotations_file} 不存在，DIAMOND 相似性将无法使用")

    # 运行 DIAMOND（如果结果文件不存在）
    diamond_preds = {}
    if os.path.exists(diamond_file) and os.path.getsize(diamond_file) > 0:
        logging.info(f"DIAMOND结果文件已存在: {diamond_file}，直接使用")
        # 解析现有的DIAMOND结果
        mapping = {}
        try:
            with open(diamond_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    it = line.split()
                    if len(it) < 3:
                        continue
                    query_id = it[0]
                    subject_id = it[1]
                    # 尝试获取bitscore（通常在第三列）
                    try:
                        bitscore = float(it[2])
                    except (ValueError, IndexError):
                        # 如果第三列不是数字，尝试其他列
                        try:
                            bitscore = float(it[11])  # bitscore在format 6的第12列
                        except (ValueError, IndexError):
                            logging.warning(f"无法解析DIAMOND行: {line[:50]}...")
                            continue
                    
                    if query_id not in mapping:
                        mapping[query_id] = {}
                    mapping[query_id][subject_id] = bitscore
        except Exception as e:
            logging.error(f"读取DIAMOND结果文件时出错: {e}")
            mapping = {}
    elif os.path.exists(diamond_db):
        logging.info(f"DIAMOND结果文件不存在，运行DIAMOND...")
        cmd = [
            "diamond", "blastp",
            "-d", diamond_db,
            "--more-sensitive",
            "-t", "/tmp",
            "-q", in_file,
            "--outfmt", "6", "qseqid", "sseqid", "bitscore",  # 明确指定列
            "-o", diamond_file
        ]
        proc = run(cmd)
        if proc.returncode != 0:
            logging.error('Error running DIAMOND! 将跳过 DIAMOND 部分')
            diamond_preds = {}
            mapping = {}
        else:
            # 解析新生成的DIAMOND结果
            mapping = {}
            try:
                with open(diamond_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        it = line.split()
                        if len(it) < 3:
                            continue
                        query_id = it[0]
                        subject_id = it[1]
                        try:
                            bitscore = float(it[2])
                        except (ValueError, IndexError):
                            continue
                        if query_id not in mapping:
                            mapping[query_id] = {}
                        mapping[query_id][subject_id] = bitscore
            except Exception as e:
                logging.error(f"解析DIAMOND结果时出错: {e}")
                mapping = {}
    else:
        logging.warning(f"Diamond DB {diamond_db} 不存在，将跳过 DIAMOND")
        mapping = {}
    
    # 处理DIAMOND结果并生成预测
    if mapping:
        for prot_id, sim_prots in mapping.items():
            annots = {}
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                if p_id in annotations:
                    allgos |= annotations[p_id]
                    total_score += score
            if total_score == 0:
                continue  # 跳过没有注释的蛋白质
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if p_id in annotations and go_id in annotations[p_id]:
                        s += score
                sim[j] = s / total_score
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
            diamond_preds[prot_id] = annots

    # Load CNN model
    logging.info(f"加载模型: {model_file}")
    model = load_model(model_file)
    logging.info(f"模型加载成功，预测术语数量: {len(terms)}")

    # 预测并输出结果
    start_time = time.time()
    total_seq = 0
    total_written = 0
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as w:
        for prot_ids, sequences in read_fasta(in_file, chunk_size):
            if not prot_ids or not sequences:
                logging.warning("跳过空的序列块")
                continue
            
            total_seq += len(prot_ids)
            logging.info(f"处理 {len(prot_ids)} 个序列 (总计: {total_seq})...")
            
            deep_preds = {}
            ids, data = get_data(sequences)
            preds = model.predict(data, batch_size=batch_size, verbose=0)
            
            if preds.shape[1] != len(terms):
                logging.error(f"预测维度不匹配: 预测={preds.shape[1]}, 术语数={len(terms)}")
                continue
            for i, j in enumerate(ids):
                prot_id = prot_ids[j]
                if prot_id not in deep_preds:
                    deep_preds[prot_id] = {}
                for l in range(len(terms)):
                    if preds[i, l] >= 0.01:
                        deep_preds[prot_id][terms[l]] = max(deep_preds[prot_id].get(terms[l], 0), preds[i, l])

            # Combine diamond + deepgo
            for prot_id in prot_ids:
                annots = {}
                if prot_id in diamond_preds:
                    for go_id, score in diamond_preds[prot_id].items():
                        # 检查go_id是否在GO ontology中
                        if go.has_term(go_id):
                            try:
                                ns = go.get_namespace(go_id)
                                alpha_val = alphas.get(ns, 0.5)
                            except:
                                alpha_val = 0.5
                            annots[go_id] = score * alpha_val
                        else:
                            # 对于非GO术语（如PF），使用默认alpha
                            alpha_val = alphas.get(go_id.split(':')[0].lower(), 0.5)
                            annots[go_id] = score * alpha_val
                
                for go_id, score in deep_preds.get(prot_id, {}).items():
                    if go.has_term(go_id):
                        try:
                            ns = go.get_namespace(go_id)
                            alpha_val = alphas.get(ns, 0.5)
                        except:
                            alpha_val = 0.5
                    else:
                        # 对于非GO术语（如PF），使用默认alpha
                        alpha_val = alphas.get(go_id.split(':')[0].lower(), 0.5)
                    
                    if go_id in annots:
                        annots[go_id] += (1 - alpha_val) * score
                    else:
                        annots[go_id] = (1 - alpha_val) * score

                # Propagate scores (only for GO terms)
                for go_id in list(annots.keys()):
                    if go.has_term(go_id):
                        try:
                            for g_id in go.get_anchestors(go_id):
                                annots[g_id] = max(annots.get(g_id, 0), annots[go_id])
                        except:
                            pass  # 如果获取祖先失败，跳过

                # 写入结果（按分数降序排列）
                sorted_annots = sorted(annots.items(), key=lambda x: x[1], reverse=True)
                written_count = 0
                for go_id, score in sorted_annots:
                    if score >= threshold:
                        w.write(f"{prot_id}\t{go_id}\t{score:.3f}\n")
                        written_count += 1
                        total_written += 1
                
                # 调试信息：如果某个蛋白质没有预测结果
                if written_count == 0 and len(sorted_annots) > 0:
                    max_score = sorted_annots[0][1]
                    if max_score > 0:
                        logging.debug(f"蛋白质 {prot_id} 的最高分数 {max_score:.3f} 低于阈值 {threshold}")

    total_time = time.time() - start_time
    logging.info(f'总预测时间: {total_time:.2f}秒')
    logging.info(f'处理序列总数: {total_seq}')
    logging.info(f'写入结果总数: {total_written}')
    logging.info(f'平均每个序列的预测数: {total_written/total_seq if total_seq > 0 else 0:.2f}')
    logging.info(f"结果已保存到: {out_file}")
    
    if total_written == 0:
        logging.warning(f"警告: 没有写入任何结果！")
        logging.warning(f"  可能的原因:")
        logging.warning(f"  1. 预测分数都低于阈值 {threshold}")
        logging.warning(f"  2. 模型预测结果为空")
        logging.warning(f"  3. 序列格式不正确")
        logging.warning(f"  请检查输入文件和模型是否正确")


def read_fasta(filename, chunk_size):
    """Read FASTA file in chunks."""
    seqs, info, seq, inf = [], [], '', ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    seqs.append(seq)
                    info.append(inf)
                    if len(info) == chunk_size:
                        yield info, seqs
                        seqs, info = [], []
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
    if seq:
        seqs.append(seq)
        info.append(inf)
    if info:
        yield info, seqs

def get_data(sequences):
    pred_seqs, ids = [], []
    for i, seq in enumerate(sequences):
        if len(seq) > MAXLEN:
            st = 0
            while st < len(seq):
                pred_seqs.append(seq[st:st+MAXLEN])
                ids.append(i)
                st += MAXLEN - 128
        else:
            pred_seqs.append(seq)
            ids.append(i)
    n = len(pred_seqs)
    data = np.zeros((n, MAXLEN, 21), dtype=np.float32)
    for i in range(n):
        data[i, :, :] = to_onehot(pred_seqs[i])
    return ids, data

if __name__ == '__main__':
    main()
