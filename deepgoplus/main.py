#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import Popen, PIPE
import time
from deepgoplus.utils import Ontology, NAMESPACES
from deepgoplus.aminoacids import to_onehot
import gzip
import os
import sys
import logging
import subprocess
import json

MAXLEN = 2000

def get_go_namespace_safe(go, go_id):
    """
    Safely get GO namespace, handling cases where go_id might be a semicolon-separated list.
    For non-GO labels such as PFxxxxx (Pfam families), return a synthetic
    'pf' namespace so that they are not filtered out downstream.
    Returns None if the term doesn't exist or is invalid.
    """
    # Check if go_id contains semicolons (multiple GO IDs)
    if ';' in go_id:
        # If it's a semicolon-separated list, try the first one
        go_id = go_id.split(';')[0].strip()

    # 如果是 Pfam 家族标签（例如 PF07686），GO 本体中不会包含这些术语，
    # 但我们仍然希望在 PF 模式下进行预测和评估，因此直接返回一个虚拟命名空间。
    if go_id.startswith('PF'):
        return 'pf'
    
    if not go.has_term(go_id):
        return None
    
    try:
        return go.get_namespace(go_id)
    except (KeyError, AttributeError):
        return None

@ck.command()
@ck.option('--data-root', '-dr', default='data/', help='Data root folder', required=True)
@ck.option('--in-file', '-if', help='Input FASTA file', required=True)
@ck.option('--out-file', '-of', default='results.tsv', help='Output result file')
@ck.option('--go-file', '-gf', default='go.obo', help='Gene Ontology')
@ck.option('--model-file', '-mf', default='model.h5', help='Tensorflow model file')
@ck.option('--terms-file', '-tf', default='terms.pkl', help='List of predicted terms')
@ck.option('--annotations-file', '-tf', default='train_data.pkl', help='Experimental annotations')
@ck.option('--diamond-db', '-dd', default='train_data.dmnd', help='Diamond Database file')
@ck.option('--diamond-file', '-df', default='diamond.res', help='Diamond Mapping file')
@ck.option('--chunk-size', '-cs', default=1000, help='Number of sequences to read at a time')
@ck.option('--threshold', '-t', default=0.1, help='Prediction threshold')
@ck.option('--batch-size', '-bs', default=32, help='Batch size for prediction model')
@ck.option('--alpha', '-a', default=0.5, help='Alpha weight parameter')
@ck.option('--save-separate', '-ss', is_flag=True, help='Save separate predictions for DIAMOND and DeepGO')
def main(data_root, in_file, out_file, go_file, model_file, terms_file, annotations_file,
         diamond_db, diamond_file, chunk_size, threshold, batch_size, alpha, save_separate):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            go_file = os.path.join(data_root, go_file)
            model_file = os.path.join(data_root, model_file)
            terms_file = os.path.join(data_root, terms_file)
            annotations_file = os.path.join(data_root, annotations_file)
            diamond_db = os.path.join(data_root, diamond_db)
            diamond_file = os.path.join(data_root, diamond_file)
            if not os.path.exists(go_file):
                raise Exception(f'Gene Ontology file ({go_file}) is missing!')
            if not os.path.exists(model_file):
                raise Exception(f'Model file ({model_file}) is missing!')
            if not os.path.exists(terms_file):
                raise Exception(f'Terms file ({terms_file}) is missing!')
            if not os.path.exists(annotations_file):
                raise Exception(f'Annotations file ({annotations_file}) is missing!')
            if not os.path.exists(diamond_db):
                raise Exception(f'Diamond database ({diamond_db}) is missing!')
        else:
            raise Exception(f'Data folder {data_root} does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)
    
    # Resolve input file path
    # First check if it's an absolute path or exists in current directory
    if os.path.isabs(in_file) or os.path.exists(in_file):
        in_file_path = in_file
    else:
        # Try to find it in data_root
        in_file_path = os.path.join(data_root, in_file)
        if not os.path.exists(in_file_path):
            # Try with just the filename in data_root
            in_file_basename = os.path.basename(in_file)
            in_file_path = os.path.join(data_root, in_file_basename)
            if not os.path.exists(in_file_path):
                logging.error(f'Input file not found: {in_file}')
                logging.error(f'Tried locations:')
                logging.error(f'  1. {in_file} (as provided)')
                logging.error(f'  2. {os.path.join(data_root, in_file)}')
                logging.error(f'  3. {in_file_path}')
                sys.exit(1)
    
    if not os.path.exists(in_file_path):
        logging.error(f'Input file ({in_file_path}) does not exist!')
        sys.exit(1)
    
    in_file = in_file_path  # Update in_file to the resolved path
    logging.info(f'Using input file: {in_file}')
    
    # Resolve output file path (if not absolute, save to data_root)
    if not os.path.isabs(out_file):
        out_file = os.path.join(data_root, out_file)
    logging.info(f'Output file will be saved to: {out_file}')

    # Load GO and read list of all terms
    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()

    # Always use command line alpha parameter (ignore metadata file)
    # This allows users to control the alpha value via command line
    logging.info(f"Using command line alpha parameter: {alpha}")
    
    # Read known experimental annotations
    annotations = {}
    df = pd.read_pickle(annotations_file)
    for row in df.itertuples():
        annotations[row.proteins] = set(row.prop_annotations)

    # Generate diamond predictions
    cmd = [
        "diamond", "blastp",  "-d", diamond_db, "--more-sensitive", "-t", "/tmp",
        "-q", in_file, "--outfmt", "6", "qseqid", "sseqid", "bitscore", "-o",
        diamond_file]
    proc = subprocess.run(cmd)

    if proc.returncode != 0:
        logging.error('Error running diamond!')
        sys.exit(1)

    diamond_preds = {}
    mapping = {}
    with open(diamond_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in mapping:
                mapping[it[0]] = {}
            mapping[it[0]][it[1]] = float(it[2])
    for prot_id, sim_prots in mapping.items():
        annots = {}
        allgos = set()
        total_score = 0.0
        for p_id, score in sim_prots.items():
            allgos |= annotations[p_id]
            total_score += score
        allgos = list(sorted(allgos))
        sim = np.zeros(len(allgos), dtype=np.float32)
        for j, go_id in enumerate(allgos):
            s = 0.0
            for p_id, score in sim_prots.items():
                if go_id in annotations[p_id]:
                    s += score
            sim[j] = s / total_score
        for go_id, score in zip(allgos, sim):
            annots[go_id] = score
        diamond_preds[prot_id] = annots
    
    # Load CNN model
    model = load_model(model_file)
    
    # Use command line alpha parameter for all namespaces
    # This allows users to control the DIAMOND/DeepGO weight ratio
    # 对于 PF（蛋白家族）标签，我们使用一个虚拟命名空间 'pf'，不依赖 GO 结构
    alphas = {
        NAMESPACES['mf']: alpha,
        NAMESPACES['bp']: alpha,
        NAMESPACES['cc']: alpha,
        NAMESPACES.get('pf', 'pf'): alpha,
    }
    logging.info(f"Using alpha={alpha} for namespaces (MF, BP, CC, PF)")
    logging.info(f"  DIAMOND weight: {alpha}, DeepGO weight: {1-alpha}")
    
    start_time = time.time()
    total_seq = 0
    w = open(out_file, 'w')
    # Write header
    w.write('protein_id\tgo_id\tscore\n')
    
    # Optionally save separate predictions
    w_diamond = None
    w_deep = None
    if save_separate:
        diamond_out = out_file.replace('.tsv', '_diamond_only.tsv')
        deep_out = out_file.replace('.tsv', '_deep_only.tsv')
        w_diamond = open(diamond_out, 'w')
        w_deep = open(deep_out, 'w')
        w_diamond.write('protein_id\tgo_id\tscore\n')
        w_deep.write('protein_id\tgo_id\tscore\n')
        logging.info(f"Saving separate predictions: {diamond_out}, {deep_out}")
    
    for prot_ids, sequences in read_fasta(in_file, chunk_size):
        total_seq += len(prot_ids)
        deep_preds = {}
        ids, data = get_data(sequences)

        preds = model.predict(data, batch_size=batch_size)
        assert preds.shape[1] == len(terms)
        for i, j in enumerate(ids):
            prot_id = prot_ids[j]
            if prot_id not in deep_preds:
                deep_preds[prot_id] = {}
            for l in range(len(terms)):
                if preds[i, l] >= 0.01: # Filter out very low scores
                    term = terms[l]
                    # Handle semicolon-separated GO IDs
                    if ';' in term:
                        # Split and process each GO ID separately
                        go_ids = [g.strip() for g in term.split(';') if g.strip()]
                        for go_id in go_ids:
                            if go_id not in deep_preds[prot_id]:
                                deep_preds[prot_id][go_id] = preds[i, l]
                            else:
                                deep_preds[prot_id][go_id] = max(
                                    deep_preds[prot_id][go_id], preds[i, l])
                    else:
                        # Single GO ID
                        if term not in deep_preds[prot_id]:
                            deep_preds[prot_id][term] = preds[i, l]
                        else:
                            deep_preds[prot_id][term] = max(
                                deep_preds[prot_id][term], preds[i, l])
        # Combine diamond preds and deepgo
        for prot_id in prot_ids:
            annots = {}
            if prot_id in diamond_preds:
                for go_id, score in diamond_preds[prot_id].items():
                    namespace = get_go_namespace_safe(go, go_id)
                    if namespace and namespace in alphas:
                        annots[go_id] = score * alphas[namespace]
            for go_id, score in deep_preds[prot_id].items():
                namespace = get_go_namespace_safe(go, go_id)
                if namespace and namespace in alphas:
                    if go_id in annots:
                        annots[go_id] += (1 - alphas[namespace]) * score
                    else:
                        annots[go_id] = (1 - alphas[namespace]) * score
            # Propagate scores with ontology structure
            gos = list(annots.keys())
            for go_id in gos:
                # Skip if go_id contains semicolons (invalid format)
                if ';' in go_id:
                    continue
                if not go.has_term(go_id):
                    continue
                try:
                    for g_id in go.get_anchestors(go_id):
                        if g_id in annots:
                            annots[g_id] = max(annots[g_id], annots[go_id])
                        else:
                            annots[g_id] = annots[go_id]
                except (KeyError, AttributeError):
                    # Skip if there's an error getting ancestors
                    continue
                
            # Write combined predictions in long format: protein_id\tgo_id\tscore
            for go_id, score in annots.items():
                if score >= threshold:
                    w.write(f'{prot_id}\t{go_id}\t{score}\n')
            
            # Optionally write separate predictions
            if save_separate:
                # Write DIAMOND-only predictions
                if prot_id in diamond_preds:
                    for go_id, score in diamond_preds[prot_id].items():
                        namespace = get_go_namespace_safe(go, go_id)
                        if namespace and namespace in alphas:
                            diamond_score = score * alphas[namespace]
                            if diamond_score >= threshold:
                                w_diamond.write(f'{prot_id}\t{go_id}\t{diamond_score}\n')
                
                # Write DeepGO-only predictions
                if prot_id in deep_preds:
                    for go_id, score in deep_preds[prot_id].items():
                        namespace = get_go_namespace_safe(go, go_id)
                        if namespace and namespace in alphas:
                            deep_score = (1 - alphas[namespace]) * score
                            if deep_score >= threshold:
                                w_deep.write(f'{prot_id}\t{go_id}\t{deep_score}\n')
    
    w.close()
    if save_separate:
        w_diamond.close()
        w_deep.close()
    total_time = time.time() - start_time
    print('Total prediction time for %d sequences is %d' % (total_seq, total_time))


def read_fasta(filename, chunk_size):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    if len(info) == chunk_size:
                        yield (info, seqs)
                        seqs = list()
                        info = list()
                    seq = ''
                inf = line[1:].split()[0]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    yield (info, seqs)

def get_data(sequences):
    pred_seqs = []
    ids = []
    for i, seq in enumerate(sequences):
        if len(seq) > MAXLEN:
            st = 0
            while st < len(seq):
                pred_seqs.append(seq[st: st + MAXLEN])
                ids.append(i)
                st += MAXLEN - 128
        else:
            pred_seqs.append(seq)
            ids.append(i)
    n = len(pred_seqs)
    data = np.zeros((n, MAXLEN, 21), dtype=np.float32)
    
    for i in range(n):
        seq = pred_seqs[i]
        data[i, :, :] = to_onehot(seq)
    return ids, data


if __name__ == '__main__':
    main()
