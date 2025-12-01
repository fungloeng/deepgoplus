#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import math
import logging
import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

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
from deepgoplus.utils import Ontology
from deepgoplus.aminoacids import to_onehot, MAXLEN

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


# -------------------------------
# ✅ GPU 设置：显存自适应 + 混合精度
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logging.info("GPU memory growth enabled + mixed precision set.")
    except Exception as e:
        logging.warning(f"GPU setup failed: {e}")


# -------------------------------
# ✅ 命令行参数定义
# -------------------------------
@ck.command()
@ck.option('--go-file', '-gf', default='data/go.obo', help='Gene Ontology OBO file')
@ck.option('--test-data-file', '-tsdf', default='data/MF_test_data.pkl', help='Test data file (.pkl)')
@ck.option('--terms-file', '-tf', default='data/terms.pkl', help='GO terms file (.pkl)')
@ck.option('--model-file', '-mf', default='data/model_mf.h5', help='Trained DeepGOPlus model (.h5)')
@ck.option('--out-file', '-o', default='data/MF_predictions_v2.pkl', help='Output predictions file')
@ck.option('--batch-size', '-bs', default=64, help='Batch size')
@ck.option('--device', '-d', default='gpu:0', help='Device, e.g., gpu:0 or cpu:0')
def main(go_file, test_data_file, terms_file, model_file, out_file, batch_size, device):
    """Run DeepGOPlus prediction with progress bar and optimized loading."""
    logging.info("Starting DeepGOPlus prediction (optimized)...")

    # -------------------------------
    # 加载 GO term 与测试数据
    # -------------------------------
    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    nb_classes = len(terms)
    test_df = pd.read_pickle(test_data_file)
    terms_dict = {v: i for i, v in enumerate(terms)}

    logging.info(f"Loaded {len(test_df)} sequences for testing.")
    logging.info(f"Number of GO terms: {nb_classes}")

    # -------------------------------
    # 加载模型
    # -------------------------------
    with tf.device('/' + device):
        logging.info(f"Loading model from {model_file} ...")
        model = load_model(model_file)
        logging.info("Model loaded successfully.")

        # -------------------------------
        # 创建数据生成器
        # -------------------------------
        test_steps = int(math.ceil(len(test_df) / batch_size))
        test_gen = DFGenerator(test_df, terms_dict, nb_classes, batch_size)

        # -------------------------------
        # tqdm 进度条预测
        # -------------------------------
        preds_list = []
        for i in tqdm(range(test_steps), desc="Predicting batches", ncols=100):
            batch_x, _ = test_gen[i]
            batch_pred = model.predict(batch_x, verbose=0)
            preds_list.append(batch_pred)

        preds = np.concatenate(preds_list, axis=0)

    # -------------------------------
    # 保存预测结果
    # -------------------------------
    test_df['preds'] = list(preds)
    test_df.to_pickle(out_file)
    logging.info(f"Predictions saved to {out_file}")
    logging.info("✅ Prediction completed successfully!")


# -------------------------------
# ✅ 数据生成器 (优化版)
# -------------------------------
class DFGenerator(Sequence):
    def __init__(self, df, terms_dict, nb_classes, batch_size):
        self.df = df
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.terms_dict = terms_dict
        self.size = len(df)
        self.cache = {}  # 缓存 to_onehot 结果加速

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_index = np.arange(
            idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))
        df = self.df.iloc[batch_index]
        data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            if seq in self.cache:
                onehot = self.cache[seq]
            else:
                onehot = to_onehot(seq)
                self.cache[seq] = onehot
            data_onehot[i, :, :] = onehot

            for t_id in getattr(row, 'prop_annotations', []):
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1

        return (data_onehot, labels)


# -------------------------------
# ✅ 主入口
# -------------------------------
if __name__ == '__main__':
    main()
