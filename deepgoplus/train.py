#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepGOPlus Training Script
训练 DeepGOPlus 模型用于蛋白质功能预测
"""
import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Flatten, Concatenate,
    MaxPooling1D
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import roc_curve, auc

from deepgoplus.utils import Ontology
from deepgoplus.aminoacids import to_onehot, MAXLEN

logging.basicConfig(level=logging.INFO)

# Allow soft placement (CPU/GPU)
tf.config.set_soft_device_placement(True)


@ck.command()
@ck.option('--go-file', '-gf', default='data/go.obo', help='Gene Ontology file in OBO Format')
@ck.option('--train-data-file', '-trdf', default='data/train_data.pkl', help='Training data (PKL)')
@ck.option('--valid-data-file', '-vldf', default='', help='Validation data (PKL). If set, overrides --split')
@ck.option('--test-data-file', '-tsdf', default='data/test_data.pkl', help='Test data (PKL)')
@ck.option('--terms-file', '-tf', default='data/terms.pkl', help='Terms (PKL) with column "terms"')
@ck.option('--model-file', '-mf', default='data/model.h5', help='Output path for the trained model (H5)')
@ck.option('--out-file', '-o', default='data/predictions.pkl', help='Output PKL with test predictions')
@ck.option('--split', '-s', default=0.9, help='Train/valid split if --valid-data-file not provided')
@ck.option('--batch-size', '-bs', default=32, help='Batch size')
@ck.option('--epochs', '-e', default=12, help='Training epochs')
@ck.option('--learning-rate', '-lr', default=0.0003, type=float, help='Learning rate for Adam optimizer')
@ck.option('--load', '-ld', is_flag=True, help='Load existing model from --model-file and skip training')
@ck.option('--logger-file', '-lf', default='data/training.csv', help='CSVLogger output file')
@ck.option('--device', '-d', default='gpu:0', help='Device string, e.g., gpu:0 or cpu:0')
def main(go_file, train_data_file, valid_data_file, test_data_file, terms_file, model_file,
         out_file, split, batch_size, epochs, learning_rate, load, logger_file, device):
    """
    DeepGOPlus training with explicit validation support.
    - Uses Adam optimizer with configurable learning rate (default: 0.0003), binary_crossentropy, EarlyStopping(patience=6), ModelCheckpoint(save_best_only=True).
    - If --valid-data-file is provided, it is used as the validation set; otherwise, train is internally split.
    """

    params = {
        'max_kernel': 129,
        'initializer': 'glorot_normal',
        'dense_depth': 0,
        'nb_filters': 512,
        'optimizer': Adam(lr=learning_rate),
        'loss': 'binary_crossentropy'
    }
    logging.info('Params: %s', params)

    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)

    # Load train/valid
    if valid_data_file:
        logging.info('Using provided validation data file: %s', valid_data_file)
        train_df = pd.read_pickle(train_data_file)
        valid_df = pd.read_pickle(valid_data_file)
    else:
        train_df, valid_df = load_data(train_data_file, terms, split)

    test_df = pd.read_pickle(test_data_file)

    with tf.device('/' + device):
        test_steps = int(math.ceil(len(test_df) / batch_size))
        test_generator = DFGenerator(test_df, terms_dict, nb_classes, batch_size)
        if load:
            logging.info('Loading pretrained model: %s', model_file)
            model = load_model(model_file)
        else:
            logging.info('Creating a new model')
            model = create_model(nb_classes, params)

            logging.info("Training data size: %d", len(train_df))
            logging.info("Validation data size: %d", len(valid_df))
            checkpointer = ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
            logger = CSVLogger(logger_file)

            valid_steps = int(math.ceil(len(valid_df) / batch_size))
            train_steps = int(math.ceil(len(train_df) / batch_size))
            train_generator = DFGenerator(train_df, terms_dict, nb_classes, batch_size)
            valid_generator = DFGenerator(valid_df, terms_dict, nb_classes, batch_size)

            model.summary()
            model.fit(
                train_generator,
                steps_per_epoch=train_steps,
                epochs=epochs,
                validation_data=valid_generator,
                validation_steps=valid_steps,
                max_queue_size=batch_size,
                workers=12,
                callbacks=[logger, checkpointer, earlystopper]
            )
            logging.info('Loading best model from checkpoint')
            model = load_model(model_file)

        logging.info('Evaluating model on test set')
        loss = model.evaluate(test_generator, steps=test_steps)
        logging.info('Test loss %f', loss)
        logging.info('Predicting on test set')
        test_generator.reset()
        preds = model.predict(test_generator, steps=test_steps)

    # Build label matrix
    test_labels = np.zeros((len(test_df), nb_classes), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                test_labels[i, terms_dict[go_id]] = 1
    logging.info('Computing ROC AUC')
    roc_auc = compute_roc(test_labels, preds)
    logging.info('ROC AUC: %.2f', roc_auc)
    test_df['labels'] = list(test_labels)
    test_df['preds'] = list(preds)

    logging.info('Saving predictions to %s', out_file)
    test_df.to_pickle(out_file)


def compute_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def create_model(nb_classes, params):
    inp_hot = Input(shape=(MAXLEN, 21), dtype=np.float32)

    kernels = range(8, params['max_kernel'], 8)
    nets = []
    for i, k in enumerate(kernels):
        conv = Conv1D(
            filters=params['nb_filters'],
            kernel_size=k,
            padding='valid',
            name='conv_' + str(i),
            kernel_initializer=params['initializer']
        )(inp_hot)
        pool = MaxPooling1D(pool_size=MAXLEN - k + 1, name='pool_' + str(i))(conv)
        flat = Flatten(name='flat_' + str(i))(pool)
        nets.append(flat)

    net = Concatenate(axis=1)(nets)
    for i in range(params['dense_depth']):
        net = Dense(nb_classes, activation='relu', name='dense_' + str(i))(net)
    net = Dense(nb_classes, activation='sigmoid', name='dense_out')(net)
    model = Model(inputs=inp_hot, outputs=net)
    model.summary()
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    logging.info('Compilation finished')
    return model


def load_data(data_file, terms, split):
    df = pd.read_pickle(data_file)
    n = len(df)
    index = np.arange(n)
    train_n = int(n * split)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
    return train_df, valid_df


class DFGenerator(Sequence):
    def __init__(self, df, terms_dict, nb_classes, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.terms_dict = terms_dict

    def __len__(self):
        return np.ceil(len(self.df) / float(self.batch_size)).astype(np.int32)

    def __getitem__(self, idx):
        batch_index = np.arange(
            idx * self.batch_size, min(self.size, (idx + 1) * self.batch_size))
        df = self.df.iloc[batch_index]
        data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.float32)
        labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            onehot = to_onehot(seq)
            data_onehot[i, :, :] = onehot
            for t_id in row.prop_annotations:
                if t_id in self.terms_dict:
                    labels[i, self.terms_dict[t_id]] = 1
        return (data_onehot, labels)

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.int32)
            labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                seq = row.sequences
                onehot = to_onehot(seq)
                data_onehot[i, :, :] = onehot
                for t_id in row.prop_annotations:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data_onehot, labels)
        else:
            self.reset()
            return self.next()


if __name__ == '__main__':
    main()