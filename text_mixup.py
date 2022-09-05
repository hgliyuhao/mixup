#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

from random import random
import numpy as np
from keras.layers import *
from keras.models import *
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
import fairies as fa
from tqdm import tqdm
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
set_gelu('tanh')  # 切换gelu版本

maxlen = 48
batch_size = 64

p = '/home/pre_models/chinese-roberta-wwm-ext-tf/'
config_path = p + 'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p + 'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

category = set()


def load_data(D):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    # D = fa.read(fileName)

    output = []

    for l in D:
        text = l["sentence"]
        label = l["label"]
        category.add(label)

    for l in D:

        for i in range(1):
            text = l["sentence"]
            label = l["label"]

            output.append((text, label, label, 0))

            diff = random.choice(D)
            diff_text = diff["sentence"]
            diff_label = diff["label"]
            mixup_rate = random.uniform(0, 2) / 10.0

            mixup_text = text[:int(len(text) *
                                   (1 - mixup_rate))] + diff_text[:int(
                                       len(diff_text) * (1 - mixup_rate))]

            output.append((mixup_text, label, diff_label, mixup_rate))

    random.shuffle(output)

    return output


def load_test_data(fileName):

    D = fa.read(fileName)

    output = []

    for l in D:
        text = l["sentence"]
        label = l["label"]
        output.append((text, label, label, 0))

    return output


import random

random.seed(0)

datas = fa.read("data/news_train.json")

train_data = [d for i, d in enumerate(datas) if i % 10 != 0]
valid_data = [d for i, d in enumerate(datas) if i % 10 == 0]

train_data = load_data(train_data)
valid_data = load_test_data(valid_data)

print('数据处理完成')

category = list(category)
category.sort()

id2label, label2id = fa.label2id(category)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (mixup_text, label, diff_label,
                     mixup_rate) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(mixup_text,
                                                      maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            labals = [0] * len(label2id)
            labals[label2id[label]] = labals[label2id[label]] + 1 - mixup_rate
            labals[label2id[diff_label]] = labals[
                label2id[diff_label]] + mixup_rate

            batch_labels.append(labals)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(config_path, checkpoint_path)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.output)

final_output = Dense(len(category), activation='sigmoid')(output)

model = Model(bert.inputs, final_output)

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('model/news.weights')
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
              (val_acc, self.best_val_acc, 0))


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_generator.forfit(),
              steps_per_epoch=len(train_generator),
              epochs=12,
              callbacks=[evaluator])
