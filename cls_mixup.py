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
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_gelu('tanh')  # 切换gelu版本

maxlen = 48
batch_size = 24
learning_rate = 1e-5

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

    output = []

    new_data = copy.copy(D)
    random.shuffle(new_data)

    for l in D:
        text = l["sentence"]
        label = l["label"]
        category.add(label)
    for time in range(2):

        random.shuffle(new_data)

        for i in range(len(D)):

            d_1 = D[i]
            d_2 = new_data[i]

            text_1 = d_1["sentence"]
            label_1 = d_1["label"]

            text_2 = d_2["sentence"]
            label_2 = d_2["label"]

            mixup_rate = random.uniform(0, 2) / 10.0

            output.append([text_1, text_2, label_1, label_2, mixup_rate])

    return output


def load_test_data(D):

    output = []

    for l in D:
        text = l["sentence"]
        label = l["label"]
        output.append([text, text, label, label, 0])

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
        batch_token_ids, batch_segment_ids, batch_token_2_ids, batch_segment_2_ids, batch_labels, batch_2_labels, batch_mixup_labels = [], [], [], [], [], [], []
        for is_end, (
                text_1,
                text_2,
                label_1,
                label_2,
                mixup_rate,
        ) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text_1, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            labals = [0] * len(label2id)
            labals[label2id[label_1]] = 1

            batch_labels.append(labals)

            token_2_ids, segment_2_ids = tokenizer.encode(text_2,
                                                          maxlen=maxlen)
            batch_token_2_ids.append(token_2_ids)
            batch_segment_2_ids.append(segment_2_ids)

            labals_2 = [0] * len(label2id)
            labals_2[label2id[label_2]] = 1

            batch_2_labels.append(labals_2)

            mixup_labels = [0] * len(label2id)
            mixup_labels[
                label2id[label_1]] = mixup_labels[label2id[label_1]] + 0.8
            mixup_labels[
                label2id[label_2]] = mixup_labels[label2id[label_2]] + 0.2

            batch_mixup_labels.append(mixup_labels)

            if len(batch_token_ids) == self.batch_size or is_end:

                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_token_2_ids = sequence_padding(batch_token_2_ids)
                batch_segment_2_ids = sequence_padding(batch_segment_2_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_2_labels = sequence_padding(batch_2_labels)
                mixup_labels = sequence_padding(batch_mixup_labels)

                yield [
                    batch_token_ids, batch_segment_ids, batch_token_2_ids,
                    batch_segment_2_ids
                ], [batch_labels, batch_2_labels, mixup_labels]

                batch_token_ids, batch_segment_ids, batch_token_2_ids, batch_segment_2_ids, batch_labels, batch_2_labels, batch_mixup_labels = [], [], [], [], [], [], []


# 加载预训练模型
bert_1 = build_transformer_model(config_path, checkpoint_path)

bert_1_cls = Lambda(lambda x: x[:, 0], name='bert_1_CLS-token')(bert_1.output)

bert_1_output = Dense(len(category),
                      activation='sigmoid',
                      name='bert_1_output')(bert_1_cls)

bert_2 = build_transformer_model(config_path, checkpoint_path)

for layer in bert_2.layers:
    layer.name = layer.name + str("_2")

bert_2_cls = Lambda(lambda x: x[:, 0], name='bert_2_CLS-token')(bert_2.output)

bert_2_output = Dense(len(category),
                      activation='sigmoid',
                      name='bert_2_output')(bert_2_cls)

output_1 = Lambda(lambda x: x * 0.8)(bert_1_cls)
output_2 = Lambda(lambda x: x * 0.2)(bert_2_cls)

output = Add()([output_1, output_2])

mixup_output = Dense(len(category), activation='sigmoid',
                     name='mixup_output')(output)

model = Model(bert_1.inputs + bert_2.inputs,
              [bert_1_output, bert_2_output, mixup_output])

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate),  # 用足够小的学习率
    loss_weights={
        'bert_1_output': 0.2,
        'bert_2_output': 0.2,
        'mixup_output': 1.
    },
    metrics=['accuracy'],
)

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true)[1].argmax(axis=1)
        y_true = y_true[1].argmax(axis=1)
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
            model.save_weights('model/news_2.weights')
        print(u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
              (val_acc, self.best_val_acc, 0))


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_generator.forfit(),
              steps_per_epoch=len(train_generator),
              epochs=12,
              callbacks=[evaluator])

    model.load_weights("model/news_2.weights")
    dev_data = fa.read("data/news_dev.json")
    dev_data = load_test_data(dev_data)
    test_generator = data_generator(dev_data, batch_size)
    score = evaluate(test_generator)
    print(score)