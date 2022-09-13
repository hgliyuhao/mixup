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
batch_size = 32
learning_rate = 8e-6

p = '/home/pre_models/chinese-roberta-wwm-ext-tf/'
config_path = p + 'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p + 'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

category = set()

D = fa.read("data/news_train.json")

for l in D:
    text = l["sentence"]
    label = l["label"]
    category.add(label)

category = list(category)
category.sort()

id2label, label2id = fa.label2id(category)


def load_data(D):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """

    output = []

    new_data = copy.copy(D)
    random.shuffle(new_data)
    mix_times = 1

    for l in D:
        text = l["sentence"]
        label = l["label"]

        mixup_labels = [0] * len(label2id)
        mixup_labels[label2id[label]] = 1

        output.append([text, text, mixup_labels])

    for t in range(mix_times):
        for i in range(len(D)):

            d_1 = D[i]
            d_2 = new_data[i]

            text_1 = d_1["sentence"]
            label_1 = d_1["label"]

            text_2 = d_2["sentence"]
            label_2 = d_2["label"]

            mixup_labels = [0] * len(label2id)
            mixup_labels[label2id[label_1]] = 0.8
            mixup_labels[
                label2id[label_2]] = mixup_labels[label2id[label_2]] + 0.2

            if label_1 != label_2:
                output.append([text_1, text_2, mixup_labels])

    random.shuffle(output)

    return output


def load_test_data(D):

    output = []

    for l in D:
        text = l["sentence"]
        label = l["label"]

        mixup_labels = [0] * len(label2id)
        mixup_labels[label2id[label]] = 1

        output.append([text, text, mixup_labels])

    return output


import random

random.seed(0)

datas = fa.read("data/news_train.json")

train_data = [d for i, d in enumerate(datas) if i % 10 != 0]
valid_data = [d for i, d in enumerate(datas) if i % 10 == 0]

train_data = load_data(train_data)
valid_data = load_test_data(valid_data)

print('数据处理完成')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (
                text_1,
                text_2,
                label,
        ) in self.sample(random):

            for text in [text_1, text_2]:

                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(label)

            if len(batch_token_ids) == self.batch_size * 2 or is_end:

                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def merge(inputs):
    """向量合并：a、b、|a-b|拼接
    """
    a, b = inputs[::2] * 0.8, inputs[1::2] * 0.2
    o = a + b
    return K.repeat_elements(o, 2, 0)


# 加载预训练模型
bert = build_transformer_model(config_path, checkpoint_path)

output = Lambda(lambda x: x[:, 0], )(bert.output)

encoder = keras.models.Model(bert.inputs, output)

output = keras.layers.Lambda(merge)(output)
output = keras.layers.Dense(units=15, activation='sigmoid')(output)
model = keras.models.Model(bert.inputs, output)

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate),  # 用足够小的学习率
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

    # model.fit(train_generator.forfit(),
    #           steps_per_epoch=len(train_generator),
    #           epochs=12,
    #           callbacks=[evaluator])

    model.load_weights("model/news.weights")
    dev_data = fa.read("data/news_dev.json")
    dev_data = load_test_data(dev_data)
    test_generator = data_generator(dev_data, batch_size)
    score = evaluate(test_generator)
    print(score)

    # 0.5716