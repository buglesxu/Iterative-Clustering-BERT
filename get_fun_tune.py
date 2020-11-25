#! -*- coding: utf-8 -*-
# 测试代码可用性: 提取特征

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import json
from keras.models import Model

from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
# from bert4keras.snippets import open
import pickle
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
# import pylcs
from keras.layers import Dropout, Dense
import random

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from keras_bert import extract_embeddings

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):

    D = []
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append(l['text'])
    return D

if __name__ == "__main__":
        
    # cls,mean,
    vector_name = 'cls'

    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

    classify_num_labels = 8

    bert = build_transformer_model(

        config_path=config_path,

        checkpoint_path=checkpoint_path,

        with_pool=True,

        return_keras_model=False,

    )


    classify_output = Dropout(rate=0.1)(bert.model.output)
    classify_output = Dense(units=classify_num_labels,
                    activation='softmax',
                    name='classify_output',
                    kernel_initializer=bert.initializer
                    )(classify_output)

    model = keras.models.Model(bert.model.input, classify_output)
    model.summary()

    model.summary()
    # model.load_weights('./chinese_L-12_H-768_A-12/best_model.weights')
    model.load_weights('./best_model2.weights')
    layer_name = 'Transformer-11-FeedForward-Norm' #获取层的名称
    intermediate_layer_model = Model(inputs=model.input, 
                                 outputs=model.get_layer(layer_name).output)#创建的新模型

    maxlen = 70

    # f1 = './data/data_split/test.json'
    # res = load_data(f1)

    # output = open('./qinghua/test_texts.pkl', 'rb')
    # texts = pickle.load(output)
    # output = open('./qinghua/test_ids.pkl', 'rb')
    # classify = pickle.load(output)
    output = open('./qinghua/test_texts.pkl', 'rb')
    texts = pickle.load(output)

    res = texts
    # ids = [i for i in range(len(texts))]
    # test_ids = random.sample(ids, 10000)
    # res = [texts[i] for i in test_ids]
    # classify = [classify[i] for i in test_ids]
    # output = open('./qinghua/test_texts_10000.pkl', 'wb')
    # pickle.dump(res, output)
    # output = open('./qinghua/test_ids_10000.pkl', 'wb')
    # pickle.dump(classify, output)

    output = []

    print('开始转换')
    i = 0
    n = len(res)
    for r in res:
        print(i, "/", n)
        i = i + 1
        token_ids, segment_ids = tokenizer.encode(r,maxlen=maxlen)

        if vector_name == 'cls':
            
            cls_vector = intermediate_layer_model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
            output.append(cls_vector)
        elif vector_name == 'mean':
            
            new = []
            vector = intermediate_layer_model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
            for i in range(768):
                temp = 0
                for j in range(len(vector)):
                    temp += vector[j][i]
                new.append(temp/(len(vector)))            
            output.append(new)

    # np.savetxt("funtune-cls-val_test.txt",output)
    np.savetxt("./qinghua/wordvector_76000_finetune3.txt", output)