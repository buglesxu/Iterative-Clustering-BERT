from sklearn.cluster import KMeans
import numpy as np
import random
import pickle
import os
import time
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
# from bert4keras.snippets import open
from keras.layers import Dropout, Dense
from keras.models import Model

def evaluate_dataProcess(classesCount, predict, classify, distance, all_texts):

    classify2id = {}
    for i in range(classesCount):
        classify2id[i] = []

    for num, c in enumerate(classify):
        classify2id[c].append(num)

    final_res = {}
    for i in classify2id:
        res = {}
        for ids in classify2id[i]:
            if predict[ids] not in res:
                res[predict[ids]] = 1
            else:
                res[predict[ids]] += 1
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        final_res[i] = res
        print(res)

    res2id = {}
    for i in range(classesCount):
        res2id[i] = 0
    for i in final_res:
        classify_num = final_res[i][0][0]
        num = final_res[i][0][1]
        if classify_num not in res2id:
            res2id[classify_num] = num
        else:
            if res2id[classify_num] < num:
                res2id[classify_num] = num
    right = 0
    for i in res2id:
        # print(res2id[classify_num])
        right += res2id[i]
    print(res2id)
    print(len(classify))
    acc_all = right / len(classify)
    print("聚类准确率：" + str(acc_all))

    # 取前10%靠近聚类中心的样本，计算准确率
    clusterLabel2trueLabel = {}
    for i in range(classesCount):
        clusterLabel2trueLabel[final_res[i][0][0]] = i

    predict2id = {} #簇类别到样本id
    for i in range(classesCount):
        predict2id[i] = []
    for id in range(len(predict)):
        predict2id[predict[id]].append(id)
    for i in range(classesCount):
        X = predict2id[i]
        Y = [distance[id] for id in X]
        sorted_y_idx_list = sorted(range(len(Y)), key=lambda x: Y[x]) # 根据距离，从低到高排序，得到排序前的索引
        Xs = [X[i] for i in sorted_y_idx_list]
        predict2id[i] = Xs
    sum = 0
    right = 0



    texts = []
    ids = []

    for i in range(classesCount):
        X = predict2id[i]
        l = int(len(X)*0.01)
        sum += l
        for j in range(l):

            # 取%作为finetune数据
            texts.append(all_texts[X[j]])
            ids.append(clusterLabel2trueLabel[i])   # 将预测标签作为微调的label

            if clusterLabel2trueLabel[i] == classify[X[j]]:
                right += 1
    acc_core = right / sum
    print("靠近聚类中心准确率：" + str(acc_core))

    # 分割微调数据为train和validation
    train_dev_ids = [i for i in range(len(ids))]
    train_ids = random.sample(train_dev_ids, int(len(train_dev_ids) / 2))
    dev_ids = list(set(train_dev_ids) - set(train_ids))
    train_labels = [ids[i] for i in train_ids]
    dev_labels = [ids[i] for i in dev_ids]
    train_texts = [texts[i] for i in train_ids]
    dev_texts = [texts[i] for i in dev_ids]

    return acc_all, acc_core, train_texts, train_labels, dev_texts, dev_labels

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, num) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            labels = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([num])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]

        for i in range(len(y_true)):
            if y_pred[i] == y_true[i]:
                right += 1
        total += len(y_true)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(vaild_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(resultPath + './best_model.weights')
        print(val_acc)
        print(self.best_val_acc)

if __name__ == "__main__":

    # 读取提取的特征
    feature = np.loadtxt("./qinghua/wordvector_50000.txt")

    curTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    folderPath = "./qinghua/" + curTime.replace(':', '-')
    os.mkdir(folderPath)

    output = open('./qinghua/test_ids_50000.pkl', 'rb')
    classify = pickle.load(output)            # true label
    output = open('./qinghua/test_texts_50000.pkl', 'rb')
    all_texts = pickle.load(output)           # texts

    # 基本信息
    classesCount = 8
    maxlen = 128
    epochs = 10  # 13
    batch_size = 16  # 32
    learning_rate = 4e-5
    crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    bert = build_transformer_model(

        config_path=config_path,

        checkpoint_path=checkpoint_path,

        with_pool=True,

        return_keras_model=False,

    )

    classify_output = Dropout(rate=0.1)(bert.model.output)
    classify_output = Dense(units=classesCount,
                            activation='softmax',
                            name='classify_output',
                            kernel_initializer=bert.initializer
                            )(classify_output)

    model = keras.models.Model(bert.model.input, classify_output)
    model.summary()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate),
        metrics=['accuracy'],

    )

    for Round in range(50):
        print("Round" + str(Round))
        resultPath = folderPath + "/Round" + str(Round)
        os.mkdir(resultPath)

        # k-means 聚类
        clf = KMeans(n_clusters=classesCount)
        s = clf.fit(feature)
        predict = clf.predict(feature)
        d = clf.transform(feature)
        distance = []
        for i in range(len(predict)):
            distance.append(d[i][predict[i]])

        feature = None
        acc_all, acc_core, train_texts, train_ids, dev_texts, dev_ids = evaluate_dataProcess(classesCount, predict, classify, distance, all_texts)

        f = open(folderPath + "/acc.txt", 'a', encoding='UTF-8')
        f.write("第" + str(Round) + "轮: acc_all, acc_core:" + str(acc_all) + ", " + str(acc_core) + '\n')
        f.close()

        train_data = []
        for i in range(len(train_ids)):
            train_data.append((train_texts[i], train_ids[i]))
        vaild_data = []
        for i in range(len(dev_ids)):
            vaild_data.append((dev_texts[i], dev_ids[i]))



        train_generator = data_generator(train_data, batch_size)
        vaild_generator = data_generator(vaild_data, batch_size)
        evaluator = Evaluator()

        # adversarial_training(model, 'Embedding-Token', 0.2)

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            # class_weight = 'auto',
            callbacks=[evaluator]
        )

        layer_name = 'Transformer-11-FeedForward-Norm'  # 获取层的名称
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)  # 创建的新模型
        maxlen = 70
        vector_name = 'cls'

        print('开始转换')
        wordvector = []
        i = 0
        n = len(all_texts)
        for r in all_texts:
            print(i, "/", n)
            i = i + 1
            token_ids, segment_ids = tokenizer.encode(r, maxlen=maxlen)

            if vector_name == 'cls':

                cls_vector = intermediate_layer_model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
                wordvector.append(cls_vector)
            elif vector_name == 'mean':

                new = []
                vector = intermediate_layer_model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
                for i in range(768):
                    temp = 0
                    for j in range(len(vector)):
                        temp += vector[j][i]
                    new.append(temp / (len(vector)))
                wordvector.append(new)

        np.savetxt(resultPath + "/wordvector_50000.txt", wordvector)
        feature = wordvector