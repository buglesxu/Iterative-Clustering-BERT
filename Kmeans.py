from sklearn.cluster import KMeans
from sklearn.cluster import Birch
# from sklearn.externals import joblib
import numpy as np
import random
import json
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn import metrics
import copy
import pickle
import fine_tune


def evaluate(predict, distance):
    # 评估准确率的函数
    # f1 = './data/data_split/test.json'
    # classify  = load_data(f1)

    output = open('./qinghua/train_ids.pkl', 'rb')
    classify = pickle.load(output)

    classify2id = {}
    for i in range(8):
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
    for i in range(8):
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
    acc = right / len(classify)
    print(acc)

    # 取前10%靠近聚类中心的样本，计算准确率
    clusterLabel2trueLabel = {}
    for i in range(8):
        clusterLabel2trueLabel[final_res[i][0][0]] = i

    predict2id = {} #簇类别到样本id
    for i in range(8):
        predict2id[i] = []
    for id in range(len(predict)):
        predict2id[predict[id]].append(id)
    for i in range(8):
        X = predict2id[i]
        Y = [distance[id] for id in X]
        sorted_y_idx_list = sorted(range(len(Y)), key=lambda x: Y[x]) # 根据距离，从低到高排序，得到排序前的索引
        Xs = [X[i] for i in sorted_y_idx_list]
        predict2id[i] = Xs
    sum = 0
    right = 0

    output = open('./qinghua/test_texts.pkl', 'rb')
    train_texts = pickle.load(output)
    texts = []
    ids = []

    for i in range(8):
        X = predict2id[i]
        l = int(len(X)*0.01)
        sum += l
        for j in range(l):

            # 取%作为finetune数据
            texts.append(train_texts[X[j]])
            ids.append(clusterLabel2trueLabel[i])

            if clusterLabel2trueLabel[i] == classify[X[j]]:
                right += 1
    acc = right / sum
    print(acc)

    # train_dev_ids = [i for i in range(len(ids))]
    # train_ids = random.sample(train_dev_ids, int(len(train_dev_ids) / 2))
    # dev_ids = list(set(train_dev_ids) - set(train_ids))
    # train_labels = [ids[i] for i in train_ids]
    # dev_labels = [ids[i] for i in dev_ids]
    # train_texts = [texts[i] for i in train_ids]
    # dev_texts = [texts[i] for i in dev_ids]
    # output = open('./qinghua/train_texts_0.01.pkl', 'wb')
    # pickle.dump(train_texts, output)
    # output = open('./qinghua/dev_texts_0.01.pkl', 'wb')
    # pickle.dump(dev_texts, output)
    # output = open('./qinghua/train_ids_0.01.pkl', 'wb')
    # pickle.dump(train_labels, output)
    # output = open('./qinghua/dev_ids_0.01.pkl', 'wb')
    # pickle.dump(dev_labels, output)

    return acc


def load_data(filename):
    classify = ['财经/交易', '产品行为', '交往', '竞赛行为', '人生', '司法行为', '灾害/意外', '组织行为', '组织关系']
    D = []
    num = 0
    with open(filename, encoding='utf-8') as f:
        for l in f:
            num += 1
            l = json.loads(l)

            classify_name = l['event_list'][0]['class']

            for i, c in enumerate(classify):
                if c in classify_name:
                    D.append(i)
                    break
    return D


if __name__ == "__main__":

    # 读取提取的特征
    # feature = np.loadtxt("data/funtune-cls-val.txt")
    # feature = np.loadtxt("./funtune-cls-val_test.txt")
    feature = np.loadtxt("./qinghua/wordvector_4000_finetune.txt")
    # feature = np.loadtxt("./wordvector_76000.txt")

    # k-means 聚类


    clf = KMeans(n_clusters=8)
    s = clf.fit(feature)
    pre = clf.predict(feature)
    d = clf.transform(feature)
    distance = []
    for i in range(len(pre)):
        distance.append(d[i][pre[i]])
    print(evaluate(pre, distance))

    # # birth聚类
    #
    # lists = [10, 25]
    # best_score = 0
    # best_i = -1
    # for i in lists:
    #     print(i)
    #     y_pred = Birch(branching_factor=i, n_clusters=8, threshold=0.5, compute_labels=True).fit_predict(feature)
    #     score = evaluate(y_pred)
    #     if score > best_score:
    #         best_score = score
    #         best_i = i
    #     print(metrics.calinski_harabasz_score(feature, y_pred))
    #     print(best_score)
    #     print(best_i)
