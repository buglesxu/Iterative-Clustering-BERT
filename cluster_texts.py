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

def evaluate(predict):

    # 评估准去率的函数
    # f1 = './data/data_split/test.json'
    # classify  = load_data(f1)

    output = open('./qinghua/test_ids.pkl', 'rb')
    classify = pickle.load(output)

    classify2id = {}
    for i in range(8):
        classify2id[i] = []

    for num,c in enumerate(classify):
        classify2id[c].append(num)

    final_res = {}
    for i in classify2id:
        res = {}
        for ids in classify2id[i]:
            if predict[ids] not in res:
                res[predict[ids]]  = 1
            else:
                res[predict[ids]] += 1   
        res = sorted(res.items(),key=lambda x:x[1],reverse=True)
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
    return acc

def load_data(filename):

    classify = ['财经/交易','产品行为','交往','竞赛行为','人生','司法行为','灾害/意外','组织行为','组织关系']
    D = []
    num = 0
    with open(filename,encoding='utf-8') as f:
        for l in f:
            num += 1
            l = json.loads(l)

            classify_name = l['event_list'][0]['class']

            for i ,c in enumerate(classify):
                if c in classify_name:
                    D.append(i)
                    break          
    return D

if __name__ == "__main__":
    
    # 读取提取的特征
    # feature = np.loadtxt("data/funtune-cls-val.txt")
    # feature = np.loadtxt("./funtune-cls-val_test.txt")
    feature = np.loadtxt("./wordvector_76000.txt")

    # k-means 聚类

    for i in range(6):
        clf = KMeans(n_clusters=8)
        s = clf.fit(feature)
        pre = clf.predict(feature)
        print(evaluate(pre))

    # birth聚类
   
    lists = [10,25]
    best_score = 0
    best_i = -1
    for i in lists:
        print(i)
        y_pred = Birch(branching_factor=i, n_clusters = 8, threshold=0.5,compute_labels=True).fit_predict(feature)
        score = evaluate(y_pred)
        if score > best_score:
            best_score = score
            best_i = i
        print(metrics.calinski_harabasz_score(feature, y_pred))
        print(best_score)
        print(best_i)
