from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import json
import random
import jieba
import pandas as pd
import csv
import pickle

def evaluate(predict, classify):

    # 评估准去率的函数
    # f1 = './data/train.json'
    # classify  = load_label(f1)

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

# 得到train.json的标签
def load_label(filename):

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

# 得到train.json的文本
def load_text(filename):

    D = []
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append(l['text'])
    return D

# 得到停用词列表
def stopwordslist(path):

    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

# 得到LDA聚类结果的标签
def getLabel(file):
    res = []
    with open(file, 'r') as f:
        next(f)
        for line in f:
            l = line.split()
            res.append(int(l[2]))
    return res

def getLdaInput_trainJson():
    # 把train.json转换为分词后的数据集，作为LDA的输入
    # *************************************************************************
    f1 = './data/train.json'
    texts = load_text(f1)
    text_list = []
    stopwords = stopwordslist("./DataZh/cn_stopwords.txt")
    for text in texts:
        sent_list = jieba.cut(text)
        outstr = ''
        for word in sent_list:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        outstr = outstr.replace('\n', '')
        text_list.append(outstr)

    fW = open("./DataZh/data.txt", 'w', encoding='UTF-8')
    for text in text_list:
        fW.write(text + '\n')
    fW.close()
    # *************************************************************************

    # 评估LDA准确率
    LDA_labels = getLabel("./DataZh/result.txt")
    print(evaluate(LDA_labels))

def splitTrainJson():
    # 将train.json抽取10%语料进行fine tune
    # *************************************************************************
    ans = []
    f2 = open('./data/train.json', encoding='UTF-8')
    for l in f2:
        ans.append(json.loads(l))
    train_dev_len = int(len(ans) * 0.1)
    ids = [i for i in range(len(ans))]
    train_dev_ids = random.sample(ids, train_dev_len)
    train_ids = random.sample(train_dev_ids, int(train_dev_len / 2))
    dev_ids = list(set(train_dev_ids) - set(train_ids))
    test_ids = list(set(ids) - set(train_ids))

    train = [ans[i] for i in train_ids]
    dev = [ans[i] for i in dev_ids]
    test = [ans[i] for i in test_ids]

    f2_w = open('./data/data_split/train.json', 'w', encoding='UTF-8')
    for l in train:
        f2_w.write(json.dumps(l, ensure_ascii=False) + '\n')
    f2_w = open('./data/data_split/dev.json', 'w', encoding='UTF-8')
    for l in dev:
        f2_w.write(json.dumps(l, ensure_ascii=False) + '\n')
    f2_w = open('./data/data_split/test.json', 'w', encoding='UTF-8')
    for l in test:
        f2_w.write(json.dumps(l, ensure_ascii=False) + '\n')
    # *************************************************************************

def getLdaInputSogouData():
    # # 把Sogou_data.csv处理成LDA所需格式.txt
    # # ['sports', 'house', 'business', 'entertainment', 'women', 'technology']
    # # *************************************************************************
    # # test = pd.read_csv('./data/Sogou_data/test.csv', encoding='UTF-8')
    # csvfile = open('./data/Sogou_data/test.csv', "r", encoding='UTF-8')
    # reader = csv.reader(csvfile)
    # fW = open("./data/Sogou_data/test.txt", 'w', encoding='UTF-8')
    # for item in reader:
    #     fW.write(item[0][1:].replace(' ', '') + '\n')
    # fW.close()
    #
    #
    # labels = ['sports', 'house', 'business', 'entertainment', 'women', 'technology']
    # f = './data/Sogou_data/test.txt'
    # fR = open(f, 'r', encoding='UTF-8')
    # texts = []
    # while True:
    #     line = fR.readline()
    #     if not line:
    #         break
    #     texts.append(line.strip().replace(' ', ''))
    # text_list = []
    # stopwords = stopwordslist("./DataZh/cn_stopwords.txt")
    # for text in texts:
    #     sent_list = jieba.cut(text)
    #     outstr = ''
    #     for word in sent_list:
    #         if word not in stopwords:
    #             # if word != '\t':
    #             outstr += word
    #             outstr += " "
    #     text_list.append(outstr)
    #
    # fW = open("./data/Sogou_data/LDA.txt", 'w', encoding='UTF-8')
    # for text in text_list:
    #     fW.write(text + '\n')
    # fW.close()
    # # *************************************************************************

    # 评估准确率
    csvfile = open('./data/Sogou_data/test.csv', "r", encoding='utf-8-sig')
    reader = csv.reader(csvfile)
    classify = []

    # fW = open("./data/Sogou_data/test.txt", 'w', encoding='UTF-8')
    for item in reader:
        classify.append(int(item[0][0]))
    LDA_labels = getLabel("./data/Sogou_data/Sogou_news.txt")
    print(evaluate(LDA_labels, classify))

def qinghua_split():
    # ["0时政", "1财经", "2家具", "3教育", "4体育", "5游戏", "6科技", "7时尚"]
    texts = []
    labels = []
    for i in range(8):
        fileName = "./qinghua/data/{}.txt".format(i)
        f = open(fileName, "r", encoding='UTF-8')
        for line in f:
            if len(line) < 500:
                texts.append(line)
            else:
                texts.append(line[:200] + line[len(line) - 300:])
            labels.append(i)

    # train_dev_len = int(len(labels) * 0.1)
    train_dev_len = 30000
    ids = [i for i in range(len(labels))]
    train_dev_ids = random.sample(ids, train_dev_len)
    train_ids = random.sample(train_dev_ids, int(train_dev_len / 2))
    dev_ids = list(set(train_dev_ids) - set(train_ids))
    test_ids = list(set(ids) - set(train_dev_ids))

    train_texts = [texts[i] for i in train_ids]
    dev_texts = [texts[i] for i in dev_ids]
    test_texts = [texts[i] for i in test_ids]

    train_labels = [labels[i] for i in train_ids]
    dev_labels = [labels[i] for i in dev_ids]
    test_labels = [labels[i] for i in test_ids]

    # output = open('./qinghua/train_texts.pkl', 'wb')
    # pickle.dump(train_texts, output)
    # output = open('./qinghua/dev_texts.pkl', 'wb')
    # pickle.dump(dev_texts, output)
    output = open('./qinghua/test_texts_50000.pkl', 'wb')
    pickle.dump(test_texts, output)
    # output = open('./qinghua/train_ids.pkl', 'wb')
    # pickle.dump(train_labels, output)
    # output = open('./qinghua/dev_ids.pkl', 'wb')
    # pickle.dump(dev_labels, output)
    output = open('./qinghua/test_ids_50000.pkl', 'wb')
    pickle.dump(test_labels, output)

def getLdaInputQinghuaData():
    output = open('./qinghua/test_texts.pkl', 'rb')
    test_texts = pickle.load(output)
    fW = open("./qinghua/LDA_all.txt", 'w', encoding='UTF-8')

    stopwords = stopwordslist("./DataZh/cn_stopwords.txt")
    for text in test_texts:
        sent_list = jieba.cut(text)
        outstr = ''
        for word in sent_list:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        # outstr = outstr.replace('\n', '')
        fW.write(outstr)

    output = open('./qinghua/test_ids.pkl', 'rb')
    classify = pickle.load(output)
    LDA_labels = getLabel("./qinghua/qinghua_all.txt")
    print(evaluate(LDA_labels, classify))

def sogou():
    # # 把Sogou_data.csv处理成LDA所需格式.txt
    # # ['sports', 'house', 'business', 'entertainment', 'women', 'technology']
    # # *************************************************************************
    # # test = pd.read_csv('./data/Sogou_data/test.csv', encoding='UTF-8')
    # csvfile = open('./data/Sogou_data/test.csv', "r", encoding='UTF-8')
    # reader = csv.reader(csvfile)
    # fW = open("./data/Sogou_data/test.txt", 'w', encoding='UTF-8')
    # for item in reader:
    #     fW.write(item[0][1:].replace(' ', '') + '\n')
    # fW.close()
    #
    #
    # labels = ['sports', 'house', 'business', 'entertainment', 'women', 'technology']
    # f = './data/Sogou_data/test.txt'
    # fR = open(f, 'r', encoding='UTF-8')
    # texts = []
    # while True:
    #     line = fR.readline()
    #     if not line:
    #         break
    #     texts.append(line.strip().replace(' ', ''))
    # text_list = []
    # stopwords = stopwordslist("./DataZh/cn_stopwords.txt")
    # for text in texts:
    #     sent_list = jieba.cut(text)
    #     outstr = ''
    #     for word in sent_list:
    #         if word not in stopwords:
    #             # if word != '\t':
    #             outstr += word
    #             outstr += " "
    #     text_list.append(outstr)
    #
    # fW = open("./data/Sogou_data/LDA.txt", 'w', encoding='UTF-8')
    # for text in text_list:
    #     fW.write(text + '\n')
    # fW.close()
    # # *************************************************************************

    # 评估准确率
    csvfile = open('./data/Sogou_data/test.csv', "r", encoding='utf-8-sig')
    reader = csv.reader(csvfile)
    classify = []

    # fW = open("./data/Sogou_data/test.txt", 'w', encoding='UTF-8')
    for item in reader:
        classify.append(int(item[0][0]))
    LDA_labels = getLabel("./data/Sogou_data/Sogou_news.txt")
    print(evaluate(LDA_labels, classify))


if __name__=="__main__":
    qinghua_split()
    print("holy shit!")