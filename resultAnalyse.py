from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics

def getLabel(file):
    res = []
    with open(file, 'r') as f:
        next(f)
        for line in f:
            l = line.split()
            res.append(int(l[2]))
    return res

def getTrueLabel(dic):
    res = []
    for key in dic.keys():
        res += [dic[key]] * 10
    return res



if __name__=="__main__":
    out = "D:\\eclipse-workspace\\test\\sample-data\\DataOut\\1\\test_out.txt"
    label = getLabel(out)

    # 这个字典是lda的结果排序与真实排序之间的映射，每次得到新的结果都需更新该字典！！！
    # k:v = 原始数据的分类：lda结果分类
    dic = {
        0: 0,
        1: 3,
        2: 1,
        3: 4,
        4: 2
    }
    trueLabel = getTrueLabel(dic)
    print()
    print("total acc:  ", metrics.accuracy_score(trueLabel, label))
    print("total f1:  ", metrics.f1_score(trueLabel, label, average='weighted') )
    print("total recall:  ", metrics.recall_score(trueLabel, label, average='micro'))
    print("total precision:  ", metrics.precision_score(trueLabel, label, average='macro'))
    print("------------------------------------------------------------")
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    print("分类报告：")
    print(classification_report(trueLabel, label, target_names=target_names))


