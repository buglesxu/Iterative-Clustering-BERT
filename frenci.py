import jieba
import os

def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

def stopwordslist(path):

    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

fileName = []
f = "5游戏"
listdir("./DataZh/DataTest/" + f, fileName)

outFile = "./DataZh/test/" + f

ff = 1

for name in fileName:

    fR = open(name, 'r', encoding='UTF-8')

    sent = fR.read().split()
    sent = "".join(sent)
    # for str in sent:
    #     str.replace(u'\u3000', u'')

    sent_list = jieba.cut(sent)

    stopwords = stopwordslist("./DataZh/cn_stopwords.txt")

    outstr = ''
    for word in sent_list:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "

    fW = open(outFile + "/" + str(ff) + ".txt", 'w', encoding='UTF-8')
    fW.write(''.join(outstr))
    ff += 1

    fR.close()
    fW.close()
