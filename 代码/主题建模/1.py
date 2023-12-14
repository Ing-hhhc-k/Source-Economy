import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import codecs
import os
import re


def chinese_word_cut(mytext, wordFlag="n,v,a,x"):
    jieba.load_userdict('prepare/用户自定义词典.txt');  # 加入自定义语料库
    words = pseg.cut(mytext)
    line_List = []
    flag_List = wordFlag.split(",")
    for w in words:
        i = 0;
        for flag in flag_List:  # 加入自定义语料库
            if (w.flag == flag_List[i] and not w.word.isnumeric() and w.word not in stopwordeu and len(
                    re.findall('\s', w.word)) == 0):
                line_List.append(w.word)
            i = i + 1
    line = ""
    for word in line_List:
        line += word + " "
    return line


def prepare_data(content_list, attr1, attrmess, stopwords):
    # 将数据格式进行转换[[数值，内容],[数值，内容],[数值，内容],[数值，内容]]
    df = pd.DataFrame(content_list, columns=[attr1, attrmess])
    # 将数据格式进行转换
    df["內容切分"] = df[attrmess].astype(str).apply(chinese_word_cut)
    n_features = 3000000
    tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                    max_features=n_features,
                                    stop_words=stopwords,
                                    max_df=0.5,
                                    min_df=20)
    tf = tf_vectorizer.fit_transform(df['內容切分'])
    return df, tf, tf_vectorizer


def print_top_words2(model, feature_names, n_top_words):
    t = 0;
    for topic_idx, topic in enumerate(model.components_):
        topicSum = np.sum(topic, axis=0)
        topic = topic / topicSum
        # topic 中存放的主题-词汇矩阵，格式是 ndarray
        print("Topic #%d:" % topic_idx)
        # tw_list.append("{name:Topic"+str(topic_idx)+","+"value:1,symbolSize:15
        # ,category:Topic"+str(topic_idx)+",draggable:false}");
        # argsort 返回从原始数值中，值小到大的索引值
        print(" | ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            t = t + 1
        print(" | ".join([str(topic[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]))


def load_stopword(path):
    stopwords = open(path, 'r', encoding='utf-8').read()
    stopwords = stopwords.split('\n')
    return stopwords


def train_one_step(n_components, tf, tf_vectorizer, n_top_words):
    lda = LatentDirichletAllocation(n_components, max_iter=200,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    print(' n_components:%s' % n_components)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words2(lda, tf_feature_names, n_top_words)
    doc_word = tf.toarray()
    doc_topic = lda.transform(tf)
    topic_word = lda.components_
    # 计算困惑度
    print(u'困惑度：')
    print(lda.perplexity(tf, sub_sampling=False))
    # data = pyLDAvis.sklearn.prepare(lda,tf,tf_vectorizer)
    # pyLDAvis.show(data)
    return doc_word, doc_topic, topic_word


import math


def savetopic_wordMat(topic_word, savePath):
    tw_list = topic_word.tolist()
    fw = open(savePath, 'w')
    topicNum = 0;
    for tw_line in tw_list:
        z = 0;
        lineSum = sum(tw_line);
        shangzhi = 0;
        for j in tw_line:
            wordProbability = j / lineSum;
            if (z < len(tw_line) - 1):
                fw.write(str(wordProbability) + " ")
            else:
                fw.write(str(wordProbability) + "\n")
            z = z + 1
            temp = wordProbability * math.log(wordProbability)
            shangzhi = shangzhi + temp;
        print("Topic:" + str(topicNum) + "的信息熵是：" + str(-shangzhi))
        topicNum = topicNum + 1


def getMaxTopicFlag(doc_topic):
    new_dt = []
    [rows, cols] = doc_topic.shape
    for i in range(rows):
        maxflag = doc_topic[i].argsort()[-1]
        listTemp = doc_topic[i].tolist()
        listTemp.append(maxflag)
        new_dt.append(listTemp)
    dtend = np.array(new_dt)
    return dtend


def saveDocTopicMat(attr1, attrmess, savePath, dtendMatrix):
    cols = ['Topic %s' % i for i in range(best_tops + 1)]
    doc_topic = pd.DataFrame(dtendMatrix, columns=cols)
    new_df = pd.concat([df[[attr1, attrmess]], doc_topic], axis=1)
    print(new_df)
    new_df.to_excel(savePath, index=False)
    print(len(new_df))


def readCroups4FilePath(path):
    end_list = []
    df = pd.read_csv(path)
    replacetext = [  # "本报[\S]+电",
        # "（本报[\S]+电）",
        "新华社[\S]+日电",
        "人民网[\S]+日电",
        "（(.*?)版权(.*?)）",
        "服务邮箱：[\s\S]+",
        "互联网新闻信息服务许可证(.*?)+",
        "分享让更多人看到(.*?)+",
        "《 人民日报 》（[\s\S]+",
        "学习路上 时习之[\s\S]+",
        "\r\n\|\r\n[\s\S]+",
        "（(.*?)记者(.*?)）"]
    t = 0
    for index, row in df.iterrows():
        list1 = []
        t += 1
        for replace in replacetext:
            row[4] = re.sub(replace, "", str(row[4]))
        list1.append(t)
        list1.append(row[4])
        end_list.append(list1)
    return end_list

if __name__ == '__main__':
     #文件名称
     # fileName="law"
     attr1='标题'
     attrmess='内容'
     stopwordeu=load_stopword("prepare/stopwords.txt")
     #读取文件夹
     end_list=readCroups4FilePath("data/中国能源网全部.csv");
     print("文件数：" , len(end_list))
     #最优主题个数
     #组织语料
     df, tf, tf_vectorizer=prepare_data(end_list,attr1,attrmess,stopwordeu)
     # #训练
     best_tops=21
     doc_word,doc_topic, topic_word=train_one_step(best_tops,tf,tf_vectorizer,n_top_words=20)
     dtendMatrix=getMaxTopicFlag(doc_topic)
     savetopic_wordMat(topic_word,'result/'+'doc_topic'+str(best_tops)+'temptw.txt')
     saveDocTopicMat(attr1,attrmess,'result/'+'doc_topic'+str(best_tops)+'tempdt.xlsx',dtendMatrix)