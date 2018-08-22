import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import numpy as np
import pandas as pd

data = open('data.csv','r',encoding='utf-8')#不要用rb模式，返回的不是字符,标注encoding='utf-8'不然会报错乱码
reader = csv.reader(data)
headers = next(reader)
#header转向第一行包括年龄，工作，房子，信用，分类
"""
利用sklearn 完成 决策树

#新建列表，用字典方式存储每个数据信息,为下一步转化为二进制格式做准备
featureList = []
labelList = []

for row in reader:
    rawdict = {}
    for i in range(1,len(row)-1): 
        rawdict[headers[i]] = row[i]
    featureList.append(rawdict)
    labelList.append(row[len(row)-1])

#python自带的数据预处理工具，把离散变量转为01
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
#print(vec.get_feature_names()) 类别
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

#采用ID3算法
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)

#可视化
with open ('decision.dot','w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),
                             out_file=f)
#terminal cd转到目录，然后dot -Tpdf filename.dot -o output.pdf
#然后就可以看到生成好了pdf

"""

#信息增益算法
def exp_entropy(dummyY1):#计算经验熵
    m,n = np.shape(dummyY1)#m行，n列
    HD = 0
    for k in range(0,n):
        Ck=np.sum(dummyY1[:,k])
        D = m
        if Ck==0:#规定 log2 (0) = 0
            HD=0
        else:
            HD = (-1)*Ck/D * np.log2(Ck/D) + HD
    return HD

def exp_condition_entropy(dummyY,dummyY2,HD):
    m,n = np.shape(dummyY)#根据特征分为n个子集
    Di = []#存储子集样本个数
    m1, n1 = np.shape(dummyY2)
    for i in range(0,n1-2):
        Di.append (np.sum(dummyY[:,i-1]))#Di=[5,5,5]#该特征的不同取值的个数

    dummy = []
    for i in range(0,n1-2):
        dummyY4=np.zeros([n1])
        for j in range(0,m1):
            if dummyY2[j][i] == 1 :
                dummyY4 = np.vstack((dummyY4,dummyY2[j]))
        dummy.append(dummyY4[1:])#dummy中存储 该特征不同类别的数组，分别计算H(Di)
    dummyA1 = 0
    for i in range(0,n1-2):
        dummyA1 = (Di[i]/m) * exp_entropy(dummy[i][:,(-1,-2)]) + dummyA1
    return (HD-dummyA1)

def getGDA(i,list,p):
    dummyY = np.array(pd.get_dummies(list[:,i]))#根据第i列特征生成矩阵
    dataprocess = []
    for j in range(0,p):#为了处理子集Di中属于类Ck的样本集合，把第i列特征和类别提取出来
        rawdict = {}
        for k in [i,5]:
            rawdict[headers[k]]=list[j][k]
        dataprocess.append(rawdict)#某一列特征生成好矩阵后，分别按照特征的不同
                                        # 存到dataprocess列表中，dataprocess[i]表示特征的第一个取值的子集Di

    vec = DictVectorizer()
    dummyY2 = vec.fit_transform(dataprocess).toarray()#将dataprocess矩阵转为二进制

    gDA = exp_condition_entropy(dummyY,dummyY2,HD)
    return gDA




#用python实现ID3算法
#第一步 计算信息增益

#输入 训练数据集D，特征A 进行数据预处理
list = []
for line in reader:#第一行
    list.append(line)#shape 14,6
list = np.array(list) #把reader中的文件转为list（15，6）

#计算经验熵
dummyY1 = np.array(pd.get_dummies(list[:,-1]))#把最后一列类别转为二进制格式 （15，2）
HD = exp_entropy(dummyY1)
print('HD=' + str(HD))

#计算HDA
p,q = np.shape(list)#（15，6）
for m in range(1,q-1):#从第二列到倒数第二列特征，分别得到其信息增益
    gda = getGDA(m,list,p)
    print('g(D,A'+str(m)+') = '+str(gda))