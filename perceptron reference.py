import numpy as np

class Perceptron():
    def __init__(self,dataSet,labels):#初始化数据集和标签
        self.dataSet =np.array(dataSet)
        self.labels = np.array(labels).transpose()#转置

    def sign(self,y):
        if y>0:
            return 1
        else:
            return -1

    def train(self):
        m,n = np.shape(self.dataSet)#读取行列
        weights = np.zeros(n)#w 初始化
        bias = 0 #b初始化
        flag = False
        while flag!= True:
            flag = True
            for i in range(m):
                y = weights * np.mat(self.dataSet[i]).T+bias
                if self.sign(y)*self.labels[i]<=0:
                    weights += self.labels[i]*self.dataSet[i]
                    bias += self.labels[i]
                    print('weight%s' % weights)
                    print('bias%s'%bias)
                    flag = False
        return weights,bias


if __name__ == '__main__':
        dataSet = [
            [3,3],
            [4,3],
            [1,1],
        ]

labels = [1,1,-1]
perceptron = Perceptron(dataSet,labels)
weights,bias = perceptron.train()
print('结果是：%s,%s'%(weights,bias))

#注意空格对齐的问题

print(list(range(0,3)))
