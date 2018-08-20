import numpy as np
import math
import matplotlib.pyplot as plt
import copy

def file2open(filename):#打开文件
    f = open(filename,'r',encoding='utf-8')
    lines = f.readlines()
    lines[0] = lines[0].lstrip('\ufeff')#去除前缀
    line_row = len(lines)#得到行数
    returnMat = np.zeros((line_row,3))#返回矩阵
    classlabel = []#记录标签

    index = 0
    for line in lines:
        line = line.strip()#去除空格
        listfromline = line.split('\t')#按照\t分割文件
        returnMat[index,:]=listfromline[0:3]
        if listfromline[-1]=='didntLike':
            classlabel.append('black')
        elif listfromline[-1]=='smallDoses':
            classlabel.append('orange')
        elif listfromline[-1]=='largeDoses':
            classlabel.append('red')
        index+=1
    return returnMat,classlabel,line_row,lines

def drawpicture(returnMat,line_row,classlabel,lines):
    plt.rcParams['font.sans-serif']=['simHei']
    plt.rcParams['axes.unicode_minus']=False
    #画第一幅图 飞行&游戏
    x_values = returnMat[:,0]
    y_values = returnMat[:,1]
    plt.subplot(2, 2, 1)
    plt.xlabel('飞行时间占比', fontsize=10)
    plt.ylabel('游戏时间占比', fontsize=10)
    for i in list(range(0,line_row)):
        plt.scatter(x_values[i],y_values[i],color=classlabel[i],s=20)

    #画第二幅图 飞行&美食
    x_values = returnMat[:, 0]
    y_values = returnMat[:, 2]
    plt.subplot(2, 2, 2)
    plt.xlabel('飞行时间占比', fontsize=10)
    plt.ylabel('美食时间占比', fontsize=10)
    for i in list(range(0, line_row)):
        plt.scatter(x_values[i], y_values[i], color=classlabel[i], s=20)
    #画第三幅图 游戏&美食
    x_values = returnMat[:, 1]
    y_values = returnMat[:, 2]
    plt.subplot(2, 2, 3)
    plt.xlabel('游戏时间占比', fontsize=10)
    plt.ylabel('美食时间占比', fontsize=10)
    for i in list(range(0, line_row)):
        plt.scatter(x_values[i], y_values[i], color=classlabel[i], s=20)


def normhanshu(Mat):#归一化
    newMat = np.zeros((line_row, 3))#写一个新矩阵，才能传回去
    for j in range(0,3):
        indexmax = np.argmax(Mat[:,j])
        indexmin = np.argmin(Mat[:,j])
        valuemax = Mat[indexmax][j]
        valuemin = Mat[indexmin][j]
        for i in range(0, line_row):
            newvalue = (returnMat[i, j] - valuemin) / (valuemax - valuemin)
            newMat[i][j] = newvalue
    return newMat

def distance(inputx,i,k):#对于某个点x的所有其他点与他的距离
    black = 0
    red = 0
    orange = 0
    curracy = 0
    L2list = []
    for j in range(0,1000):
        List = {}
        if j != i:
            L1 = abs((inputx[0]-Mat[j][0])**3+(inputx[1]-Mat[j][1])**3+(inputx[2]-Mat[j][2])**3)
            L2 = math.pow(L1,1.0/3)#得到点与x的距离
            List = {'distance':L2,'number':j}
            List = copy.deepcopy(List)
            L2list.append(List)
    L2list = sorted(L2list,key = lambda e:e.__getitem__('distance'))
    for list in L2list[:k]:#设定k，k中最多的那个颜色对比与测试点的颜色
        m = list['number']
        if classlabel[m] == 'black':
            black = black + 1
        if classlabel[m] == 'orange':
            orange = orange + 1
        if classlabel[m] == 'red':
                red = red + 1
    maxim = max(black,orange,red)
    if maxim == black:
        if classlabel[i]=='black':
            curracy = curracy + 1
    if maxim == orange:
        if classlabel[i] == 'orange':
            curracy = curracy + 1
    if maxim == red:
        if classlabel[i] == 'red':
            curracy = curracy + 1

    return curracy

def classvector(k):
    rate = 0
    for i in range(0,100):#用百分之十的数据测试
        curracy = distance(Mat[i],i,k)
        if curracy ==1:
            rate = rate + 1
    return rate

def classperson(inputx,k):#检测一个人
    black = 0
    red = 0
    orange = 0
    L2list = []

    for j in range(0,1000):
        List = {}
        L1 = abs((inputx[0]-Mat[j][0])**3+(inputx[1]-Mat[j][1])**3+(inputx[2]-Mat[j][2])**3)
        L2 = math.pow(L1,1.0/3)#得到点与x的距离
        List = {'distance':L2,'number':j}
        List = copy.deepcopy(List)
        L2list.append(List)
    L2list = sorted(L2list,key = lambda e:e.__getitem__('distance'))
    for list in L2list[:k]:#设定k=10
        m = list['number']
        if classlabel[m] == 'black':
            black = black + 1
        if classlabel[m] == 'orange':
            orange = orange + 1
        if classlabel[m] == 'red':
                red = red + 1
    maxim = max(black,orange,red)
    if maxim == black:
        print('you may dislike the person.')
    if maxim == orange:
        print('you may somelike the person.')
    if maxim == red:
        print('you may very like the person.')


if __name__ == '__main__':

    returnMat,classlabel,line_row,lines=file2open('datingTestSet.txt')
    Mat = normhanshu(returnMat)#归一化
    drawpicture(Mat, line_row, classlabel, lines)#画图

    value_max = np.argmax(returnMat[:,0])
    value_min = np.argmin(returnMat[:, 0])
    print(returnMat[value_max][0])
    print(returnMat[value_min][0])
    curracies = []#计算分类准确率，画图
    for k in range(1,50):
        rate = classvector(k)#设定k值
        curracy = rate/100.0
        curracies.append(curracy)
        #print('k = ' + str(k) + ' curracy = ' + str(curracy))
    ks = list(range(1,50))

    plt.subplot(2, 2, 4)
    plt.plot(ks, curracies, linewidth=5)
    plt.xlabel('k值', fontsize=10)
    plt.ylabel('分类准确率', fontsize=10)

    plt.show()
    #print('curracy = ' + str(curracy)+'%')


#测试

#输入数据
    distance1 = int(float(input('please input the flight distance:')))
    playtime = float(input('please input the time you play games'))
    icecream = float(input('please input the icecream you eat'))



    for j in range(0,3):#数据归一化
        indexmax = np.argmax(returnMat[:,j])
        indexmin = np.argmin(returnMat[:,j])
        newvaluemax = returnMat[indexmax]
        newvaluemin = returnMat[indexmin]


distance_norm = (distance1 - newvaluemin[0]) / (newvaluemax[0] - newvaluemin[0])
playtime_norm = (playtime - newvaluemin[0]) / (newvaluemax[0] - newvaluemin[0])
icecream_norm = (icecream - newvaluemin[0]) / (newvaluemax[0] - newvaluemin[0])
inputx = [distance_norm,playtime_norm,icecream_norm]


classperson(inputx,k=10)
