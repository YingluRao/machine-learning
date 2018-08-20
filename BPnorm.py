import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
#输入
xa = np.array([[2,1,3,3,1,1,0.697,0.460],[3,1,2,3,1,1,0.774,0.376],
             [3,1,3,3,1,1,0.634,0.264],[2,1,2,3,1,1,0.608,0.318],
             [1,1,3,3,1,1,0.556,0.215],[2,2,3,3,2,2,0.403,0.237],
             [3,2,3,2,2,2,0.481,0.149],[3,2,3,3,2,1,0.437,0.211],
             [3,2,2,2,2,1,0.666,0.091],
             [2,3,1,3,3,2,0.243,0.267],[1,3,1,1,3,1,0.245,0.057],
             [1,1,3,1,3,2,0.343,0.099],[2,2,3,2,1,1,0.639,0.161],
             [1,2,2,2,1,1,0.657,0.198],[3,2,3,3,2,2,0.360,0.370],
             [1,1,3,1,3,1,0.593,0.042],[2,1,2,2,2,1,0.719,0.103]])
ya = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

learninga = 0.1
#训练初始化
la = 2 #输出层
qa = 9 #隐藏层
da = 8 #输入层

wa = np.random.random((9,1))
va = np.random.random((9,8))

threshold1a = np.random.random((1,1))#阈值y
threshold2a = np.random.random((9,1))#阈值bh
#计算
def sigmoida(x):
    return 1.0 / (1.0 + np.exp(-x))

counta = 0
Eksa = []
errora = 1
errorsa = []
while counta<= 10000 or  errora <=0.001:
    yksa = []
    for ia in range(0,17):
        aa = np.dot(xa[ia],va.T)#矩阵乘法 隐层输入 9*8 8*1 9, 表示一维数组
        aa = np.array(aa).reshape(9,1)
        ba = sigmoida(aa-threshold2a)#隐层输出  9,1
        Ba = np.dot(wa.T,ba) #输出层输入
        yka = sigmoida(Ba-threshold1a)#输出 1,1
        yksa.append(yka)
        Eka = 0.5 * math.pow((yka-ya[ia]),2)
        Eksa.append(Eka)
        gja = yka*(1-yka)*(ya[ia]-yka)
        dthreshold1a = (-1)*learninga*gja
        xa[ia] = np.mat(xa[ia])
        ea = ba * (1-ba) *wa * gja
        dva =learninga * ea * xa[ia]
        dthreshold2a = (-1)*learninga*ea
        dwa = learninga * gja * ba

    #更新
        va = va + dva
        threshold1a = threshold1a +dthreshold1a
        wa = wa + dwa
        threshold2a = threshold2a + dthreshold2a

    errora = np.sum(Eksa)/len(Eksa)
    errorsa.append(errora)
    counta = counta + 1
    print(counta)
print(yksa)
xa = range(1,counta+1)
ya = errorsa
plt.subplot(2,1,1)
plt.scatter(xa,ya)
plt.title('norm error backpropagation ')
plt.xlabel('times')
plt.ylabel('errors')

"""BP accumulate"""

#输入
x = np.array([[2,1,3,3,1,1,0.697,0.460],[3,1,2,3,1,1,0.774,0.376],
             [3,1,3,3,1,1,0.634,0.264],[2,1,2,3,1,1,0.608,0.318],
             [1,1,3,3,1,1,0.556,0.215],[2,2,3,3,2,2,0.403,0.237],
             [3,2,3,2,2,2,0.481,0.149],[3,2,3,3,2,1,0.437,0.211],
             [3,2,2,2,2,1,0.666,0.091],
             [2,3,1,3,3,2,0.243,0.267],[1,3,1,1,3,1,0.245,0.057],
             [1,1,3,1,3,2,0.343,0.099],[2,2,3,2,1,1,0.639,0.161],
             [1,2,2,2,1,1,0.657,0.198],[3,2,3,3,2,2,0.360,0.370],
             [1,1,3,1,3,1,0.593,0.042],[2,1,2,2,2,1,0.719,0.103]])
y = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
y = y.reshape(17,1)
learning = 0.1
#训练初始化
l = 2 #输出层
q = 9 #隐藏层
d = 8 #输入层

w = np.random.random((9,1))
v = np.random.random((9,8))

threshold1 = np.random.random((1,1))#阈值y
threshold2 = np.random.random((1,9))#阈值bh
#计算
def sigmoid(x):
    m,n = np.shape(x)
    for i in range(0,m):
        for j in range(0,n):
            x[i][j] = 1/(1+math.exp(-x[i][j]))
    return x

count = 0
Eks = []
error = 1
errors = []
yk = []

while count<= 10000 and  error >0.001:

    a = np.dot(x,v.T)#矩阵乘法 隐层输入 17*8 8*9 17*9
    b = sigmoid(a-threshold2)#隐层输出  17*9
    B = np.dot(b,w) #输出层输入17*9 9*1 17*1
    yk = sigmoid(B-threshold1)#输出 17*1
    Ek = 0.5 * (yk-y) * (yk-y)
    gj = yk*(1-yk)*(y-yk) #17*1
    dthreshold1 = (-1)*learning*gj #17*1
    e = b * (1-b) * np.dot(gj,w.T) # 17*1 1*9 17*9
    dv =learning * np.dot(e.T,x)  # 9*17,17*8 9*8
    dthreshold2 = (-1)*learning*e #17*9
    dw = learning * np.dot(b.T,gj) #9*17,17*1 9*1

    v = v + dv# 9*8
    threshold1 = threshold1 +dthreshold1 #17*1
    w = w + dw #9*1
    threshold2 = threshold2 + dthreshold2#17*9

    error = np.sum(Ek)/len(Ek)
    errors.append(error)
    count = count + 1


print(count)
print(yk)
x = range(1,count+1)
plt.subplot(2,1,2)
plt.scatter(x,errors)
plt.xlabel('times')
plt.ylabel('errors')
plt.title('accumulated error backpropagation')


plt.show()