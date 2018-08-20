import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义输入
x = tf.placeholder(tf.float32,shape=[None,784])#784像素点决定一张图 图片大小28*28
y = tf.placeholder(tf.float32,shape=[None,10])#决定是哪一个数字
keep_prob = tf.placeholder('float')
def conv2d(input,filter):
    return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')#stride[0]=stride[3]==0
def maxpool(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #ksize表示pool窗口2*2，strides表示height,width步长都为2
def w_initial(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)#返回一个变量
def b_initial(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#第一层卷积层 输出大小[-1,28,28,32]
w_conv1 = w_initial([5,5,1,32])#filet_height,filter_width,inchannels,outchannels
b_conv1 = b_initial([32])#初始化为输出大小
x_image = tf.reshape(x,[-1,28,28,1])#-1表示自动推测这个维度的size,inheight,inwidth,inchannels
out1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

print(np.shape(out1))
#第一层池化层 输出大小为[-1,14,14,32]
out_pool1 = maxpool(out1)
print(np.shape(out_pool1))

#第二层卷积层 输出大小为[-1,14,14,64]
w_conv2 = w_initial([5,5,32,64])
b_conv2 = b_initial([64])
out2 = tf.nn.relu(conv2d(out_pool1,w_conv2)+b_conv2)
print(np.shape(out2))
#第二层池化层 输出大小为[-1，7，7，64]
out_pool2 = maxpool(out2)
print(np.shape(out_pool2))

#进入全连接层[-1,400]
out_pool2_flat = tf.reshape(out_pool2,[-1,7*7*64])
print(out_pool2_flat)

#输出第一层全连接层[-1,1024]
w_connect1 = w_initial([7*7*64,1024])
b_connect1 = b_initial([1024])
connect_out1 = tf.nn.relu(tf.matmul(out_pool2_flat,w_connect1)+b_connect1)
#考虑过拟合
out_drop = tf.nn.dropout(connect_out1,keep_prob)

#输出层softmax
w_out = w_initial([1024,10])
b_out = b_initial([10])
out = tf.nn.softmax(tf.matmul(out_drop,w_out)+b_out)
print(np.shape(out))

#损失函数 交叉熵
cross_entropy = -tf.reduce_sum(y*tf.log(out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_predict = tf.equal(tf.argmax(out,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))

#建立会话，初始化

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100) #给None赋值100，每次随机取100张图片
        sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
        if i % 50 == 0 :
            print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.5}))


