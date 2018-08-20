import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.random_normal([1,out_size]) + 0.1)
    Wx_b = tf.matmul(inputs,Weight) + biases
    if activation_function is None:
        outputs = Wx_b
    else:
        outputs = activation_function(Wx_b)
    return outputs


x_data = np.linspace(-1,1,200).reshape([200,1])#设置x数据（200，1）
noise = np.random.normal(0,0.05,x_data.shape)#设置噪音
y_data = np.square(x_data) - 0.5 + noise

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data,s=40)


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1)
print(np.shape(prediction))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),axis = 1))#注意沿轴相加
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])#擦除
            except Exception:
                pass
            lines = plt.plot(x_data,sess.run(prediction,feed_dict={xs:x_data}),'r-',linewidth = 5)
            plt.pause(0.5)#暂停显示
plt.show()
