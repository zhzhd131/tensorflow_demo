'''
Created on 2017年9月27日

@author: zhangzd
'''


import tensorflow as tf


import matplotlib.pyplot as plt
import numpy as np


data = []
label = []
np.random.seed(0)

# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(500):
    x1 = np.random.uniform(-1,1)
    x2 = np.random.uniform(0,2)
    if x1**2 + x2**2 <= 1:
        data.append([np.random.normal(x1, 0.1),np.random.normal(x2,0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)
        
data = np.hstack(data).reshape(-1,2)
label = np.hstack(label).reshape(-1, 1)
plt.scatter(data[:,0], data[:,1], c=label,
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
#plt.show()

#2. 定义一个获取权重，并自动加入正则项到损失的函数。
def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


#3. 定义神经网络
x = tf.placeholder(tf.float32, shape=(None, 2),name='x')
y_ = tf.placeholder(tf.float32, shape=(None, 1),name='y')
sample_size = len(data)

# 每层节点的个数
layer_dimension = [2,10,5,3,1]

n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0] #2

# 循环生成网络结构
for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.003)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias,name='layer_')
    in_dimension = layer_dimension[i]

y= cur_layer

# 损失函数的定义。
mse_loss = tf.div(tf.reduce_sum(tf.pow(y_ - y, 2)), sample_size, name='mse_loss') 
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))


#4. 训练不带正则项的损失函数mse_loss。
# 定义训练的目标函数mse_loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss)
TRAINING_STEPS = 40000

with tf.Session() as sess:
    #print(sess.run(get_weight,feed_dict={shape: [2, 3], lambda1: 0.003}))
    
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all() #将图形、训练过程等数据合并在一起
    writer = tf.summary.FileWriter('logs/',sess.graph) #将训练日志写入到logs文件夹下
    
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label})
        if i % 2000 == 0:
            los=sess.run(mse_loss, feed_dict={x: data, y_: label})
           # writer.add_summary(los,i) #将日志数据写入文件
            print("After %d steps, mse_loss: %f" % (i,los))

    # 画出训练后的分割曲线       
    xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)

plt.scatter(data[:,0], data[:,1], c=label,
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()
'''


TRAINING_STEPS = 100
global_step = tf.Variable(0)
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 10 == 0:
            print(sess.run(LEARNING_RATE))
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            
            x_value = sess.run(x)
            print ("After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value))       
       
'''