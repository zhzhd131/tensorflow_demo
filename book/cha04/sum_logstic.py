'''
Created on 2017年10月27日

@author: zhangzd
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]),name="w")
    add_summaries(Weights, "Weights")
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name="b")
    add_summaries(biases, "biases")
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs





x_data = np.linspace(-10,10,300)[:, np.newaxis]

noise = np.random.normal(0, 0.1, x_data.shape)
y_data = 1.5*np.sin(x_data) +x_data- 1.5 
y_data[30]=-3
y_data[10]=17


# y_data2 = 4*np.square(x_data)+4*x_data - 7 
#plt.plot(x_data, y_data,'.-')
# plt.plot(x_data, y_data2)

f1 = plt.figure(1)  
plt.subplot(111)  
plt.scatter(x_data,y_data,s=0.1) 

#加载保存 模型，
# saver = tf.train.import_meta_graph("save/model.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess, "save/model.ckpt")
#     sess.run(tf.get_default_graph().get_tensor_by_name("v1:0")
#plt.show()

# 2.定义节点准备接收数据
# define placeholder for inputs to network  
with tf.name_scope("inpute"):
    xs = tf.placeholder(tf.float32, [None, 1],name="inpute-x")
    ys = tf.placeholder(tf.float32, [None, 1],name="inpute-x")


with tf.name_scope("layer"):
    l1=add_layer(xs, 1, 101, tf.nn.relu)
    
   # l2=add_layer(l1, 200, 101, tf.nn.relu)
   # output2=add_layer(l1, 101, 1, tf.nn.relu)
    #输出层（1个神经元）
    W2 = tf.Variable(tf.random_normal([101,1]))
    
    b2 = tf.Variable(tf.zeros([1,1])+0.1)
    Wx_plus_b2 = tf.matmul(l1,W2) + b2
    output2 = Wx_plus_b2

# 4.定义 loss 表达式
# the error between prediciton and real data    
#损失
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-output2),reduction_indices=[1])) #在第一维上，偏差平方后求和，再求平均值，来计算损失
    tf.summary.scalar("loss", loss)
# 5.选择 optimizer 使 loss 达到最小                   

# 这一行定义了用什么方式去减少 loss，学习率是 0.1 
with tf.name_scope("train"):      
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

# with tf.name_scope("accracy"):     
#     acc =tf.reduce_mean()

meger=tf.summary.merge_all()
init=tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session()  as  sess:
    sess.run(init)
    writer=tf.summary.FileWriter("logs/", sess.graph)
    for i in range(500): #训练1000次
        summary, _,loss_value = sess.run([meger,train_step,loss],feed_dict={xs:x_data,ys:y_data}) #进行梯度下降运算，并计算每一步的损失
        writer.add_summary(summary, i)
        if(i%50==0):
            print(loss_value) # 每50步输出一次损失
            

    saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
    plt.plot(x_data, sess.run(output2,feed_dict={xs:x_data}))
    plt.show()
# 
# v = tf.Variable(0, dtype=tf.float32, name="v")
# for variables in tf.global_variables(): 
#     print("___"+variables.name)
#     
# ema = tf.train.ExponentialMovingAverage(0.99)
# 
# maintain_averages_op = ema.apply(tf.global_variables())
# for variables in tf.global_variables(): 
#     print(variables.name)
#     
# 
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     
#     sess.run(tf.assign(v, 10))
#     sess.run(maintain_averages_op)
#     # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
#     saver.save(sess, "Saved_model/model2.ckpt")
#     print(sess.run([v, ema.average(v)]))
# 
# v = tf.Variable(0, dtype=tf.float32, name="v2")
# 
# # 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
# with tf.Session() as sess:
#     saver.restore(sess, "Saved_model/model2.ckpt")
#     print("----")
#     print(sess.run(v))