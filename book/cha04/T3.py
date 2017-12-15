'''
Created on 2017年10月30日

@author: zhangzd
'''
import tensorflow as  tf


from tensorflow.examples.tutorials.mnist import input_data
import tensorboard


#y=ax*x +bx+c 
#tensorboard 使用

import tensorboard as board




mnist=input_data.read_data_sets("MNIST_data",  one_hot=True)
batch_size=500
n_batch=mnist.train._num_examples//batch_size
print(n_batch)

with tf.name_scope("input"):
    x=tf.placeholder(tf.float32, [None,784],name="input_x")
    y=tf.placeholder(tf.float32, [None,10],name="input_y")

# keep_prob=tf.placeholder(tf.float32)
# lr =tf.Variable(0.1,tf.float32)

#W=tf.Variable(tf.zeros([784,10]))
with tf.name_scope("layer"):
    with tf.name_scope("layer_1"):
        W1=tf.Variable(tf.truncated_normal([784,500], stddev=0.1),name="weight")
        
        b1=tf.Variable(tf.zeros([500])+0.1,name="bais")
        L1 =tf.nn.tanh(tf.matmul(x,W1)+b1,name="tanh")
# L1_drop=tf.nn.dropout(L1, keep_prob)
    with tf.name_scope("layer_2"):
        W2=tf.Variable(tf.truncated_normal([500,10], stddev=0.1),name="weight")
        b2=tf.Variable(tf.zeros([10])+0.1,name="bais")
        L2 =tf.nn.tanh(tf.matmul(L1,W2)+b2,name="tanh")
# L2_drop=tf.nn.dropout(L2, keep_prob)
# 
# # W3=tf.Variable(tf.truncated_normal([200,100], stddev=0.1))
# # b3=tf.Variable(tf.zeros([100])+0.1)
# # L3 =tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
# # L3_drop=tf.nn.dropout(L3, keep_prob)
# 
# 
# W4=tf.Variable(tf.truncated_normal([300,10], stddev=0.1))
# b4=tf.Variable(tf.zeros([10])+0.1)
    prediction=tf.nn.softmax(L2,name="softmax")
with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.square(prediction-y))
#loss=-tf.reduce_mean(prediction*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#global_step=tf.Variable(0)
#learning_rate=tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
#optimizer =tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with tf.name_scope("optimizer"):
    optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)
init=tf.global_variables_initializer();

with tf.name_scope("accracy"):
    with tf.name_scope("correct_pre"):
        correct_pre=tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    with tf.name_scope("accracy"):    
        accracy=tf.reduce_mean(tf.cast(correct_pre,tf.float32))

with  tf.Session() as sess:
    sess.run(init)
    tf.summary.FileWriter("/logs/", sess.graph)
    for index in range(2):
        #sess.run(tf.assign(lr, 0.001*(0.95**index)))
        for batch in range(n_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
           
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
           #              print(sess.run(loss,feed_dict={x:batch_x,y:batch_y}))  
             
        #learning_ra=sess.run(lr)   
        test_acc=sess.run(accracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        #train_acc=sess.run(accracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:0.1}) 

        print(" index:"+str(index)+" test_acc:"+str(test_acc)+" learning_ra:")
# x=tf.Variable(3,name="name1")
# 
# #a=tf.Variable([3,4])
# new_value=tf.add(x, 3, name="add")
# update=tf.assign(x, new_value*10,  name="upadete")
# 
# b =3;
# c=4
# 
# #add =tf.add(x, a, name='sddd')
# init =tf.global_variables_initializer()
# with  tf.Session() as see:
#     see.run(init)
#     print(see.run(x))
#     print(see.run(update))
#     print(x)
#     for _ in  range(5):
#         print("X:%d,update:%d)"%(see.run(x),see.run(update)))
    #print(see.run(b+c))
    #print(see.run(b+c))
    #print(see.run(add))
# x_t = np.random.rand(100) #[3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                          #7.042,10.791,5.313,7.997,5.654,9.27,3.1]
# y_t = x_t*5+7
# cc=[1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]
# print(type(x_t) ) 
# print(type(cc) )                        
# a=tf.Variable(0.);
# b=tf.Variable(0.);
# y_=a*x_t+b
# 
# loss=tf.reduce_mean(tf.square(y_t-y_))
# 
# optimizer=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# #global_step, var_list, gate_gradients, aggregation_method, colocate_gradients_with_ops, name, grad_loss
# #train =optimizer.minimize(loss)
# 
# init=tf.global_variables_initializer()
# with tf.Session() as see:
#     see.run(init)
#     for step in range(100):
# #
#         see.run(optimizer)
#         print(see.run([a,b,loss]))
# ss=np.random.normal(0,0.2,10)
# print(ss)  
# x_data= np.linspace(-0.5, 0.5, 200)  
# noise=np.random.normal(0,0.2,x_data.shape)  
# y_data=np.squeeze(x_data)*3+noise   
# 
# x=tf.placeholder(tf.float16, [None,1])   
# y=tf.placeholder(tf.float16, [None,1]) 

        
  
