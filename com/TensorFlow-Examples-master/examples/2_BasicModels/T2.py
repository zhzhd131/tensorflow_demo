'''
Created on 2017年9月4日

@author: zhangzd
'''
import numpy

import  tensorflow as  tf
import matplotlib.pyplot as plt

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),  name="weights")
print(weights.shape)


train_X = numpy.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
train_Y = numpy.asarray([1.5,4.0,6.5,9.0,11.5,14.0,16.5,19.0,21.5,24.0,26.5,29.0,31.5,34.0,36.5,39.0,41.5,44.0,46.5,49.2])
n_samples = train_X.shape[0]

X = tf.placeholder("float", name='X')
Y = tf.placeholder("float", name='Y')



W = tf.Variable(tf.zeros([1,1],  name="WEIGHT"))
B = tf.Variable(tf.zeros([1], name="BIAS"))
rng = numpy.random
W = tf.Variable(rng.randn(), name="weight")
B = tf.Variable(rng.randn(), name="bias")

# pred=tf.multiply(X, W, "MUl")+B
pred = tf.multiply(X, W)+B

cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2*n_samples)
learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate, True, name)


init = tf.global_variables_initializer()


training_epochs = 1000
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
           # print("x ,   y :", x, y, '\n')
            
            sess.run(optimizer, feed_dict={X: x, Y: y})
      
    
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(B), '\n')
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(B), label='Fitted line')
    plt.legend()
    plt.show()
#Training cost= 0.0813284 W= [[ 0.23506787]] b= [ 0.81658989] 