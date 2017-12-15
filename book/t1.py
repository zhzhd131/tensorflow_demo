#!/usr/bin/env python
# encoding: utf-8
'''
book.t1 -- shortdesc

book.t1 is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2017 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import tensorflow as tf

import sys
import time


x_trian=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
y_trian=[5.5,8.5,11.5,14.5,17.5,20.5,23.5,26.5,29.5,32.5,35.5,38.5,41.5,44.5,47.5,50.5,53.5,56.5,59.5,62.5]
#i*3+5.5

Y = tf.placeholder(tf.float32,  name='Y')
X = tf.placeholder(tf.float32,  name='X')


w =tf.Variable(1,  name='weight',  dtype=tf.float32)
b =tf.Variable(1.1,  name='bais',  dtype=tf.float32)
print(' load处理耗时:%f '%((time.time())))

Y_ =w*X+b
 
T=tf.nn.relu(32)   

loss=tf.div(tf.reduce_sum(tf.pow((Y_-Y), 2)),x_trian.__len__(), name='loss')  
#tf.summary.histogram('loss', loss)
#tf.histogram_summary('loss', loss)

learning_rate=0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init =tf.global_variables_initializer()
summary = tf.summary.merge_all()

with tf.Session() as sess:
#选定可视化存储目录    
#合并到Summary中    
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
   
    for index in range(1000):
       for (x, y) in zip(x_trian, y_trian):
           #print(y)
           sess.run(optimizer,feed_dict={X: x, Y: y})
    
       if (index+1)%50 ==0:
            c = sess.run(loss, feed_dict={X: x, Y: y})
            ww=sess.run(w)
            bb=sess.run(b)
            print("index:%d，w：%f,b:%f   ,loss:%f"%(index,ww,bb,c))
            summary_str = sess.run(summary, feed_dict={X: x_trian, Y:y_trian})
   
           # print("After %d training step(s), index on training batch is %g." % (step, loss_value))
            
           # result = sess.run(merged,feed_dict={X: x, Y: y}) #merged也是需要run的    
           
           # writer.add_summary(ww,index)     
            
   # summary_writer.close()
          
    c = sess.run(loss, feed_dict={X: x_trian, Y:y_trian})
    print("w：%f,b:%f   ,loss:%f"%(sess.run(w),sess.run(b),c))

    print("Optimization Finished!")            