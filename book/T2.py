'''
Created on 2017年9月30日

@author: zhangzd
'''
import tensorflow as tf

#2. 定义一个获取权重，并自动加入正则项到损失的函数。
def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var

weights = tf.constant([[1.0, 2.], [-3., 4.]])
init_opt=tf.initialize_all_variables()
server2 =tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_opt)
    sess.run(tf.contrib.layers.l1_regularizer(.5)(weights))
    
            # (1+2+3+4)*.5 ⇒ 5
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
    server2.save(sess, 'fifi.cpt')
            # (1+4+9+16)*.5*.5 ⇒ 7.5
    
