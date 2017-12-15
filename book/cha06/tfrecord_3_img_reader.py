'''
Created on 2017年12月12日

@author: zhangzd
'''
# -*- coding:utf-8 -*- 

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import  book.cha06.cteate_image as im

# '''read data
# 从 tfrecord 文件中读取数据，对应数据的格式为png / jpg 等图片数据。
# '''

#if __name__=='__main__':
#     tfrecords_filename = "model/training.tfrecord"
#     filename_queue = tf.train.string_input_producer([tfrecords_filename],) #读入流中
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label0': tf.FixedLenFeature([], tf.int64),
#                                            'image' : tf.FixedLenFeature([], tf.string),
#                                        })  #取出包含image和label的feature对象
# 
#     image = tf.decode_raw(features['image'],tf.int64)
#     image = tf.reshape(image, [160, 60, 3])
#     label = tf.cast(features['label0'], tf.int64)
#     
#     with tf.Session() as sess: #开始一个会话
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
# #         coord=tf.train.Coordinator()
# #         threads= tf.train.start_queue_runners(coord=coord)
#         for i in range(20):
#             print(i)
#             example, l = sess.run([image,label])#在会话中取出image和label
#             print(l)
#             img=Image.fromarray(example)#这里Image是之前提到的
#             img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
#             print(example, l)
#   
# #         coord.request_stop()
# #         coord.join(threads)

def decodeimage():
    filename_queue = tf.train.string_input_producer(['model/training.tfrecord'], num_epochs=None, shuffle=True)
    # **2.创建一个读取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # **3.根据你写入的格式对应说明读取的格式
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       }
                                       )
    img = features['image']
    # 这里需要对图片进行解码
    img = tf.image.decode_png(img, channels=3)  # 这里，也可以解码为 1 通道
    img = tf.reshape(img, [160, 60, 3])  # 28*28*3
       
 
    return img,features['label0'],features['label1'],features['label2'],features['label3']

if __name__=='__main__':  
    # **1.把所有的 tfrecord 文件名列表写入队列中

       
    img,lb0,lb1,lb2,lb3=decodeimage()  
      
    
    # **4.通过 tf.train.shuffle_batch 或者 tf.train.batch 函数读取数据
    """
    这里，你会发现每次取出来的数据都是一个类别的，除非你把 capacity 和 min_after_dequeue 设得很大，如
    X_batch, y_batch = tf.train.shuffle_batch([img, label], batch_size=100,
                                              capacity=20000, min_after_dequeue=10000, num_threads=3)
    这是因为在打包的时候都是一个类别一个类别的顺序打包的，所以每次填数据都是按照那个顺序填充进来。
    只有当我们把队列容量舍得非常大，这样在队列中才会混杂各个类别的数据。但是这样非常不好，因为这样的话，
    读取速度就会非常慢。所以解决方法是：
    1.在写入数据的时候先进行数据 shuffle。
    2.多存几个 tfrecord 文件，比如 64 个。
    """
#      X_batch, y_batch = tf.train.shuffle_batch([img, label], batch_size=2,
#                                               capacity=20, min_after_dequeue=3, num_threads=1)
    img_batch, lb0_batch,lb1_batch,lb2_batch,lb3_batch = tf.train.shuffle_batch([img,lb0,lb1,lb2,lb3], batch_size=3,
                                              capacity=20, min_after_dequeue=3, num_threads=1)   
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
       
    # **5.启动队列进行数据读取
    # 下面的 coord 是个线程协调器，把启动队列的时候加上线程协调器。
    # 这样，在数据读取完毕以后，调用协调器把线程全部都关了。
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    y_outputs = list()
    for i in range(2):
        _img_batch, _lb0_batch,_lb1_batch,_lb2_batch,_lb3_batch = sess.run([img_batch, lb0_batch,lb1_batch,lb2_batch,lb3_batch])
        print(_img_batch[0].reshape(-1).shape)
        print(np.mean(_img_batch[0],-1 ).shape)
        print("len:%d"%(len(_img_batch)))
        for k in range(len(_img_batch)):
            print(im.decode_getname(_lb0_batch[k],_lb1_batch[k],_lb2_batch[k],_lb3_batch[k]),  end=' ')
        img=Image.frombytes("RGB",(160,60),_img_batch[i].tostring())#函数参数中宽度高度要注意。构建24×16的图片
#         img.show()
    #     img.save("model/reverse_%d.bmp"%(i))#保存部分图片查看
          
       
    # **6.最后记得把队列关掉
    coord.request_stop()
    coord.join(threads)