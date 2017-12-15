import os 
import tensorflow as tf 
from PIL import Image  

import numpy as np
# #路径
# cwd='/home/'
# #类别
# classes={'test2':1,
#          'test':2}
# #tfrecords格式文件名
# writer= tf.python_io.TFRecordWriter("mydata.tfrecords") 
# 
# for index,name in enumerate(classes):
#     class_path=cwd+name+'/'
#     for img_name in os.listdir(class_path): 
#         img_path=class_path+img_name #每一个图片的地址
# 
#         img=Image.open(img_path)
#         img_raw=img.tobytes()#将图片转化为二进制格式
#         example = tf.train.Example(features=tf.train.Features(feature={
#             #value=[index]决定了图片数据的类型label
#             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#             'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#         })) #example对象对label和image数据进行封装
#         writer.write(example.SerializeToString())  #序列化为字符串
# 
# writer.close()

VERIFY_CODES = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

def crate_exa(name,imagedata):
   
    
    labe_str=name.split(".")[0].upper()
   
    lable1=[0,0,0,0]
    for i in range(len(labe_str)):
        #lables=np.zeros(len(VERIFY_CODES),dtype="int8")
        lable1[i]=VERIFY_CODES.find(labe_str[i]);
       # lables[VERIFY_CODES.find(labe_str[i])]=1
    
    #aa=''.join(str(i) for i in lables)  
    img_tf=tf.train.Example(features=tf.train.Features(feature={
    #value=[index]决定了图片数据的类型label
    "label0": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable1[0]])),
    "label1": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable1[1]])),
    "label2": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable1[2]])),
    "label3": tf.train.Feature(int64_list=tf.train.Int64List(value=[lable1[3]])),
    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imagedata]))
    })) #example对象对label和image数据进行封装  
    return img_tf;



    

def create_tf():
    in_path="cap/"
    out_path="" 
    index2 =0
    writer= tf.python_io.TFRecordWriter("mydata.tfrecords--"+str(index2)) 
    index =1
    for img_name in os.listdir(in_path): 
        print(img_name)
        img_path=in_path+img_name #每一个图片的地址
        img_raw=Image.open(img_path).tobytes()
        img_tf=crate_exa(img_name,img_raw)
        writer.write(img_tf.SerializeToString())  #序列化为字符串 
        index=index+1 
        if index%200==0 :
            writer.close()
            writer= tf.python_io.TFRecordWriter("mydata.tfrecords--"+str(index2+1)) 
            index2=index2+1
        
     

def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
#     img = tf.reshape(img, [64, 64, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label

#create_tf()        
'''
swd = 'F:\\testdata\\show\\'
filename_queue = tf.train.string_input_producer(["mydata.tfrecords"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
#tf.decode_raw可以将字符串解析成图像对应的像素数组
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [36,136,3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    #启动多线程
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(28):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save(swd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)
'''