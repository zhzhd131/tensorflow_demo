import os 
import tensorflow as tf 
from PIL import Image  

import numpy as np
import  book.cha06.cteate_image as im
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

in_path="train/"
out_path="model/" 

def crate_exa(name,imagedata):
   
    
    labe_str=name.split(".")[0].upper()
   
    lable1=[0,0,0,0]
    for i in range(len(labe_str)):
        #lables=np.zeros(len(VERIFY_CODES),dtype="int8")
        lable1[i]=im.number.index(labe_str[i]);
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

    index2 =0
    writer= tf.python_io.TFRecordWriter(out_path+"cap1.tfrecords"+str(index2)) 
    index =1
    for img_name in os.listdir(in_path): 
        print(img_name)
        img_path=in_path+img_name #每一个图片的地址
        img_raw=Image.open(img_path)
        img_raw.resize((224,224))
        np.array(img_raw.convert('L'))
        img_raw=img_raw.tobytes()
        img_tf=crate_exa(img_name,img_raw)
        writer.write(img_tf.SerializeToString())  #序列化为字符串 
        index=index+1 
#         if index%1000==0 :
#             writer.close()
#             print('-----------------')
#             writer= tf.python_io.TFRecordWriter("mydata.tfrecords--"+str(index2+1)) 
#             index2=index2+1
        
     
    writer.close()
#     writer= tf.python_io.TFRecordWriter("mydata.tfrecords--"+str(index2+1)) 



#create_tf()  
def read_tfFile(path):
   
   
    tfrecords_filename = path
    #test_write_to_tfrecords(tfrecords_filename)
    filename_queue = tf.train.string_input_producer([tfrecords_filename]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'],tf.int64)
    #image = tf.reshape(image, [160,60])
    label = tf.cast(features['label0'], tf.int64)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            example, l = sess.run([image,label])#在会话中取出image和label
            print(example.shape)
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            Image._show(img) 
            img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            print(example, l)

        coord.request_stop()
        coord.join(threads)


read_tfFile(out_path+"cap1.tfrecords0")
    
#     _,tf_example=tf.TFRecordReader().read(tf.train.string_input_producer([path]))
#     featrues=tf.parse_single_example(tf_example, features={
#             "img_raw": tf.FixedLenFeature([],tf.string),
#             "label0": tf.FixedLenFeature([],tf.int64),
#             "label1": tf.FixedLenFeature([],tf.int64),
#             "label2": tf.FixedLenFeature([],tf.int64),
#             "label3": tf.FixedLenFeature([],tf.int64),
#             
#         })
#     image_raw=tf.reshape( tf.decode_raw(featrues["img_raw"],tf.uint8),[160,60])
# #     image= tf.reshape(image_raw, [244,244])
#    
#     image=tf.cast(image,tf.float32)/255.0
#     image=tf.subscribe(image,0.5)
#     image=tf.multiply(image, 2.0)
#     
#     label0=tf.cast(featrues["label0"], tf.int32)
#     label1=tf.cast(featrues["label1"], tf.int32)
#     label2=tf.cast(featrues["label2"], tf.int32)
#     label3=tf.cast(featrues["label3"], tf.int32)
#     
#     return image,label0,label1,label2,label3;  
# 
# 
# image,label0,label1,label2,label3=read_tfFile(out_path+"cap1.tfrecords")
# Image._show(image) 
#    
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