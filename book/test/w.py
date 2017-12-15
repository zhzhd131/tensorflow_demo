'''
Created on 2017年12月12日

@author: zhangzd
'''
# -*- coding = utf-8 -*-

from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import  walk
from os.path import join

#图片存放位置
DATA_DIR = 'train2/'

#图片信息
IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3
NUM_TRAIN = 2
NUM_VALIDARION = 2

#读取图片
def read_images(path):
    filenames = next(walk(path))[2]
    num_files = len(filenames)
    images = np.zeros((num_files,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = np.zeros((num_files, ), dtype=np.uint8)
    f = open('label.txt')
    lines = f.readlines()
    #遍历所有的图片和label，将图片resize到[227,227,3]
    for i,filename in enumerate(filenames):
        img = imread(join(path,filename))
        img = imresize(img,(IMG_HEIGHT,IMG_WIDTH))
        images[i] = img
        labels[i] = int(lines[i])
    f.close()
    return images,labels

#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(images,labels,name):
    #获取要转换为TFRecord文件的图片数目
    num = images.shape[0]
    #输出TFRecord文件的文件名
    filename = name+'.tfrecords'
    print('Writting',filename)
    #创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        #将图像矩阵转化为一个字符串
        img_raw = images[i].tostring()
        #将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[i])),
            'image_raw': _bytes_feature(img_raw)}))
        #将example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')

def main(argv):
    print('reading images begin')
    start_time = time.time()
    train_images,train_labels = read_images(DATA_DIR)
    print(train_images.shape)
    print(train_labels)
    

    #get validation
#     validation_images = train_images[:NUM_VALIDARION,:,:,:]
#     validation_labels = train_labels[:NUM_VALIDARION]
#     train_images = train_images[NUM_VALIDARION:,:,:,:]
#     train_labels = train_labels[NUM_VALIDARION:]

    #convert to tfrecords
    print('convert to tfrecords begin')
    start_time = time.time()
    convert(train_images,train_labels,'train')
#     convert(validation_images,validation_labels,'validation')
    duration = time.time() - start_time
    print('convert to tfrecords end , cost %d sec' %duration)

if __name__ == '__main__':
    tf.app.run()