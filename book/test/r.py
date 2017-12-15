import numpy as np
from os.path import join
import tensorflow as tf
from PIL import Image

#TFRcord文件
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

#图片信息
NUM_CLASSES = 196
IMG_HEIGHT = 60
IMG_WIDTH =160
IMG_CHANNELS = 3
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS



def read_and_decode(filename_queue):
    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #从文件中读出一个样例
    _,serialized_example = reader.read(filename_queue)
    #解析读入的一个样例
    features = tf.parse_single_example(serialized_example,features={
        'label0':tf.FixedLenFeature([],tf.int64),
        'image':tf.FixedLenFeature([],tf.string)
        })
    #将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image'],tf.uint8)
    label = tf.cast(features['label0'],tf.int32)
    print(image.shape)
     
    image.set_shape([IMG_PIXELS])
    image = tf.reshape(image,[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image,label

if __name__ == '__main__':
    tfrecords_filename = "model/training.tfrecord"
    filename_queue = tf.train.string_input_producer([tfrecords_filename],) #读入流中
    read_and_decode(filename_queue)