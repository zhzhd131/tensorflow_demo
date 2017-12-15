'''
Created on 2017年12月14日

@author: zhangzd
'''
import tensorflow as tf

import numpy as np

import glob

import random

from PIL import Image

from skimage import io

from skimage import transform,data

from PNReader import *

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',

'V', 'W', 'X', 'Y', 'Z']

hanzi = ['京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑', '湘', '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂', '甘', '晋', '蒙', '陕',

'吉', '闽', '贵', '粤', '青', '藏', '川', '宁', '琼']

dicthz = {'京': 462, '津': 463, '沪': 464, '渝': 465, '冀': 466, '豫': 467, '云': 468, '辽': 469, '黑': 470, '湘': 471, '皖': 472, '鲁': 473, '新': 474, '苏': 475, '浙': 476, '赣': 477, '鄂': 478, '桂': 479, '甘': 480, '晋': 481, '蒙': 482, '陕': 483, '吉': 484, '闽': 485, '贵': 486, '粤': 487, '青': 488, '藏': 489, '川': 490, '宁': 491, '琼': 492}

# 文本转向量

char_set = number + ALPHABET + hanzi + ['_'] + ['_'] + ['_'] + ['_'] # 如果验证码长度小于4, '_'用来补齐

# char_set = number + ALPHABET + ['_'] + ['_'] # 如果验证码长度小于4, '_'用来补齐

CHAR_SET_LEN = len(char_set)

# CHAR_SET_LEN = 7

image_path = "/../images/"

test_path = "/../images/"

testlabel_path = "/../labels.txt"

label_path = "/../labels.txt"

PATH = "/../images/*.jpg"

# 图像大小

IMAGE_HEIGHT = 40 # 45

IMAGE_WIDTH = 100 # 180

MAX_NP = 7

# CHAR_SET_LEN = 7

image_file = "/Users/adminjackfy/downloads/trainlibs/chepai/images/藏GPEI07.jpg"

def getPicture(path):

    return glob.glob(path)

def getSplitData(path):

    result = getPicture(path)
    
    length = len(result)
    
    trainLengh=int(length*0.9)
    
    train = result[0:trainLengh]
    
    test =result[trainLengh:length-1]
    
    #train = result[0:int(length * 0.8)]
    
    #test = [i for i in result if i not in train]
    
    return train, test

def sampleTrain(length,trainData):

    return random.sample(trainData,length)

def getImageAndName(path):

    name = path.split("/")[-1].split(".")[0]
    
    # captcha_image = Image.open(path)
    
    # captcha_image = np.array(captcha_image)
    
    # with tf.name_scope('read_image'):
    
    img = 1.0 - io.imread(path, as_grey=True)
    
    img = transform.resize(img,(40, 100))
    
    # io.imsave('/Users/adminjackfy/downloads/trainlibs/chepai/test/images/last.jpg',img)
    
    # tf.summary.image('read_image', img, 3)
    
    return name, img

def get_next_batch(data):

    batch_size = len(data)
    
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    
    batch_y = np.zeros([batch_size, MAX_NP * CHAR_SET_LEN])
    
    for i in range(batch_size):
    
    text, image = getImageAndName(data[i])
    
    # image = convert2gray(image)
    
    # batch_x[i, :] = image.flatten() / 255 # (image.flatten()-128)/128 mean为0
    
    batch_x[i, :] = image.flatten()
    
    batch_y[i, :] = text2vec(text)

    return batch_x, batch_y

def text2vec(text): 
    text_len = len(text) 
    if text_len > MAX_NP: 
        raise ValueError('验证码最长4个字符') 
    vector = np.zeros(MAX_NP * CHAR_SET_LEN) 
def char2pos(c): 
    if c == '_': 
        k = 62 
        return k 
    k = ord(c) - 48 
    if k > 9: 
        k = ord(c) - 55 
    if k > 35: 
        k = ord(c) - 61 
    if k > 61: # 说明是中文字
         k = dicthz[c] # 
         raise ValueError('No Map') 
     return k 
    for i, c in enumerate(text): #
        print text idx = i * CHAR_SET_LEN + char2pos(c) # print("idx" + str(idx)) #print i,CHAR_SET_LEN,char2pos(c),idx vector[idx] = 1 return vector# 向量转回文本def vec2text(vec): # char_pos = vec.nonzero()[0] char_pos = vec text = [] for i, c in enumerate(char_pos): char_at_pos = i # c/63 char_idx = c % CHAR_SET_LEN if char_idx < 10: char_code = char_idx + ord('0') zm = chr(char_code) elif char_idx < 36: char_code = char_idx - 10 + ord('A') zm = chr(char_code) elif char_idx < 62: char_code = char_idx - 36 + ord('a') zm = chr(char_code) elif char_idx == 62: char_code = ord('_') zm = chr(char_code) else: zm = "N" # 说明是中文 for key in dicthz: if char_idx == dicthz[key]: zm = key # raise ValueError('error') text.append(zm) return "".join(text)
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) #None代表不限条数的输入Y = tf.placeholder(tf.float32, [None, MAX_NP * CHAR_SET_LEN])keep_prob = tf.placeholder(tf.float32) # dropoutdef variable_summaries(var, name): """Attach a lot of summaries to a Tensor.""" with tf.name_scope('summaries'): mean = tf.reduce_mean(var) tf.summary.scalar('mean/' + name, mean) with tf.name_scope('stddev'): stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean))) tf.summary.scalar('sttdev/' + name, stddev) tf.summary.scalar('max/' + name, tf.reduce_max(var)) tf.summary.scalar('min/' + name, tf.reduce_min(var)) tf.summary.histogram(name, var)def detect_np_cnn(X, keep_prob, w_alpha=0.01, b_alpha=0.1): with tf.name_scope('image_input'): x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]) tf.summary.image('image_input', x, 3)
    with tf.name_scope('input_cnn_filter1'): with tf.name_scope('input_weight1'): w_c1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1)) variable_summaries(w_c1, 'input_cnn_filter1/input_weight1') with tf.name_scope('input_biases1'): b_c1 = tf.Variable(tf.constant(0.1, shape=[32])) variable_summaries(b_c1, 'input_cnn_filter1/input_biases1') conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)) tf.summary.histogram('input_cnn_filter1', conv1) conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') conv1 = tf.nn.dropout(conv1, keep_prob)# print(conv1.get_shape())with tf.name_scope('input_cnn_filter2'): with tf.name_scope('input_weight2'): w_c2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1)) variable_summaries(w_c2, 'input_cnn_filter2/input_weight2') with tf.name_scope('input_biases2'): b_c2 = tf.Variable(tf.constant(0.1, shape=[64])) variable_summaries(b_c2, 'input_cnn_filter2/input_biases2')conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))tf.summary.histogram('input_cnn_filter2', conv2)conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')conv2 = tf.nn.dropout(conv2, keep_prob)# print(conv2.get_shape())with tf.name_scope('input_cnn_filter3'): with tf.name_scope('input_weight3'): w_c3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1)) variable_summaries(w_c3, 'input_cnn_filter3/input_weight3') with tf.name_scope('input_biases3'): b_c3 = tf.Variable(tf.constant(0.1, shape=[64])) variable_summaries(b_c3, 'input_cnn_filter3/input_biases3')conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))tf.summary.histogram('input_cnn_filter1', conv3)conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')conv3 = tf.nn.dropout(conv3, keep_prob)# print(conv3.get_shape())# Fully connected layerwith tf.name_scope('input_fully_layer'): with tf.name_scope('input_fully_weight'): w_d = tf.Variable(tf.truncated_normal([5 * 13 * 64, 1024], stddev=0.1)) variable_summaries(w_d, 'input_fully_layer/input_fully_weight') with tf.name_scope('input_fully_biases'): b_d = tf.Variable(tf.constant(0.1, shape=[1024])) variable_summaries(b_d, 'input_fully_layer/input_fully_biases')dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))tf.summary.histogram('input_fully_layer', dense)dense = tf.nn.dropout(dense, keep_prob)with tf.name_scope('output'): with tf.name_scope('output_w'): w_out = tf.Variable(tf.truncated_normal([1024, MAX_NP * CHAR_SET_LEN], stddev=0.1)) variable_summaries(w_out, 'output/output_w') with tf.name_scope('output_b'): b_out = tf.Variable(tf.constant(0.1, shape=[MAX_NP * CHAR_SET_LEN])) variable_summaries(b_out, 'output/output_b')out = tf.add(tf.matmul(dense, w_out), b_out)tf.summary.histogram('output', out)# out = tf.nn.softmax(out)return out
    traindata, testdata = getSplitData(PATH)def train_detect_np_cnn(max_step=200): X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) Y = tf.placeholder(tf.float32, [None, MAX_NP * CHAR_SET_LEN]) keep_prob = tf.placeholder(tf.float32) # dropout output = detect_np_cnn(X, keep_prob) # loss # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y)) with tf.name_scope('loss'): loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y)) tf.summary.scalar('loss', loss) # 可视化观看常量 # 最后一层用来分类的softmax和sigmoid有什么不同？ # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰 with tf.name_scope('train'): optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss) predict = tf.reshape(output, [-1, MAX_NP, CHAR_SET_LEN]) max_idx_p = tf.argmax(predict, 2) YY = tf.reshape(Y, [-1, MAX_NP, CHAR_SET_LEN]) max_idx_l = tf.argmax(YY, 2) correct_pred = tf.equal(max_idx_p, max_idx_l) with tf.name_scope('accuracy'): accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) tf.summary.scalar('accuracy', accuracy) # 可视化观看常量 saver = tf.train.Saver() with tf.Session() as sess: # 合并到Summary中 merged = tf.summary.merge_all() # 选定可视化存储目录 writer = tf.summary.FileWriter("/Users/adminjackfy/downloads/trainlibs/chepai/", sess.graph) test_writer = tf.summary.FileWriter("/Users/adminjackfy/downloads/trainlibs/chepai/test/", sess.graph) sess.run(tf.global_variables_initializer()) step = 0 while True: batch_x, batch_y = get_next_batch(sampleTrain(128, traindata)) _, lossSize = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8}) # writer.add_summary(summary, step) if step % 5 == 0: print("step is:" + str(step), u"损失函数大小为" + str(lossSize)) batch_x_test, batch_y_test = get_next_batch(testdata) summary, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.}) writer.add_summary(summary, step) # print(sess.run(max_idx_p, feed_dict={X:batch_x_test, Y: batch_y_test, keep_prob: 1.})) # print(sess.run(max_idx_l, feed_dict={X:batch_x_test, Y: batch_y_test, keep_prob: 1.})) print("step is:" + str(step), "acc is :" + str(acc)) if step == max_step: saver.save(sess, "./model/crack_capcha.model") break step += 1 # 训练

def predict(testdata):
     X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) 
     keep_prob = tf.placeholder(tf.float32) 
     output = detect_np_cnn(X, keep_prob) 
     saver = tf.train.Saver() 
     with tf.Session() as sess: # 
         sess.run(tf.global_variables_initializer()) 
         saver.restore(sess, tf.train.latest_checkpoint('./model/')) 
         batch_size = len(testdata) 
         count = 0 
         for i in range(batch_size): 
             text, image = getImageAndName(testdata[i]) # 
             image = convert2gray(image) # 
             captcha_image = image.flatten() / 255 
             captcha_image = image.flatten() 
             predict = tf.argmax(tf.reshape(output, [-1, MAX_NP, CHAR_SET_LEN]), 2) 
             text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1}) 
             predict_text = text_list[0].tolist() predict_value = vec2text(predict_text) 
             flag = text == predict_value 
             if flag: 
                 count += 1
                  print("真实值: {}, 预测值: {}, 是否相等: {}".format(text, predict_value, flag)) 
                  print(' 识别结果: {}/{}={}'.format(count, batch_size, count / batch_size))

if __name__ == '__main__': train_detect_np_cnn(max_step=10000) #训练10000次 #predict(testdata)