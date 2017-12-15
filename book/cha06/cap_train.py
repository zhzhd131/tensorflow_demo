'''
Created on 2017年12月5日

@author: zhangzd
'''
import tensorflow as tf
from PIL import Image
from  nets import nets_factory
import  book.cha06.cteate_image as im
import book.cha06.tfrecord_3_img_reader as read



CHAR_SET_LEN=im.number.__len__()
IMAGE_HEIGHT=60
IMAGE_WEIGHT=160
BATCH_SIZE=25
TF_FILE="model/cap1.tfrecords0"
x=tf.placeholder(tf.float32,[None,160,60])
y0=tf.placeholder(tf.float32, [None])

y1=tf.placeholder(tf.float32, [None])

y2=tf.placeholder(tf.float32, [None])

y3=tf.placeholder(tf.float32, [None])
r1=tf.Variable(0.003,dtype=tf.float32)

# def read_tfFile(path):
#    
#     
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
#     image= tf.reshape(image_raw, [244,244])
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

if __name__ == '__main__':
    image ,label0,label1,label2,label3 =read.decodeimage()
    batch_image ,batch_label0,batch_label1,batch_label2,batch_label3 =tf.train.shuffle_batch([image ,label0,label1,label2,label3], batch_size=BATCH_SIZE, capacity=2000,  min_after_dequeue=3, num_threads=1)
    
    train_network_fn=nets_factory.get_network_fn(
        "alexnet_v2",num_classes=4,weight_decay=0.005,is_training=True)
    
    with tf.Session() as  sess:
        X=tf.reshape(x,[BATCH_SIZE,160,60,3])   
        logist0,logist1,logist2,logist3,endpoints= train_network_fn(X)
        
        one_hot_lables0=tf.one_hot(indices=tf.cast(y0,tf.int32), depth=CHAR_SET_LEN)
        one_hot_lables1=tf.one_hot(indices=tf.cast(y1,tf.int32), depth=CHAR_SET_LEN)
        one_hot_lables2=tf.one_hot(indices=tf.cast(y2,tf.int32), depth=CHAR_SET_LEN)
        one_hot_lables3=tf.one_hot(indices=tf.cast(y3,tf.int32), depth=CHAR_SET_LEN)
        
        loss0=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist0,lables=one_hot_lables0))
        loss1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist1,lables=one_hot_lables1))
        loss2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist2,lables=one_hot_lables2))
        loss3=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist3,lables=one_hot_lables3))
        
        total_loss=(loss0+loss1+loss2+loss3)/4.0
        
        optimizer =tf.train.AdamOptimizer(learning_rate=r1).minimize(total_loss)
    
        #准确率
        correct_prediction0=tf.equal(tf.argmax(one_hot_lables0, 1), tf.argmax(logist0,1))
        acc0=tf.reduce_mean(tf.cast(correct_prediction0,tf.float32))
        
        correct_prediction1=tf.equal(tf.argmax(one_hot_lables1, 1), tf.argmax(logist1,1))
        acc1=tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
        
        correct_prediction2=tf.equal(tf.argmax(one_hot_lables2, 1), tf.argmax(logist2,1))
        acc2=tf.reduce_mean(tf.cast(correct_prediction2,tf.float32)) 
        
        correct_prediction3=tf.equal(tf.argmax(one_hot_lables3, 1), tf.argmax(logist3,1))
        acc3=tf.reduce_mean(tf.cast(correct_prediction3,tf.float32))       
        
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        coord=tf.train.Coordinator()
        
        threads=tf.train.start_queue_runners(sess, coord)
        
        
        for i in range(6000):
            b_image,b_lable0,b_lable1,b_lable2,b_lable3=sess.run([ batch_image ,batch_label0,batch_label1,batch_label2,batch_label3 ])
            sess.run(optimizer,feed_dict={x:b_image,y0:b_lable0,y1:b_lable1,y2:b_lable2,y3:b_lable3})
            
            if i%20==0:
                if(i%1000==0):
                    sess.run(tf.assign(r1,r1/3))
                    
                a0,a1,a2,a3,loss_=sess.run([acc0,acc1,acc2,acc3,total_loss],feed_dict={x:b_image,y0:b_lable0,y1:b_lable1,y2:b_lable2,y3:b_lable3})    
                learning_ra=sess.run(r1)
                print("次数： %d 损失，% .3f 准确率 :%.3f,%.3f,%.3f,%.3f"%(i,loss_,a0,a1,a2,a3)) 
                
                if i%1000==0:
                    saver.save(sess, "model/cap.model", global_step=1)   

    
        coord.request_stop()
        coord.join(threads)