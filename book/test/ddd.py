'''
Created on 2017年10月23日

@author: zhangzd
'''
from flask import Flask
import tensorflow as tf
import sys
import urllib.request
from flask.globals import request
import time
#import threadpool 


app = Flask(__name__)
t=time.time()
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
sess2 = tf.Session()
softmax_tensor2  = sess2.graph.get_tensor_by_name('final_result:0')
print(' load处理耗时:%f '%((time.time()-t)))


def getscore3(image_url):

    try:

        t2=time.time()     
        image_data = urllib.request.urlopen(image_url).read()
        print(' image_data处理耗时:%f '%((time.time()-t2)))
        t2=time.time()
        predictions = sess2.run(softmax_tensor2, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        scores = {'url':image_url}
        flag=True
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            scores[human_string]=(score)
            if(flag):
               scores["lable"]=human_string
               scores["score"]=(score)
               flag = False
                
            #print('%s = %.5f ,' % (human_string, score),end='')
        print(' 计算处理耗时:%f '%((time.time()-t2)))
        return  scores
    except Exception as e:
        #sess2.close()
        #sess2 = None

        print (e)

       # print(" top_k:",end='');print((time.time()- t),end='')
def mul_img_handle(img_list):
    t=time.time()
    pool = threadpool.ThreadPool(4) 
    requests = threadpool.makeRequests(getscore3, img_list) 
    [pool.putRequest(req) for req in requests] 
    pool.wait()
    print(' 本次处理耗时:%d '%((time.time()-t)))
    
@app.route('/img/mul')
def mul():

    imgurls=request.args.get('urls')
    print(imgurls)
    if imgurls is None:
        return 'pleace input image url'

    result = {
    'requestId' : 123456789,
    'code' : 1000,
    'msg' : '操作成功',
    'result':'XXXXX'
    }

    list1 = []
    for imgurl in imgurls.split(','):
        list1.append(getscore3(imgurl))


    data_list = {
    'dataList' : list1
    }
    result['result']=data_list
    str = result.__str__().replace("'",'"')
    return str,200,{'Content-Type':'application/json;charset=UTF-8'}
    #return result.__str__().replace("'",'"')       
    #return render_template('data.json'), 201, {'Content-Type': 'application/json'}

@app.route('/img/one')
def one():

    imgurl=request.args.get('url')
    if imgurl is None:
        return 'pleace input image url'

    result = {
    'requestId' : 123456789,
    'code' : 1000,
    'msg' : '操作成功',
    'result':'XXXXX'
    }


    result['result']=getscore3(imgurl)
    str = result.__str__().replace("'",'"')
    return str,200,{'Content-Type':'application/json;charset=UTF-8'}
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=sys.argv[1],threaded=True)
