'''
Created on 2017年9月12日

@author: zhangzd
'''
import pymysql
import redis
import time
import datetime
from time import sleep
import tensorflow as tf 
import urllib.request  as urequest
import threadpool 
THRESHOLD_VALUE =0.85

DB_HOST='rm-2zefubs0983lvpe38.mysql.rds.aliyuncs.com'
DB_USER='statistics_user'
DB_PASSWD='Rad8h99Lk#dpxcdfaa'
DB_DATABASE='bvcs_statistics'
#上次时间戳
PRE_TIME = 0 
PRE_TIME_TEMP = 0 
REDIS_HOST ='55b1932718124bf5.m.cnbja.kvstore.aliyuncs.com'
REDIS_PASSWORD ='55b1932718124bf5:99Blive8888'
ROOM_SCREENSHOT_ID='ROOM_SCREENSHOT_ID_'
ROOM_SCREENSHOT_HANDLE='ROOM_SCREENSHOT_HANDLE'

pool = redis.ConnectionPool(host=REDIS_HOST, port=6379, password=REDIS_PASSWORD)

IMG_COUNT = 0

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
sess2 = None
softmax_tensor2  = None  

      
def getscore(image_url):
    global sess2,softmax_tensor2,IMG_COUNT
    IMG_COUNT=IMG_COUNT+1
    try:
        if sess2 is None:
             sess2 = tf.Session()
             softmax_tensor2 = sess2.graph.get_tensor_by_name('final_result:0')
        try: 
            t=time.time()      
            image_data = urequest.urlopen(image_url).read()
            print("download time:%d"%((time.time()-t)*1000))
            t=time.time()
        except Exception as e:
            return  {'lable':'zhengchang','score':0}      
        predictions = sess2.run(softmax_tensor2, {'DecodeJpeg/contents:0': image_data})  
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        #scores = {} 
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if human_string=='seqing' or human_string=='guanggao':   
                print('')
            #scores[human_string]=(score)
            #print('%s = %.5f ,' % (human_string, score))
            print("cal score time:%d"%((time.time()-t)*1000))
            #返回概率最大的标签
            return  {'lable':human_string,'score':score}
            #return  [human_string,score]
            #print('%s = %.5f ,' % (human_string, score),end='')
        
        #return  scores
    except Exception as e:
        sess2.close()
        sess2 = None
        
        print (e)
            


def db_query(sql):
    # 打开数据库连接
    db = pymysql.connect(host=DB_HOST, port=3306, user=DB_USER, passwd=DB_PASSWD, database=DB_DATABASE)
    cursor = db.cursor()
     
    # SQL 查询语句
    #sql = "SELECT * FROM user WHERE id > 100"
    list2 = []
    try:
       # 执行SQL语句
       cursor.execute(sql)
       # 获取所有记录列表
       results = cursor.fetchall()
       for row in results:
          fname = row[0]
          list2.append(fname)
           # 打印结果
          #print (fname,lname,age)
    except:
       print ("Error: unable to fetch data")
     
    # 关闭数据库连接
    db.close()

    #print(str_convert) 
    #print(' result size %d ,' % (list2.__len__()))
    return  list2


def redis2(room_list):
    r = redis.Redis(connection_pool=pool)
    pipe = r.pipeline(transaction=False)
    

    for room_id in room_list:
        #pipe.zrevrange(ROOM_SCREENSHOT_ID+str(room_id), 0, 1, withscores=False, score_cast_func=float)
        pipe.zrevrangebyscore('ROOM_SCREENSHOT_ID_'+str(room_id), 9999990675000, PRE_TIME,  withscores=False, score_cast_func=float)
    
    list2=pipe.execute()
    
    #print('           SCREENSHOT_list size:%d'%(list2.__len__()))
    

    
    index=0
    for temp in list2:
        #print('                           room:%d img：%d'%(room_list[index],temp.__len__()))
        for t in temp:
            url=str(t)[2:len(str(t))-1]
            #鉴黄开始
            lable=getscore(url)
            if lable['lable'] !='zhengchang' and lable['score'] > THRESHOLD_VALUE:
            
               pipe.zadd(ROOM_SCREENSHOT_HANDLE,url,lable['score'])
               
        index=index+1   
        
           
    pipe.execute()    
        #print(temp)
   # r = redis.Redis(host='192.168.50.49', port=6379)
    #r.set('age','22')
    #print(r.get('age'))def getDbRoom():
   # Map<String, Double> isRealRoom = RedisUtilNew.pipelineZscores(RedisKeyConstant.ROOM_ORDER, roomIdList);
    #Map<String, Set<String>> roomScrCol = RedisUtilNew.pipelineZrevrange(RedisKeyConstant.ROOM_SCREENSHOT_ID, roomIdList, 0, 0);
    
   # print(r.get('ROOM_SERVERURLS_43741'))

def mul_img_handle():
    global  PRE_TIME_TEMP,PRE_TIME
    
    now = datetime.datetime.now()
    delta = datetime.timedelta(hours=-12)
    n_days = now + delta
    sql ="SELECT r.id FROM  bvcs.`room` r ,bvcs.`ruser`  ru WHERE  r.`creator_id`=ru.id AND ru.organization_id>0 AND r.`create_at`>'"+n_days.strftime('%Y-%m-%d %H:%M:%S')+"' AND  r.`duration`=0  AND r.id>2548000"
    
    room_list=db_query(sql) 
    print('      room_list size:%d'%(room_list.__len__()))
    index = 1
    temp_list=[]
    par_list=[]
    for temp in room_list:
        if(index % 5!=0):
            temp_list.append(temp)
        else:
            par_list.append(temp_list)
            temp_list=[]
        index=index+1 
        
    if(temp_list.__len__()>0):
        par_list.append(temp_list)
        temp_list.clear()           
    
    
    
   
    
    t=time.time()
    
    PRE_TIME_TEMP =int(round(time.time()*1000)) 
    pool = threadpool.ThreadPool(5) 
    requests = threadpool.makeRequests(redis2, par_list) 
    [pool.putRequest(req) for req in requests] 
    pool.wait()
    print(' 本次处理耗时:%d 图片总数量:%d  PRE_TIME:%d'%((time.time()-t),IMG_COUNT,PRE_TIME))
    PRE_TIME =PRE_TIME_TEMP 
           
def img_hande():
    global  PRE_TIME_TEMP,PRE_TIME
    
    now = datetime.datetime.now()
    delta = datetime.timedelta(hours=-12)
    n_days = now + delta
    sql ="SELECT r.id FROM  bvcs.`room` r ,bvcs.`ruser`  ru WHERE  r.`creator_id`=ru.id AND ru.organization_id>0 AND r.`create_at`>'"+n_days.strftime('%Y-%m-%d %H:%M:%S')+"' AND  r.`duration`=0  AND r.id>2548000"
    print(sql) 
    room_list=db_query(sql) 
    print('      room_list size:%d'%(room_list.__len__()))  
     
    t=time.time()
    PRE_TIME_TEMP =int(round(time.time()*1000))
    redis2(room_list)  
    print(' 本次处理耗时:%d 图片总数量:%d  PRE_TIME:%d'%((time.time()-t),IMG_COUNT,PRE_TIME))
    PRE_TIME =PRE_TIME_TEMP 
    
if __name__ == "__main__":
    global sess2 ,softmax_tensor2,PRE_TIME
  

    sess2 = tf.Session()
    softmax_tensor2 = sess2.graph.get_tensor_by_name('final_result:0') 
    print(PRE_TIME)
    #global  PRE_TIME  
    PRE_TIME = int(round(time.time()*1000-5*1000))
    
    print(PRE_TIME)
    while True:
        #处理速度跟不上 只处理前8秒的图片
        PRE_TIME = int(round(time.time()*1000-8*1000))
        img_hande()
        print('休眠 1秒')
        time.sleep(1) # 休眠0.1秒 
    #str=db_query("SELECT  url FROM  bvcs.`room_screen_print` r WHERE r.`room_id`=2545024 limit 100")
 
    #redis2()