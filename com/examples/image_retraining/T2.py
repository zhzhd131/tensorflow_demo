import os
import time
import threadpool 
import tensorflow as tf    

    
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
sess2 = tf.Session()
softmax_tensor2 = sess2.graph.get_tensor_by_name('final_result:0')   



    
def getscore2(image_path):
   
    try:
        try: 
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        except Exception as e:
            print(e)  
            return  0     
        predictions = sess2.run(softmax_tensor2, {'DecodeJpeg/contents:0': image_data})  
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('img :%s  ,%s = %.5f ,' % (image_path,human_string, score))
            return score
    except:
        print (e)


def mul_img_handle(img_list):
    t=time.time()
    pool = threadpool.ThreadPool(4) 
    requests = threadpool.makeRequests(getscore2, img_list) 
    [pool.putRequest(req) for req in requests] 
    pool.wait()
    print(' 本次处理耗时:%d '%((time.time()-t)))
          
def gci(filepath):
    #遍历filepath下所有文件，包括子目录
    img_list =[] 
    index =1
    files = os.listdir(filepath)
    for fi in files:
        if(index==5000):
            break
        fi_d = os.path.join(filepath,fi)            
        if os.path.isdir(fi_d):

            continue                
        else:
            index=index +1
            img_list.append(os.path.join(filepath,fi_d))
            #getscore2(image_path=os.path.join(filepath,fi_d))
  
    return  img_list 
    
if __name__ == "__main__":
    #str=db_query("SELECT  url FROM  bvcs.`room_screen_print` r WHERE r.`room_id`=2545024 limit 100")
    t=time.time() 
    img_list=gci('/home/zzd/data/img3/')
    for path in img_list:
        getscore2(path)
    #mul_img_handle(img_list)
    print('计算耗时%d'%(time.time()-t))