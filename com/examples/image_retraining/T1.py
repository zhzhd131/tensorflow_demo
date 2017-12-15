import os
import time
#import threadpool 
import request as request3
  


INDEX =1



PATH="/home/wait/"  

def down(url):
    try:
        temp=str(url).replace("\n", "").split(sep="/");
        filename=temp[temp.__len__()-1]
        
        print(filename)
        request3.urlretrieve(url, PATH+filename)
    except Exception as e:
          print(e)   
    #image_data = urllib.request.urlopen(image_url).read();
def mul_img_handle(img_list):
    t=time.time()
    pool = threadpool.ThreadPool(4) 
    requests = threadpool.makeRequests(down, img_list) 
    [pool.putRequest(req) for req in requests] 
    pool.wait()
    print(' 本次处理耗时:%d '%((time.time()-t)))    
    
def test7():
    list=[]
    f=open('tmp.txt')
    index =1
    while 1:
        line = f.readline()
        list.append(line)
        index =index+1
        if not line:
            break
 
    

    f.close() 
    print(index) 
    return  list   
    

    
        
if __name__ == "__main__":
   
  list= test7()
  mul_img_handle(list)
  #down("http://wsjt.wopaitv.com/live-c4c8d333d46e5ba3--20171025190118.jpg")
    