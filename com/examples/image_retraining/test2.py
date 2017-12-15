'''
Created on 2017年9月14日

@author: zhangzd
'''

'''
Created on 2017年9月12日

@author: zhangzd
'''

import request   as request2
import threadpool 
import time


def ppp(index):
    iii=index%2
    image_url='http://123.103.7.233:8080/img/one?url=http://wsjt.wopaitv.com/blive-dc201cf110c63f6a--20170817155149.jpg'
    #image_url='http://123.103.7.233:8080/img/sort'
    #image_url='http://123.103.7.233:8080/img/mul?urls=http://wsjt.wopaitv.com/blive-dc201cf110c63f6a--20170817155149.jpg,http://wsjt.wopaitv.com/blive-9774354e8ecda078--20170822134722.jpg,http://wsjt.wopaitv.com/blive-9774354e8ecda078--20170822134808.jpg,http://wsjt.wopaitv.com/blive-9774354e8ecda078--20170822134843.jpg,http://wsjt.wopaitv.com/blive-9774354e8ecda078--20170822134906.jpg'
    print(image_url)
    image_data = request2.urlopen(image_url).read()
    print(image_data)
def ppp2(index):
    
    return index  
    
def mul_img_handle(img_list):
    t=time.time()
    pool = threadpool.ThreadPool(8) 
    requests = threadpool.makeRequests(ppp, img_list) 
    [pool.putRequest(req) for req in requests] 
    pool.wait()
    print(' 本次处理耗时:%d '%((time.time()-t)))


def bubble_sort(lists):
    # 冒泡排序
    count = len(lists)
    for i in range(0, count):
        for j in range(i + 1, count):
            if lists[i] > lists[j]:
                lists[i], lists[j] = lists[j], lists[i]
    return lists  
  
if __name__ == "__main__":    
    img_list =[]
    for i  in range(5000):
        img_list.append(2000-i)
    #print(img_list) 
    t=time.time()   
   # bubble_sort(img_list) 
    mul_img_handle(img_list)
    print(' 本次处理耗时:%f '%((time.time()-t)))
    print(img_list)    
    #mul_img_handle(img_list)