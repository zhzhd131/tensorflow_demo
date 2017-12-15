#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2017年10月20日

@author: zm-06-01-037
'''
import re
import 
import numpy
import pandas
import codecs
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import json

print '这段代码可以把字典的所有key输出为一个数组'
def str( content, encoding='utf-8'):  
    # 只支持json格式  
    # indent 表示缩进空格数  
    return json.dumps(content, encoding=encoding, ensure_ascii=False, indent=4)  
     

w_18=codecs.open( '18.txt','r','utf-8')
print w_18
w_18_Content=w_18.read()
w_18.close()

w_19=codecs.open("19.txt","r","utf-8")
w_19_Content=w_19.read()
w_19.close()

stat=[]
stopwords=set(["的","和","是","在","要","为","我们","以","把","了","到","上","有"])




segs=jieba.cut(w_18_Content)
#print segs
#print ", ".join(segs) 
zhPattern=re.compile('[\u4e00-\u9fa5]+')
for seg in segs:
    #print  # u'\u6c49'
    if zhPattern.search(repr(seg)):
        if seg not in stopwords:
            
            stat.append({
                "word":seg,
                "from":"十八大"
            })
    
    
segs=jieba.cut(w_19_Content)
for seg in segs:
    if zhPattern.search(seg):
        if seg not in stopwords:
            stat.append({
                "word":seg,
                "from":"十九大"
            })


#print     stat        
statDF=pandas.DataFrame(stat)
#statDF.pivot(index, columns, values)

ptStat=statDF.pivot_table(
    index="word",
    columns="from",
    fill_value=0,
    aggfunc=numpy.size
)
#print ptStat['十八大'].to_dict()
dddd=ptStat['十八大'].to_dict()

d = {
 'AdaHJa是否在A中出现m': 95, #key : value
 'Lisa': 85,
 'Bart': 59
 }

list =[u'方法使得可以',u'方法使得中文234324234']
print list
print d
wordc=WordCloud(
   

    )
wordc.fit_words(dddd)
plt.imshow(wordc)
plt.show()
#print ptStat