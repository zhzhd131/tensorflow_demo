#!/usr/local/bin/python2.7
# encoding: utf-8
'''
book.test.T3 -- shortdesc

book.test.T3 is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2017 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''
from flask import Flask
import sys
app = Flask(__name__)


index=1
app = Flask(__name__)
print('head------------------')

def test(index):
    print('test excute: %d'%(index))

@app.route('/one')   
def one():
    global index
    index=index+1
    test(index)
    
   
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=sys.argv[1],threaded=True)