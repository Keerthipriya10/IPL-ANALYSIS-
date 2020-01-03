from __future__ import print_function

import sys

import numpy as np
from pyspark import SparkContext
from operator import add
def splitf(x):
	return x.split(',')
#['ball', '1', '0.1', 'Sunrisers Hyderabad', 'DA Warner', 'S Dhawan', 'TS Mills', '0', '0', '""', '""']
def pairruns(x):
    key=(x[4]+'-'+x[6])
    value=None
    list1=['run out','retired hurt']
    if len(x[9])>2 and x[9] not in list1:
        value='W'
    else:
        value=int(x[7])	
    fkey=key+'|'+str(value)
    return fkey,1
def givebb(x):
    key=list(x[0].split('|'))
    value=key[1],x[1]
    return key[0],(value)
def returnval(x):
    key=x[0]
    bats=[0,0,0,0,0,0,0]
    for i in x[1]:
        if i[0]=='5':
            continue
        elif i[0]=='0':
            bats[0]=i[1]
        elif i[0]=='1':
            bats[1]=i[1]
        elif i[0]=='2':
            bats[2]=i[1]
        elif i[0]=='3':
            bats[3]=i[1]
        elif i[0]=='4':
            bats[4]=i[1]
        elif i[0]=='6':
            bats[5]=i[1]
        else:
            bats[6]=i[1]      
    return key,bats
#[0,1,2,4,6,wickets]
#('DA Warner-A Choudhary', [1, 0, 0, 1, 1, 1])
def probrare(x):
    prob=[0,0,0,0,0,0,0]
    for i in range(len(x[1][0])):
        prob[i]=float(x[1][0][i])/float(x[1][1])
    return x[0],prob
def pars(x):
    key=x[0]
    value=x[1]
    balls=sum(x[1])
    return key,(value,balls)
#('R Dravid-MG Johnson', [0.7058823529411765, 0.0, 0.0, 0.23529411764705882, 0.0, 0.058823529411764705])	
def comaset(x):
    bb=x[0].split('-')
    string=bb[0]+","+bb[1]+","+str(x[1][0])+","+str(x[1][1])+","+str(x[1][2])+","+str(x[1][3])+","+str(x[1][4])+","+str(x[1][5])+","+str(x[1][6])
    return string
    
     
          

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: <>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonKMeans")
    lines = sc.textFile(sys.argv[1])
    lines=lines.map(lambda x:splitf(x))
    player=lines.filter(lambda x:x[0]=="ball")
    balbat=player.map(lambda x:pairruns(x)).reduceByKey(add)
    bblist=balbat.map(lambda x:givebb(x)).groupByKey().map(lambda x : (x[0], list(x[1])))
    noofballs=bblist.map(lambda x:returnval(x)).map(lambda x:pars(x)).filter(lambda x:x[1][1]>0)
    prob=noofballs.map(lambda x:probrare(x)).map(lambda x:comaset(x))
    prob.saveAsTextFile("/home/chaitra/bigdata/Step-2/every-player")
    #print(prob)
    sc.stop()

