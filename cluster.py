from __future__ import print_function

import sys
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
import numpy as np
from pyspark import SparkContext
from operator import add

def splitre(x):
    key=x.split(',') 
    #return key[1],key[0] 
    return key[0],key[1]
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
    bb=x[0].split('-') #(('B Kumar', 'SR Watson'), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    bat_bowl=(bb[0],bb[1])
    return bat_bowl,x[1]
def somefunc(x):
    return x[0][0],(x[0][1],x[1])
def clusterlist(x):
    return x[0]
def bowlerfunc(x):
    bowler=x[1][0][0]
    value=(x[0],x[1][0][1],x[1][1]) 
    return bowler,value   
def clustt(x):
    batsmen=x[1][0][0]
    bowler=x[0]
    prob=x[1][0][1]
    bac=x[1][0][2]
    boc=x[1][1]
    return batsmen,bowler,prob,bac,boc
def findproc(x):
    #('SK Warne', 'SS Sarkar', [0.375, 0.5, 0.125, 0.0, 0.0, 0.0, 0.0], '3', '1')
    key=(x[3],x[4])
    value=x[2]
    return key,value
'''(('4', '3'), [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.6666666666666666, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333333333333333], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
'''
def values(x):
    key=x[0]
    final=[0,0,0,0,0,0,0]
    length=len(x[1])
    for i in range(len(x[1][0])):
        psum=0
        for lis in x[1]:
            psum+=lis[i]
        final[i]=psum/length
    return key,final   
def strringfunc(x):
    #(('2', '1'), [0.4612983312983314, 0.34920838420838424, 0.02702075702075702, 0.002564102564102564, 0.021947496947496946, 0.006495726495726496, 0.13146520146520144])
    clcl=x[0][0]+","+x[0][1]+","+str(x[1][0])+","+str(x[1][1])+","+str(x[1][2])+","+str(x[1][3])+","+str(x[1][4])+","+str(x[1][5])+","+str(x[1][6])
    return clcl
sc = SparkContext(appName="Cluster_v/s_Cluster")
#clustering batsman
bats = sc.textFile("/home/chaitra/bigdata/Step-2/cluster-bat")
clusterbat=bats.map(lambda x:splitre(x))#.groupByKey().map(lambda x : (x[0], list(x[1])))
#cllistba=bats.map(lambda x:clusterlist(x)).collect()
##clustering bowlers
bowls= sc.textFile("/home/chaitra/bigdata/Step-2/cluster-bowl")
clusterbowl=bowls.map(lambda x:splitre(x))
#cllistbo=bowls.map(lambda x:clusterlist(x)).collect()
#data for player-player statistics
lines = sc.textFile("/home/chaitra/bigdata/Step-2/alldata.csv")
lines=lines.map(lambda x:splitf(x))
player=lines.filter(lambda x:x[0]=="ball")
balbat=player.map(lambda x:pairruns(x)).reduceByKey(add)
bblist=balbat.map(lambda x:givebb(x)).groupByKey().map(lambda x : (x[0], list(x[1])))
noofballs=bblist.map(lambda x:returnval(x)).map(lambda x:pars(x)).filter(lambda x:x[1][1]>0) #('DA Warner-DR Smith', ([1, 3, 0, 0, 2, 1, 0], 7))
prob=noofballs.map(lambda x:probrare(x)).map(lambda x:comaset(x)) #('DA Warner-CJ Jordan', [0.5, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0])
probo=prob.map(lambda x:somefunc(x))
clusterb=probo.join(clusterbat) #('DJ Muthuswami', (('HV Patel', [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), '4')) for batsmen clustering
clusterbo=clusterb.map(lambda x:bowlerfunc(x))
clusterbow=clusterbo.join(clusterbowl) #('SS Sarkar', (('RA Jadeja', [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5], '1'), '1')) for bowler cluster
cluster=clusterbow.map(lambda x: clustt(x))
procc=cluster.map(lambda x:findproc(x)).groupByKey().map(lambda x : (x[0], list(x[1])))
procf=procc.map(lambda x:values(x)).map(lambda x:strringfunc(x))
#clusterrr=procf.map(lambda x:x[0]).collect()
procf.saveAsTextFile("/home/chaitra/bigdata/Step-2/cluster-cluster")
#print(procf)
sc.stop()

