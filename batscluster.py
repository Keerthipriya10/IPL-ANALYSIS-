from __future__ import print_function

import sys
import math
import numpy as np
from pyspark import SparkContext
from operator import add
def nearst_centroid(point, centroids):
    cluster = 0
    closest = math.inf
    for i in range(len(centroids)):
        dist=sum([(a - b) ** 2 for a, b in zip(point,centroids[i])])
        initialdist=math.sqrt(dist)
        if initialdist < closest:
            closest = initialdist
            cluster = i
    return cluster
def nearst_cluster(p, centers):
    bestIndex = len(centers)+1
    closest = math.inf
    i=0
    while (i<len(centers)):
            Dist = np.sum((np.array(p[1]) - centers[i]) ** 2)
            if Dist < closest:
                closest = Dist
                bestIndex = i
            i=i+1
    key=p[0]+","+str(bestIndex)
    return key
def transform(x):
	return (x[6],np.array([float(x[1]),float(x[2]),float(x[3])/float(x[0]),float(x[4])/float(x[0]),float(x[5])/float(x[0])]))
def conver(x):
	return np.array([float(x[1]),float(x[2]),float(x[3])/float(x[0]),float(x[4])/float(x[0]),float(x[5])/float(x[0])])
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="PythonKMeans")
    lines = sc.textFile(sys.argv[1])
    indices=[2,7,9,11,13,14,0]
    header = lines.first() 
    lines = lines.filter(lambda row:row != header)
    data = lines.map(lambda line:line.split(",")).map(lambda x:[x[idx] for idx in indices])		
    each_center= data.map(lambda x:conver(x))
    batsman=data.map(lambda x:transform(x))
    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])

    kPoints = each_center.takeSample(False, K, 1)
    Dist = 100.0
    
    while Dist > convergeDist:
        closest = each_center.map(lambda p:(nearst_centroid(p, kPoints),(p,1)))
        #print(closest.collect()[0:5])
        #print("\n\n\n\n")
        pointStats = closest.reduceByKey(add)
        #print(pointStats.collect()[:1])
        #print("\n\n\n\n")
        newPoints = pointStats.map(lambda st: (st[0], st[1][0]/ st[1][1])).collect()
        #print("newpoints",newPoints[0:5])
        #print("\n\n\n\n")
        Dist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)
        for (iK, p) in newPoints:
            kPoints[iK] = p
        
      
    #print("Final centers: " + str(kPoints))
    batcluster = batsman.map(lambda x : nearst_cluster(x,kPoints))
    batcluster.saveAsTextFile("/home/chaitra/bigdata/Step-2/cluster-bat")
    #print(batcluster.collect())

    sc.stop()
