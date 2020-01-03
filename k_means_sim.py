from __future__ import print_function
import random
import sys
import numpy as np
from pyspark import SparkContext
from operator import add
def spplit(x):
    csva=x.split(',')
    cv=[i for i in csva]
    return cv
def pppp(x):
    #['1', '0', '0.17940323565323563', '0.414550727050727', '0.09604700854700852', '0.0', '0.13121253746253744', '0.13333194583194585', '0.045454545454545456']
    bacn=x[0]
    bocn=x[1]
    lll=[float(x[2]),float(x[3]),float(x[4]),float(x[5]),float(x[6]),float(x[7])]
    summ=sum(lll)
    rep=[i/summ for i in lll]
    for i in range(1,len(rep)):
        rep[i]+=rep[i-1]
    wc=float(x[8])
    return [bacn,bocn,rep,wc]
def findruns(x):
    first=x[0]
    last=x[5]
    randomrun=random.SystemRandom()#random.random()
    randomruns = randomrun.uniform(first,last)
    if randomruns <= x[0]:                         
        return 0				    
    elif randomruns <= x[1]:                     
        return 1
    elif randomruns <= x[2]:
        return 2
    elif randomruns <= x[3]:
        return 3
    elif  randomruns <= x[4]:
        return 4
    elif randomruns <= x[5]:
        return 6
if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: <>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(appName="PythonKMeans")
    #lines = sc.textFile(sys.argv[1])
    cluster=sc.textFile("/home/chaitra/bigdata/Step-2/cluster-cluster/")#0,0,0.26458333333333334,0.41875,0.06666666666666667,0.0,0.09375,0.11458333333333333,0.041666666666666664
    cluster=cluster.map(lambda x:spplit(x)).map(lambda x:pppp(x))
    #('1', '0', [0.18794624687481828, 0.6222374847374846, 0.7228581603581602, 0.7228581603581602, 0.8603189138903423, 0.9999999999999999], 0.045454545454545456)
    batsman = sc.textFile("/home/chaitra/bigdata/Step-2/cluster-bat/")
    batsman = batsman.map(lambda x:spplit(x))#[['Z Khan', '3']]
    #print(batsman.collect())
    bowler =  sc.textFile("/home/chaitra/bigdata/Step-2/cluster-bowl/")
    bowler=bowler.map(lambda x:spplit(x)) #[['MG Johnson', '1']]
    #print(bowler.collect()) 
    playplay= sc.textFile("/home/chaitra/bigdata/Step-2/player-player/")#S Dhawan,YS Chahal,0.15384615384615385,0.6666666666666666,0.10256410256410256,0.0,0.02564102564102564,0.0,0.05128205128205128
    playplay= playplay.map(lambda x:spplit(x)).map(lambda x:pppp(x))
    #('DA Warner', 'CR Brathwaite', [0.27272727272727276, 0.6363636363636365, 0.9090909090909092, 0.9090909090909092, 0.9090909090909092, 1.0], 0.08333333333333333)
    team0_1=sc.textFile("/home/chaitra/bigdata/Step-2/bat-order.txt").collect()
    team1_1=sc.textFile("/home/chaitra/bigdata/Step-2/bowl-order.txt").collect()
    print("First Innings")
    striker_a = team0_1[0]
    non_striker_a = team0_1[1]
    bowlr_a = team1_1[0]
    nextbatsman_a = 2
    wickets_a = 0
    totalruns_a = 0
    overs_a = 0
    outprob_a = 0
    count_a = 1
    noprob_a = 1
    cumprob_a=[]
    #batcl=batsman.filter(lambda x:x[0]=='1').map(lambda x:x[1]).collect()#[['MC Henriques', '1']]
    s_ns_prob_a={}
    s_ns_prob_a[striker_a]=1
    s_ns_prob_a[non_striker_a]=1
    while(overs_a<20 and wickets_a < 10):
        balls_a = 0
        oversc_a=0
        while(balls_a<6 and wickets_a <10):
            #[('DA Warner', 'CR Brathwaite', [0.27272727272727276, 0.6363636363636365, 0.9090909090909092, 0.9090909090909092, 0.9090909090909092, 1.0], 0.08333333333333333)]
            ##('1', '0', [0.18794624687481828, 0.6222374847374846, 0.7228581603581602, 0.7228581603581602, 0.8603189138903423, 0.9999999999999999], 0.045454545454545456)
            try:
                row = playplay.filter(lambda x:(x[0]==striker_a and x[1]==bowlr_a))
                outprob_a = row.map(lambda x:x[3]).collect()[0] #some number
                cumprob_a= row.map(lambda x:x[2]).collect()[0]#[0.16216216216216217, 0.8648648648648649, 0.972972972972973, 0.972972972972973, 1.0, 1.0]
            except:
                batcl=batsman.filter(lambda x:x[0]==striker_a).map(lambda x:x[1]).collect()[0]
                bowlcl=bowler.filter(lambda x:x[0]==bowlr_a).map(lambda x:x[1]).collect()[0]
                row=cluster.filter(lambda x:x[0]==batcl and x[1]==bowlcl)
                outprob_a=row.map(lambda x:x[3]).collect()[0]
                cumprob_a= row.map(lambda x:x[2]).collect()[0]		
            runs_a = 0
            outtime_a=False
            #noprob=noprob-outprob 
            s_ns_prob_a[striker_a]= s_ns_prob_a[striker_a]-outprob_a #1-out probability,initial
            if (s_ns_prob_a[striker_a] > 0.5):
                runs_a=findruns(cumprob_a)
            elif s_ns_prob_a[striker_a] < 0.5:
                wickets_a = wickets_a+1
                striker_a = team0_1[nextbatsman_a]
                nextbatsman_a = (nextbatsman_a+1)%11
                outtime_a=True
                s_ns_prob_a[striker_a] = 1
                print("Wicket taken")
            if(outtime_a==False):
                totalruns_a = totalruns_a+runs_a
                oversc_a=oversc_a+runs_a
                if (runs_a==1 or runs_a==3):
                    striker_a,non_striker_a = non_striker_a,striker_a
            balls_a = balls_a+1
        striker_a,non_striker_a = non_striker_a,striker_a                 
        bowlr_a = team1_1[count_a]                    
        count_a = count_a+1
        overs_a = overs_a+1
        print("Over No:{} \t Runs Scored in over:{}".format(overs_a,oversc_a))
    print("Total Runs:"+str(totalruns_a)+"\t Wickets Taken:"+str(wickets_a))
    print("-------------------------------------------------------------------------")
  ##########################FOR TEAM B ############################## 
    print("Second Innings")
    team0_2=sc.textFile("/home/chaitra/bigdata/Step-2/bat-order1.txt").collect()
    team1_2=sc.textFile("/home/chaitra/bigdata/Step-2/bowl-order1.txt").collect()
    striker_b = team0_2[0]
    non_striker_b = team0_2[1]
    bowlr_b = team1_2[0]
    nextbatsman_b = 2
    wickets_b = 0
    totalruns_b = 0
    overs_b = 0
    outprob_b = 0
    count_b = 1
    noprob_b = 1
    cumprob_b=[]
    #batcl=batsman.filter(lambda x:x[0]=='1').map(lambda x:x[1]).collect()#[['MC Henriques', '1']]
    s_ns_prob_b={}
    s_ns_prob_b[striker_b]=1
    s_ns_prob_b[non_striker_b]=1
    while(overs_b<20 and wickets_b< 10):
        balls_b = 0
        oversc_b=0
        while(balls_b<6 and wickets_b <10):
            #[('DA Warner', 'CR Brathwaite', [0.27272727272727276, 0.6363636363636365, 0.9090909090909092, 0.9090909090909092, 0.9090909090909092, 1.0], 0.08333333333333333)]
            ##('1', '0', [0.18794624687481828, 0.6222374847374846, 0.7228581603581602, 0.7228581603581602, 0.8603189138903423, 0.9999999999999999], 0.045454545454545456)
            try:
                row = playplay.filter(lambda x:(x[0]==striker_b and x[1]==bowlr_b))
                outprob_b = row.map(lambda x:x[3]).collect()[0] #some number
                cumprob_b= row.map(lambda x:x[2]).collect()[0]#[0.16216216216216217, 0.8648648648648649, 0.972972972972973, 0.972972972972973, 1.0, 1.0]
            except:
                batcl=batsman.filter(lambda x:x[0]==striker_b).map(lambda x:x[1]).collect()[0]
                bowlcl=bowler.filter(lambda x:x[0]==bowlr_b).map(lambda x:x[1]).collect()[0]
                row=cluster.filter(lambda x:x[0]==batcl and x[1]==bowlcl)
                outprob_b=row.map(lambda x:x[3]).collect()[0]
                cumprob_b= row.map(lambda x:x[2]).collect()[0]		
            runs_b = 0
            outtime_b=False
            #noprob=noprob-outprob 
            s_ns_prob_b[striker_b]= s_ns_prob_b[striker_b]-outprob_b #1-out probability,initial
            if (s_ns_prob_b[striker_b] > 0.5):
                runs_b=findruns(cumprob_b)
            elif s_ns_prob_b[striker_b] < 0.5:
                wickets_b = wickets_b+1
                striker_b = team0_2[nextbatsman_b]
                nextbatsman_b = (nextbatsman_b+1)%11
                outtime_b=True
                s_ns_prob_b[striker_b] = 1
                print("Wicket taken")
            if(outtime_b==False):
                totalruns_b = totalruns_b+runs_b
                oversc_b=oversc_b+runs_b
                if (runs_b==1 or runs_b==3):
                    striker_b,non_striker_b = non_striker_b,striker_b
            balls_b = balls_b+1
        striker_b,non_striker_b = non_striker_b,striker_b                
        bowlr_b = team1_2[count_b]                    
        count_b = (count_b+1)%11
        overs_b = overs_b+1
        print("Over No:{} \t Runs Scored in over:{}".format(overs_b, oversc_b))
    print("Total Runs:"+str(totalruns_b)+"\t Wickets Taken:"+str(wickets_b))
    print("-------------------------------------------------------------------------") 
    if(totalruns_a> totalruns_b):
        print("Team 1 Won!\n")
    elif(totalruns_b> totalruns_a):
        print("Team 2 Won!\n")
    else:
        print("Tie")
    #print(cluster)
    
    sc.stop()
    
            
            
        
     
    
    
