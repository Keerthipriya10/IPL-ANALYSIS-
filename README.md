# IPL-ANALYSIS-
Using Big data analytics,we have developed a Simulator for IPL matches with the help of clustering and collaborative filtering by using the past IPL statistics of players.

Introduction
Our main aim is to predict the score of the match. Here we use two methods to predict the over-by-over score :
  1)Clustering Method by choosing a random probability
  
  2)Collaborative Filtering
  
Software Dependencies:
    Apache Hadoop- HDFS
    
    Apache Spark
    
    Python 3
    
    PySpark
    
DESIGN:

Step 1:
  The main dataset that we used for prediction was taken from various sites.But there is a chance that many batsman-bowler combinations wouldn't have occured which would cause inaccuracies in prediction.Hence, we clustered batsmen and bowlers based on their attributes and similarities. Hence when we encounter a combination that didn't occur or a very rare combination, we predict the outcome based on the cluster they belong to. So to cluster players,we collected player profiles. We web-scraped the player profiles from http://www.espncricinfo.com and converted to a csv format which is easy to analyse.This csv file was loaded into HDFS. Next, we wrote a python code for clustering batsmen and bowlers using PySpark. The clustering was done using the K-Means algorithm. To optimize the value of k (the number of clusters),we plotted an elbow plot for different values of k. The visualization yielded an optimal value of 5 for batsmen and value of 4 for bowlers. The parameters used for clustering batsman – Runs, Strike Rate, Average, Number of 4s,6s,100's and 50's. The parameters used for clustering bowlers – Wickets, Economy, Average, Strike Rate.
  
 Step 2:
  Once the cluster datasets were obtained,we performed cluster vs cluster statistics using the cluster datasets which had ball by ball outcome of all matches from 2008-2017.We maintained a dataset of batsmen cluster vs bowler cluster probabilities (0s,1s,2s,3s,4s,6s,wickets). Every time a new combination or a rare combination was encountered, we found the batsman cluster number and the bowler cluster number and obtained the cluster vs cluster combination probabilities and considered those probabilities for prediction. We wrote a PySpark code to simulate an entire IPL match. To predict the outcome of a ball, we searched for the batsman-bowler combination in the player-player statistics. If not found, or is the batsman had faces less than 10 balls from that bowler we predict the clusters for the combination and obtained the probabilities for the particular cluster combination. A random number is generated and the outcome is decided based on the range of cumulative probabilities in which it falls. The outcome could be 0,1,2,3,4,6 or a wicket. This is repeated for all balls until 20 overs are up or there are no wickets left and we are not considering any extras.We output the over by over prediction to a text file.
  
Step 3:
  The second method used is collaborative filtering using Spark Mllib.Here we feed the data of player vs player cumulative probabilites to the ALS Collaborative filtering Python code which analyses the dataset.So when a new combination or a rare combination is encountered,it automatically analyses the dataset and predicts the probabilities of the combination which is then used to predict the outcome. Thus we simulate the match given players of the teams and predict the score  on the basis of the trained collaborative filtering model.
