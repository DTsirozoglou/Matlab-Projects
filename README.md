Statistical Methods for Machine Learning:

  Built  an affine model of data provided by the Sunspot Index Data Center (SIDC) 
by applying a linear and a non-linear classification method. The goal was to build 
a model so as to find a mapping for predicting the number of sunspots  based on 
previous observations.

  Performed  binary  classification  using  support  vector  machines  (SVMs)  to  data 
from The Sloan Digital Sky Survey.

  Performed  principal component analysis of the training data available from the 
well  known  UCI  benchmark  repository  (Frank  and  Asuncion,  2010, 
http://archive.ics.uci.edu/ml/datasets/seeds).  Performed  3-means clustering of 
the same training data. Classified the 3 image classes using discriminant analysis 
(LDA) and the Nearest neighbor classification.

In order to run we need the files: 

1.	The code is structured such that there is one main file that we can run to reproduce all the results presented in our report. This main file is Exam which calls all the below files:
2.	
2.	The datasets for the First case:                                                               sunspotsTestStatML.dt, sunspotsTrainStatML.dt

3.	The datasets for the Second case: quasarsStarsStatMLTest.dt,quasarsStarsStatMLTrain.dt

4.	The datasets for the Third case: 				                            seedsTest.dt, seedsTrain.dt

5.	The created Matlab functions used for the 1st case:
 linearRegr and Neural

6.	The SVM software LIBSVM (which can be downloaded  from http://www.csie.ntu.edu.tw/~cjlin/libsvm)  used for the 2nd case: 
svmpredict2  and svmtrain2  (As we did in the 3rd Assignement, after running the make command to compile the SVM we rename the files created from svmtrain to svmtrain2 and from svmpredict to svmpredict2 so as to avoid confusion between the matlab function svmtrain!)

7.	The created Matlab functions used for the 3rd  case: 
PCAnalysis , KMeansCluster,KNN and LDAnalysis

The code is for matlab and the questions are sorted one below the other. There are “clear all” commands in the beginning of each case. It is recommended to run all of the three Cases at once, or you can also run each Case separately 
