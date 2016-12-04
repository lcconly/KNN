code management:https://github.com/lcconly/KNN

Assignment for Advanced machine learning: implement a k_NN

Running Environment: Ubuntu 14.04

Library needed: numpy, scipy(see http://www.scipy.org/install.html to know how to install)

open source libary: progressbar(see https://github.com/niltonvolpato/python-progressbar)



File declaration：

/progressbar: open source to show progress and rest time

a.labels: labels of simple data

a.mtx: matrix of simple data

b.labels: labels of complex data

b.mtx: matrix of complex data

result_all.txt: accuracy of different parameter for data b

KNN.py: code for knn



running method:
(NB: put matrix file and label file in the same location with the code)

/***input to run the code***/
$ python KNN.py

/***notices to input the file of matrix end with .mtx(example in a.mtx)***/
Input the file of matrix as *.mtx:

/***notices to input the file of assigned labels end with .labels(example in b.labels)***/
Input the file of label as *.labels:

/***notices to choose Automatic calculation or manually input***/
Input "Yes" to get all accuracy of different parametr k and weighted.
  "No" to input parameter by user: Yes

/***notices to input parameter k (integer) range from 1 to 10 ***/
Input the parameter k (an integer from 1 to 10):

/***notice to input parameter weighted True to be weighted Knn and False to be unweighted Knn***/
Input True(weighted) or False(unweighted):

/***notice to input parameter to decide distance 1 as euclidean_distance and 2 as cosine_similarity***/
Input the parameter to determine similarity (1 as euclidean_distance and 2 as cosine_similarity): 

/***progressbar to show progress of 10 cross validation && accuracy***/
Progressing: <<<|########################################|>>> 100% Time: 0:00:00
accuracy: 80.0000   %

/***notice to show continue or not***/
Continue to input another K or change parameter weighted ?
(Input 'y' to continue or anything else to exit): 


