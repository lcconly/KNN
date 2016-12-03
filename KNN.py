#!/usr/bin/env python
# coding=utf-8
#*************************************************************************
#	> File Name: KNN.py
#	> Author: liucheng
#	> Mail: lcconlycs@gmail.com 
#       > Created Time: Mon 07 Nov 2016 11:48:44 AM PST
#*************************************************************************
import os,re
import numpy
import math
import random
import time
import copy
from scipy.sparse import csr_matrix,vstack
from scipy.io.mmio import mminfo,mmread,mmwrite
from progressbar import Bar,Percentage,ETA,ProgressBar

MAX_NUM=999999#define max num for divied by zero
#####get upper path of the code
__location__=os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
os.system("clear")
#Input matrix filename using Regular Expressions to match
mtx_file_name=input("Input the file of matrix as *.mtx: ")
pattern=re.compile(r'.*\.mtx')
match=re.match(pattern,mtx_file_name)
mtx_file_location=os.path.join(__location__,mtx_file_name)
#judge wether filename is follow the rule *.mtx or exist
while match==None or os.path.isfile(mtx_file_location)==False:
    mtx_file_name=input("Error file name or file does not exist, Input again: ")
    match=re.match(pattern,mtx_file_name)
    mtx_file_location=os.path.join(__location__,mtx_file_name)

#using scipy.sparse to open file and store the data
mtx=csr_matrix(mmread(mtx_file_location),dtype=float)

#Input label filename using Regular Expressions to match
label_file_name=input("Input the file of label as *.labels: ")
pattern=re.compile(r'.*\.labels')
match=re.match(pattern,label_file_name)
label_file_path=os.path.join(__location__,label_file_name)
#judge wether filename is follow the rule *.labels or exist
while match==None or os.path.isfile(label_file_path)==False:
    label_file_name=input("Error label name or file does not exist, Input again: ")
    match=re.match(pattern,label_file_name)
    label_file_path=os.path.join(__location__,label_file_name)
label_file=open(label_file_path)
label_list=list()
for line in label_file.readlines():
    label=line.split(',')
    if len(label)==2:
        label_list.append(label[1])

#define function to calculate the euclidean distance of two array, using numpy
def euclidean_distance(array1, array2):
#    col_arr1=list()
#    col_arr2=list()
#    sum=0
#    for col1 in array1.nonzero():
#        col_arr1.append(col1)
#    for col2 in array2.nonzero():
#        col_arr2.append(col2)
#    for i in col_arr1[1][:]:
#        if i in col_arr2[1]:
#            sum=sum+(array1[0,i]-array2[0,i])*(array1[0,i]-array2[0,i])
#        else:
#            sum=sum+(array1[0,i])*(array1[0,i])
#    for i in col_arr2[1][:]:
#        if i not in col_arr1[1]:
#            sum=sum+array2[0,i]*array2[0,i]
#    return math.sqrt(sum)
    return 1.0/(1.0+numpy.sqrt(numpy.sum(numpy.square(array1.todense()-array2.todense()))))

#define similarity
def cosine_similarity(array1,array2):
    in1=numpy.mat(array1.todense())
    in2=numpy.mat(array2.todense())
    return 0.5+0.5*float(in1 * in2.T )/(numpy.linalg.norm(in1)*numpy.linalg.norm(in2))
            


#k is the k nearest neighbours, matrix is training matrix, arry is un-labeled array
#if weighted is True, the result is weighted KNN. Otherwise, it will be wigheted KNN
#parameter to determine euclidean_distance or cosine_similarity
def k_NN(k,matrix,arr,weighted,label_list,parameter):
    distance_dict={}
    for i in range(matrix.shape[0]):
        ########distance as euclidean_distance
        if parameter is 1:
            distance_dict.setdefault(i,euclidean_distance(matrix.getrow(i),arr))
        ########distance as cosine_similarity 
        if parameter is 2:
            distance_dict.setdefault(i,cosine_similarity(matrix.getrow(i),arr))
    sorted_dict={}
    #reserverse sorted the distance dictionary
    if parameter is 1:
        sorted_dict= sorted(distance_dict.items(), key=lambda d:d[1], reverse = False)
    if parameter is 2:
        sorted_dict= sorted(distance_dict.items(), key=lambda d:d[1], reverse = True) 
    count=0
    result_dict={}
    for item in sorted_dict[:]:
        if count<k:
            result_dict.setdefault(item[0],item[1])
        else:
            break
        count=count+1
    if not weighted==True:
        #unweighted KNN
        count_dict={}
        #count the item and merge 
        for item in result_dict.items():
            #print("%d : %s"%(item[0],label_list[item[0]]))
            if label_list[item[0]] not in count_dict.keys():
                count_dict[label_list[item[0]]]=1
            else:
                count_dict[label_list[item[0]]]=count_dict[label_list[item[0]]]+1
        #sort the result to get the result
        sorted_count_dict= sorted(count_dict.items(), key=lambda d:d[1], reverse = True)
        result_label=sorted_count_dict[0][0]
    else:   
        #weighted KNN
        weight_count_dict={}
        for item in result_dict.items():
            #calculate weighted and merge differnet item
            if label_list[item[0]] not in weight_count_dict.keys():
                #there are some cases that two arrays are the same.
                if not item[1]==0:
                    weight_count_dict[label_list[item[0]]]=1/item[1]
                else:
                    weight_count_dict[label_list[item[0]]] =MAX_NUM
            else: 
                if not item[1]==0:
                    weight_count_dict[label_list[item[0]]]=weight_count_dict[label_list[item[0]]]+1/item[1]
                else:
                    weight_count_dict[label_list[item[0]]]=weight_count_dict[label_list[item[0]]]+MAX_NUM
        #sort the result and get highest weighted one
        if parameter is 1:
            sorted_weighted_count_dict= sorted(weight_count_dict.items(), key=lambda d:d[1], reverse = True)
        if parameter is 2:
            sorted_weighted_count_dict= sorted(weight_count_dict.items(), key=lambda d:d[1], reverse = False)

        result_label=sorted_weighted_count_dict[0][0]
    return result_label

#ten cross validation to calculate the accuracy
def ten_cross_validation(k,weighted,matrix,label_list,parameter):
    widgets = ["Progressing:",' <<<', Bar(), '>>> ',Percentage(),' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=matrix.shape[0])
    # maybe do something
    pbar.start()
    sum_accuracy=0
    each_fold_num=int(matrix.shape[0]/10)
    #generate a random list
    random_total=[n for n in range(matrix.shape[0])]
    random.shuffle(random_total)
    #divie the test as ten folds
    for i in range(10):
        count_accuracy=0
        end_point=(i+1)*each_fold_num
        if i==9:
            end_point=matrix.shape[0]
        #get selected data as un-labeled array
        random_list=[]
        new_label_list=copy.copy(label_list)
        for j in range(i*each_fold_num,end_point):
            random_list.append(random_total[j])
        random_list.sort(reverse=True)
        for j in random_list:
            del(new_label_list[j])
        temp_matrix=None
        #using the rest data form as a matrix and set as training data
        for j in range(matrix.shape[0]):
            if j not in random_list:
                if temp_matrix is None:
                    temp_matrix=matrix.getrow(j)
                else:
                    temp_matrix=vstack([temp_matrix,matrix.getrow(j)])

        #define as csr_matrix to reduce time and space cost
        temp_matrix=csr_matrix(temp_matrix)
        #calculate accuracy
        for j in range(i*each_fold_num,end_point):
            if label_list[random_total[j]]==k_NN(k,temp_matrix,matrix.getrow(random_total[j]),weighted,new_label_list,parameter):
                count_accuracy=count_accuracy+1
            pbar.update(j)
        #print("accuracy of fold number %d : %-10.4f%%\n"%(i,100*count_accuracy/(end_point-i*each_fold_num)))
        sum_accuracy=sum_accuracy+count_accuracy/(end_point-i*each_fold_num)
    pbar.finish()
    return sum_accuracy/10

#single test
'''
temp_matrix=None
for j in range(mtx.shape[0]):
    if not j==4:
        if temp_matrix is None:
            temp_matrix=mtx.getrow(j)
        else:
            temp_matrix=vstack([temp_matrix,mtx.getrow(j)])
del(label_list[4])
print(k_NN(3,temp_matrix,mtx.getrow(4),False,label_list,1))
'''

#all parameter

'''f=open("result_sim.txt",'w')
for k in range (1,11):
    print("k=%d weighted=%s : %-10.4f%%"%(k,"True",100*(ten_cross_validation(k,True,mtx,label_list,2))),file=f)
    print("k=%d weighted=%s : %-10.4f%%"%(k,"False",100*(ten_cross_validation(k,False,mtx,label_list,2))),file=f)
f.close()
'''

#UI
is_continue="y"
while is_continue=="y":
    #input parameter k which range from 1 to 10
    k=input("Input the parameter k (an integer from 1 to 10): ")
    while k.isdigit()==False or isinstance(int(k),int)==False or int(k)>10 or int(k)<1:
        k=input("Error k, Input k (an integer from 1 to 10) again: ")
    weighted=input("Input True(weighted) or False(unweighted): ")
    while not (weighted=="False" or weighted=="True"):
        weighted=input("Error input, Input agian [True(weighted) or False(unweighted)]: ")
    #input parameter which represent euclidean_distance or cosine_similarity
    parameter=input("Input the parameter to determine similarity\n(1 as euclidean_distance and 2 as cosine_similarity): ")
    while parameter.isdigit()==False or isinstance(int(parameter),int)==False or int(parameter)>2 or int(parameter)<1:
        parameter=input("Error parameter, Input parameter (1 or 2) again: ")

    #calculate accuracy
    accuracy=ten_cross_validation(int(k),weighted,mtx,label_list,parameter)
    print("accuracy: %-10.4f%%"%(accuracy*100))
    is_continue=input("\n\nContinue to input another K or change parameter weighted ?"+ 
            "\n(Input 'y' to continue or anything else to exit): ")

