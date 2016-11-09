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

MAX_NUM=99999#define max num for divied by zero
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
    return numpy.sqrt(numpy.sum(numpy.square(array1.todense()-array2.todense())))
            

#define function to get k nearest ones, it returns the assigned label
#k is the k nearest neighbours, matrix is training matrix, arry is un-labeled array
#if weighted is True, the result is weighted KNN. Otherwise, it will be wigheted KNN
def k_NN(k,matrix,arr,weighted,label_list):
    distance_dict={}
#    time1=time.time()
    for i in range(matrix.shape[0]):
#        distance_dict.setdefault(i,euclidean_distance(matrix.getrow(i).todense(),array.todense()))
        distance_dict.setdefault(i,euclidean_distance(matrix.getrow(i),arr))
#    print("time: %f\n"%(time.time()-time1))
    #reserverse sorted the distance dictionary
    sorted_dict= sorted(distance_dict.items(), key=lambda d:d[1], reverse = False)
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
        sorted_weighted_count_dict= sorted(weight_count_dict.items(), key=lambda d:d[1], reverse = True)
        result_label=sorted_weighted_count_dict[0][0]
    return result_label

#ten cross validation to calculate the accuracy
def ten_cross_validation(k,weighted,matrix,label_list):
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
                temp_matrix=vstack([temp_matrix,matrix.getrow(j)])
#        for j in range(i*each_fold_num,end_point):
#            temp_matrix=csr_matrix()
#            for m in temp_matrix[random_total[j]].nonzero()[1]:
#                print(m)
#                numpy.delete(temp_matrix,[random_total[j],m],None) 
#        mmwrite("temp.mtx",temp_matrix)

        #define as csr_matrix to reduce time and space cost
        temp_matrix=csr_matrix(temp_matrix)
        #calculate accuracy
        for j in range(i*each_fold_num,end_point):
            if label_list[random_total[j]]==k_NN(k,temp_matrix,matrix.getrow(random_total[j]),weighted,new_label_list):
                count_accuracy=count_accuracy+1
            pbar.update(j)
        #print("accuracy of fold number %d : %-10.4f%%\n"%(i,100*count_accuracy/(end_point-i*each_fold_num)))
        sum_accuracy=sum_accuracy+count_accuracy/(end_point-i*each_fold_num)
    pbar.finish()
    return sum_accuracy/10
#print(mtx)
#result=k_NN(3,mtx,csr_matrix([5.0,8.0]).getrow(0),False,label_list)
#print(result)
is_continue="y"
while is_continue=="y":
    #input parameter k which range from 1 to 10
    k=input("Input the parameter k (an integer from 1 to 10): ")
    while k.isdigit()==False or isinstance(int(k),int)==False or int(k)>10 or int(k)<1:
        k=input("Error k, Input k (an integer from 1 to 10) again: ")
    weighted=input("Input True(weighted) or False(unweighted): ")
    while not (weighted=="False" or weighted=="True"):
        weighted=input("Error input, Input agian [True(weighted) or False(unweighted)]: ")

    #calculate accuracy
    accuracy=ten_cross_validation(int(k),weighted,mtx,label_list)
    print("accuracy: %-10.4f%%"%(accuracy*100))
    is_continue=input("\n\n\nContinue to input another K or change parameter weighted ?"+ 
            "\n(Input 'y' to continue or anything else to exit): ")
