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
#####get upper path of the code
__location__=os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))

#Input matrix filename && judge wether it is follow the rule *.mtx
mtx_file_name=input("Input the file of matrix as *.mtx: ")
pattern=re.compile(r'.*.mtx')
match=re.match(pattern,mtx_file_name)
while match==False:
    mtx_file_name=input("Error file name, Input again: ")
    match=re.match(pattern,mtx_file_name)

#using scipy.sparse to open file and store the data
mtx_file_location=os.path.join(__location__,mtx_file_name)
mtx=csr_matrix(mmread(mtx_file_location),dtype=float)

#open Labels files and store the data in label_dict as a dictionary
label_file_name=input("Input the file of label as *.labels: ")
pattern=re.compile(r'.*.labels')
match=re.match(pattern,label_file_name)
while match==False:
    label_file_name=input("Error label name, Input again: ")
    match=re.match(pattern,label_file_name)
label_file_path=os.path.join(__location__,label_file_name)
label_file=open(label_file_path)
label_list=list()
for line in label_file.readlines():
    label=line.split(',')
    if len(label)==2:
        label_list.append(label[1])

#define function to calculate the distance of two document, using numpy
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
            

#define function to get k nearest ones, it returns the assign label
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
        for item in result_dict.items():
            if label_list[item[0]] not in count_dict.keys():
                count_dict[label_list[item[0]]]=1
            else:
                count_dict[label_list[item[0]]]=count_dict[label_list[item[0]]]+1
        sorted_count_dict= sorted(count_dict.items(), key=lambda d:d[1], reverse = True)
        result_label=sorted_count_dict[0][0]
    else:   
        #weighted KNN
        weight_count_dict={}
        for item in result_dict.items():
            if label_list[item[0]] not in weight_count_dict.keys():
                if not item[1]==0:
                    weight_count_dict[label_list[item[0]]]=1/item[1]
                else:
                    weight_count_dict[label_list[item[0]]] =9999
            else: 
                if not item[1]==0:
                    weight_count_dict[label_list[item[0]]]=weight_count_dict[label_list[item[0]]]+1/item[1]
                else:
                    weight_count_dict[label_list[item[0]]]=weight_count_dict[label_list[item[0]]]+9999
        sorted_weighted_count_dict= sorted(weight_count_dict.items(), key=lambda d:d[1], reverse = True)
        result_label=sorted_weighted_count_dict[0][0]
    return result_label
def ten_cross_validation(k,weighted,matrix,label_list):
    sum_accuracy=0
    each_fold_num=int(matrix.shape[0]/10)
    random_total=[n for n in range(matrix.shape[0])]
    random.shuffle(random_total)
    for i in range(10):
        count_accuracy=0
        end_point=(i+1)*each_fold_num
        if i==9:
            end_point=matrix.shape[0]
        random_list=[]
        new_label_list=copy.copy(label_list)
        for j in range(i*each_fold_num,end_point):
            random_list.append(random_total[j])
        random_list.sort(reverse=True)
        for j in random_list:
            del(new_label_list[j])
        temp_matrix=None
        for j in range(matrix.shape[0]):
            if j not in random_list:
                temp_matrix=vstack([temp_matrix,matrix.getrow(j)])
#        for j in range(i*each_fold_num,end_point):
#            temp_matrix=csr_matrix()
#            for m in temp_matrix[random_total[j]].nonzero()[1]:
#                print(m)
#                numpy.delete(temp_matrix,[random_total[j],m],None) 
#        mmwrite("temp.mtx",temp_matrix)
        temp_matrix=csr_matrix(temp_matrix)
        for j in range(i*each_fold_num,end_point):
            if label_list[random_total[j]]==k_NN(k,temp_matrix,matrix.getrow(random_total[j]),weighted,new_label_list):
                count_accuracy=count_accuracy+1
            print("j: %d  count_accuracy: %d   each fold num: %d"%(j,count_accuracy,each_fold_num))
        sum_accuracy=sum_accuracy+count_accuracy/(end_point-i*each_fold_num)
    return sum_accuracy/10
#print(mtx)
#result=k_NN(3,mtx,csr_matrix([5.0,8.0]).getrow(0),False,label_list)
#print(result)
k=input("Input the parameter k (an integer): ")
weighted=input("Input True(weighted) or False(unweighted): ")
accuracy=ten_cross_validation(int(k),weighted,mtx,label_list)
print("accuracy: %-10.4f%%"%(accuracy*100))
