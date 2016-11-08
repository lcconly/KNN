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
from scipy.sparse import csr_matrix
from scipy.io.mmio import mminfo,mmread,mmwrite
#####get upper path of the code
__location__=os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))

#Input matrix filename && judge wether it is follow the rule *.mtx
mtx_file_name=input("Input the file of matrix as *.mtx: ")
pattern=re.compile(r'/\.mtx')
match=re.match(pattern,mtx_file_name)
while match==False:
    mtx_file_name=input("Error file name, Input again: ")
    match=re.match(pattern,mtx_file_name)

#using scipy.sparse to open file and store the data
mtx_file_location=os.path.join(__location__,mtx_file_name)
mtx=csr_matrix(mmread(mtx_file_location))
#print(mtx.shape)#get the shape of matrix

#open Labels files and store the data in label_dict as a dictionary
label_file_name=input("Input the file of label as *.labels: ")
pattern=re.compile(r'/\.labels')
match=re.match(pattern,label_file_name)
while match==False:
    label_file_name=input("Error label name, Input again: ")
    match=re.match(pattern,label_file_name)
label_file_path=os.path.join(__location__,label_file_name)
label_file=open(label_file_path)
label_dict=dict()
for line in label_file.readlines():
    label=line.split(',')
    if len(label)==2:
        label_dict[label[0]]=label[1]

#define function to calculate the distance of two document, using numpy
def euclidean_distance(array1, array2):
    return numpy.sqrt(numpy.sum(numpy.square(array1-array2)))

#define function to get k nearest ones, it returns the assign label
#k is the k nearest neighbours, matrix is training matrix, arry is un-labeled array
#if weighted is True, the result is weighted KNN. Otherwise, it will be in wigheted KNN
def k_NN(k,matrix,array,weighted):
    array=matrix.getrow(0)
    distance_dict={}
    for i in range(matrix.shape[0]):
        distance_dict.setdefault(i,euclidean_distance(matrix.getrow(i).todense(),array.todense()))
    #reserverse sorted the distance dictionary
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
        for item in result_dict.items():
            if label_dict[str(item[0])] not in count_dict.keys():
                count_dict[label_dict[str(item[0])]]=1
            else:
                count_dict[label_dict[str(item[0])]]=count_dict[label_dict[str(item[0])]]+1
        sorted_count_dict= sorted(count_dict.items(), key=lambda d:d[1], reverse = True)
        result_label=sorted_count_dict[0][0]
    else:   
        #weighted KNN
        weight_count_dict={}
        for item in result_dict.items():
            if label_dict[str(item[0])] not in weight_count_dict.keys():
                weight_count_dict[label_dict[str(item[0])]]=1/item[1]
            else: 
                weight_count_dict[label_dict[str(item[0])]]=weight_count_dict[label_dict[str(item[0])]]+1/item[1]
        sorted_weighted_count_dict= sorted(weight_count_dict.items(), key=lambda d:d[1], reverse = True)
        result_label=sorted_weighted_count_dict[0][0]
    return result_label

arr=[]
result=k_NN(7,mtx,arr,True)
print(result)
