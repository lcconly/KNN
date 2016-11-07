#!/usr/bin/env python
# coding=utf-8
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
    match=re.match(path,mtx_file_name)

#using scipy.sparse to open file and store the data
mtx_file_location=os.path.join(__location__,mtx_file_name)
mtx=csr_matrix(mmread(mtx_file_location))
#print(mtx.shape)#get the shape of matrix

#define function to calculate the distance of two document, using numpy
def euclidean_distance(array1, array2):
    return numpy.sqrt(numpy.sum(numpy.square(array1-array2)))
#define function to get k nearest ones
def k_NN(k,matrix,array):
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
#            print(str(item[0])+"  "+str(item[1]))
            result_dict.setdefault(item[0],item[1])
        else:
            break
        count=count+1
#    print(result_dict)
    return result_dict
arr=[]
k_NN(7,mtx,arr)

