# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:36:50 2021

@author: 24132
"""

import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


file = r'D:\浏览器下载\yolov4-pytorch-master_\yolov4-pytorch-master\2007_train.txt'

data = []
with open(file,'r')as f:
    a = f.readlines()
    a = [i.split(' ')[1:] for i in a]
    for i in a:
        for j in i:
            data.append(j)
    data = [i.split(',')[:-1] for i in data]
print(data)

def fun1(data):
    data_change = []
    for i in data:
        w,h = int(float(i[2])-float(i[0])),int(float(i[3])-float(i[1]))
        
        data_change.append([w,h])
    return data_change

# print(fun1(data))


data = np.array(fun1(data))

k_means = KMeans(n_clusters=15, random_state=10)

k_means.fit(data)

y = k_means.predict(data)
label = set(y)

plt.scatter(data[:,0],data[:,1],c=y)
plt.show()

# print(k_means.cluster_centers_)

sum_ = {}
for i in label:
    sum_[i] = 0
    
for i in y:
    sum_[i] += 1
    
    
values = list(sum_.values())
sort = sorted(sum_.values())

res = []
for i in sort[-9:]:
    res.append(values.index(i))
    
print(sum_)
print(res)
    
anchor = []
for i in range(len(res)):
    anchor.append(k_means.cluster_centers_[res[i]])
# print(anchor)


ratio = 2000/416
res = []
for i in anchor:
    res.append(list((i/ratio).astype(int)))
# print(res)

def fun2(res):
    res_area = [i[0]*i[1] for i in res]
    area_sort = sorted(res_area)
    index = []
    for i in area_sort:
        index.append(res_area.index(i))
    res_ = []
    # print(index)
    for i in index:
        res_.append(res[i])
    print(res_)
print('\nfinal result:')    
fun2(res)

