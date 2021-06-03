import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


file = r'/home/yuzhengbo/下载/yolov4-pytorch-master/input/ground-truth'

list_dir = os.listdir(file)
data = []
for i in list_dir:
    with open(file+'/'+i,'r')as f:
        a = f.readlines()
        a = [i.split('\n')[0] for i in a]
        a = [i.split(' ')[1:] for i in a]
        data.extend(a)
        
# print(data)

def fun1(data):
    data_change = []
    for i in data:
        w,h = int(float(i[2])-float(i[0])),int(float(i[3])-float(i[1]))
        
        data_change.append([w,h])
    return data_change

data = np.array(fun1(data))

k_means = KMeans(n_clusters=9, random_state=10)

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
    
# print(res)
    
anchor = []
for i in range(len(res)):
    anchor.append(k_means.cluster_centers_[i])
print(anchor)

# ratio = 2000/1664

# for i in anchor:
#     print((i/ratio).astype(int))


