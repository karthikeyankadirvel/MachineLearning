# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 22:22:54 2019

@author: karth
"""

import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
#%%

data=pd.read_csv(r"C:\Users\karth\JupyterProjects\clusterdata.csv")
#%%
data.describe()
'''
             one        two
count  87.000000  87.000000
mean    0.481660   0.493573
std     0.281038   0.271722
min     0.001752   0.000516
25%     0.216059   0.230264
50%     0.550424   0.527104
75%     0.691850   0.682338
max     0.979188   0.991122

'''
#%%

def euclidean_distance(one,two):
    sq_distance=0
    for i in range(len(one)):
        sq_distance=(one[i]-two[i])**2
    ed=sqrt(sq_distance)
    return ed

#%%

#%%
euclidean_distance(data['one'],data['two'])
''' 0.2419919577935824'''

#%%

np.random.seed(10)
k=3
centroid={i+1:[np.random.rand(),np.random.rand()] for i in range(k)}

#%%

plt.scatter(data['one'],data['two'],color='k')
col_map={1:'r',2:'y',3:'b'}
for c in centroid.keys():
    plt.scatter(*centroid[c],color=col_map[c])
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

#%%
def distance(one,two,cen):
    dis=[]
    
    for i in range(len(one)):
        sq_distance1=(one[i]-cen[0])**2
        sq_distance2=(two[i]-cen[1])**2
        d=sq_distance1+sq_distance2
        dis.append(np.sqrt(d))
    
    return dis
def assigment(data,centroid):
    for c in centroid.keys():
        data['Distance_from_{}'.format(c)]=distance(data['one'],data['two'],centroid[c])
    cen_distance_col=['Distance_from_{}'.format(i) for i in centroid.keys()]

    data['closest']=data.loc[:,cen_distance_col].idxmin(axis=1)
    data['closest']=data['closest'].map(lambda x: int(x.lstrip("Distance_from_")))
    data['color']=data['closest'].map(lambda x: col_map[x])
    return data
            
#%%
d=assigment(data,centroid)

#%%
plt.scatter(d['one'],d['two'],color=d['color'],alpha=0.5)
for c in centroid.keys():
    plt.scatter(*centroid[c], color=col_map[c])
plt.show()
#%%
import copy
old_centroid=copy.deepcopy(centroid)

def update_centroid(k):
    for c in centroid.keys():
        centroid[c][0]=np.mean(d[d['closest']==c]['one'])
        centroid[c][1]=np.mean(d[d['closest']==c]['two'])
    return k


#%%
i=0    
while True:
    closest_centroid=d['closest'].copy(deep=True)
    centroid=update_centroid(centroid)
    d=assigment(d,centroid)
    i+=1
    if closest_centroid.equals(d['closest']):
        print(i)
        break
    if i>500:
        print(500)
        break
    
#%%
plt.scatter(d['one'],d['two'],color=d['color'],alpha=0.5)
for c in centroid.keys():
    plt.scatter(*centroid[c], color=col_map[c])
plt.show()
