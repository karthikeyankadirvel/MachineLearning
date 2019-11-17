# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:53:27 2019

@author: karth
"""
import numpy as np
import math
import pandas as pd
from collections import Counter
import os
os.chdir(r"C:\Users\karth\JupyterProjects\MachineLearning")
#%%
data=pd.read_excel("data.xlsx")
x=data[['Check_pain', 'Leg_pain', 'Kideney_pain', 'Head_ache', 'sweating']]
y=data['Heart_disease']
#%%
def gini(class_prob):
    g=1-sum([p**2 for p in class_prob])
    return g
#%%
c_p=[0.0025,0.2,0.1]
#%%
def class_probability(labels):
    total_count=len(labels)
    prob={l:count/total_count for l,count in zip(Counter(labels).keys(),Counter(labels).values())}
    return prob
#%%
lab=[0,1,0,0,0,1,1,0,0]
class_probability(y)
#%%

def gini_index(labels):
    clas_prob=class_probability(labels)
    g_i=gini(clas_prob.values())
    return g_i

#%%
def gini_variable(x,y):
    class_prob_vs=[]
    x_subsets_v=[]
    y_subsets_v=[]
    gini_v_s={}
    for v in x:
        class_prob_v= class_probability(x[v])
        class_prob_vs.append(class_prob_v)
        x_subsets=[]
        leaf_gini=[]
        y_subsets=[]
        for a in x[v].unique():
            x_subset=x[x[v]==a]
            x_subsets.append(x_subset)
            y_v=y[x_subset.index]
            y_subsets.append(y_v)
            l_g=gini_index(y_v)
            leaf_gini.append(l_g)
    
        x_subsets_v.append(x_subsets)
        y_subsets_v.append(y_subsets)
        gini_v=sum([g*p for g,p in zip(leaf_gini,class_prob_v.values())])
        gini_v_s.update({v:gini_v})
    return gini_v_s,x_subsets_v,y_subsets_v
#%%
gini_values,subset,y_sub=gini_variable(x,y)  
#%%

key_min=min(gini_values.keys(),key=(lambda k:gini_values[k]))
for i in range(len(x[key_min].unique())):
    x_u=subset[key_min][i]
    y_u=y_sub[key_min][i]
    
    
    
    

        
    
    