# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:35:04 2019

@author: karth
"""

import numpy as np

class svm:
    def __init__(self,learning=0.001,lambda_par=0.01,n_inter=100):
        self.w=None
        self.b=None
        self.lr=learning
        self.lambda_par=lambda_par
        self.n_inter=n_inter
        
    def fit(self,x,y):
        samples,features=x.shape
        y_=np.where(y<=0,-1,1)
        
        self.w=np.zeros(features)
        self.b=0
        
        for _ in range(self.n_inter):
            for id,xv in enumerate(x):
                condition=y_[id]*(np.dot(self.w,xv)-self.b)>=1
                if condition:
                    self.w-=self.lr*(2*self.lambda_par*self.w)
                else:
                    self.w-=self.lr*(2*self.lambda_par*self.w)-(np.dot(y_[id],xv))
                    self.b-=self.lr*y+[id]
        
        
    
    
    def predict(self,x):
        output=np.dot(self.w,x)-self.b
        return output
#%%
class svm_scratch:
    def __init__(self, learning_rate=0.001, lambda_para=0.01,n_iters=10000):
        self.w=None
        self.b=None
        self.lr=learning_rate
        self.lam=lambda_para
        self.n_iters=n_iters
    def fit(self,x,y):
        y_=np.where(y<=0,-1,1)
        samp,feat=x.shape
        
        self.w=np.zeros(feat)
        self.b=0
        
        for _ in range(self.n_iters):
            for id,xv in enumerate(x):
                condition=y_[id]*(np.dot(xv,self.w)-self.b)>=1
                if condition:
                    self.w-=self.lr*(2*self.lam*self.w)
                else:
                    self.w-=self.lr*(2*self.lam*self.w-np.dot(xv,y_[id]))
                    self.b-=self.lr*y_[id]
        pass
    
    
    def predict(self,x):
        linear_output=np.dot(x,self.w)-self.b
        return np.sign(linear_output)
    #%%
x=np.array([[1,2,3],
            [3,4,5],
            [3,4,2],
            [23,33,11],
            [32,66,21]])
y=np.array([-1,-1,-1,1,1])

#%%
from sklearn import datasets
x, y =  datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
#%%
s=svm_scratch()
s.fit(x,y)
clf=s
X=x
import matplotlib.pyplot as plt
def visualize_svm():
     def get_hyperplane_value(x, w, b, offset):
          return (-w[0] * x + b + offset) / w[1]

     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)
     plt.scatter(X[:,0], X[:,1], marker='o',c=y)

     x0_1 = np.amin(X[:,0])
     x0_2 = np.amax(X[:,0])

     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

     ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
     ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
     x1_min = np.amin(X[:,1])
     x1_max = np.amax(X[:,1])
     ax.set_ylim([x1_min-3,x1_max+3])

     plt.show()
     ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')


visualize_svm()
#%%
class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)