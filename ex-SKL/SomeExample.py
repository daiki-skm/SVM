# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:42:56 2019

@author: che
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:06:02 2019

@author: che
"""

import pandas as pd
#import matplotlib
import re
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

#from matplotlib import pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)



#news= pd.read_csv('testHW2.csv')
#
#data = news.values[:,1:]
#Labels = news.values[:,0]
#import numpy as np

#import random

data=np.genfromtxt('testHW2.csv',delimiter=',')
print (type(data))

data= np.random.permutation(data)
print (data)
X = data[:,1:]
print('X = ')
print(X)
y0 =  [int(i) for i in data[:,0]]
y =  np.array(y0)
print('y = ')
print(y)

#print(news, data, Labels)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 
X_train= X[:14]
X_test=X[14:]
y_train=y[:14]
y_test=y[14:]

#



from sklearn.svm import SVC

#Binarizing the training and test labels

from sklearn import preprocessing
from sklearn import utils

#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(y_train)
#print(training_scores_encoded)
#print(utils.multiclass.type_of_target(y_train))
#print(utils.multiclass.type_of_target(y_train.astype('int')))
#print(utils.multiclass.type_of_target(training_scores_encoded))

from sklearn.metrics import accuracy_score
#predicting_scores_encoded = lab_enc.fit_transform(y_test)
#print(utils.multiclass.type_of_target(predicting_scores_encoded))
#clf = SVC(gamma = 'auto')
clf = SVC(kernel='linear', random_state=None)
clf.fit(X_train, y_train)
ps=clf.predict(X_test)
#
#print(clf.score(X_test, y_test))
print(accuracy_score(y_test, ps))
'''
for name, model in models:
    model = model.fit(X_train, training_scores_encoded)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(predicting_scores_encoded, y_pred)*100))
    #print("%s -> ClfRprt: %%%.2f" % (name,metrics.classification_report(predicting_scores_encoded, y_pred)*100))
''' 

















                                                                                                                           # https://stackabuse.com/text-classification-with-python-and-scikit-learn/
