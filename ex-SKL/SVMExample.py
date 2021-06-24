# -*- coding: utf-8 -*-
#%matplotlib inline

# Import necessary data
from sklearn import datasets
import numpy as np

# Ayameta data
iris = datasets.load_iris()

# Example 
X = iris.data[:, [2, 3]]
# print('X = ', X)
# Class Label 
y = iris.target
# print('y = ', y)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None )

# print('X_train = ', X_train)
# print('X_test = ', X_test)
# print('y_train = ', y_train)
# print('y_test = ', y_test)
# Standarize Data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
# print('X_train_std = ', X_train_std)
X_test_std = sc.transform(X_test)
# print('X_test_std = ', X_test_std)

from sklearn.svm import SVC
# Linear SVM 
model = SVC(kernel='linear', random_state=None)

# Learn MOdel 
model.fit(X_train_std, y_train)


#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression(random_state=None)


from sklearn.metrics import accuracy_score

# Acc of Model 
pred_train = model.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print('Accuracy for Training Data: %.2f' % accuracy_train)


# Test Data Acc 
pred_test = model.predict(X_test_std)
accuracy_test = accuracy_score(y_test, pred_test)
print('Accuracy for Test Data: %.2f' % accuracy_test)


# Classification Result Display 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
plt.style.use('ggplot')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

fig = plt.figure(figsize=(13, 8))
plot_decision_regions(X_combined_std, y_combined, clf=model,  res=0.02)
plt.show()
