from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Ayameta data
iris = datasets.load_iris()

# Example 
X = iris.data[:, [2, 3]]
# print('X = ', X)

# Class Label
Y = iris.target
# print('Y = ', Y)

# Split data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
# print('X_train = ', X_train)
# print('X_test = ', X_test)
# print('Y_train = ', Y_train)
# print('Y_test = ', Y_test)

# Standarize Data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
# print('X_train_std = ', X_train_std)
X_test_std = sc.transform(X_test)
# print('X_test_std = ', X_test_std)

# Linear SVM
model = SVC(kernel='linear', random_state=None)

# Learn Model
model.fit(X_train_std, Y_train)

# Acc of Model 
pred_train = model.predict(X_train_std)
accuracY_train = accuracy_score(Y_train, pred_train)
print('Accuracy for Training Data: %.2f' % accuracY_train)

# Test Data Acc 
pred_test = model.predict(X_test_std)
accuracY_test = accuracy_score(Y_test, pred_test)
print('Accuracy for Test Data: %.2f' % accuracY_test)

# Classification Result Display 
plt.style.use('ggplot')

X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))

fig = plt.figure(figsize=(13, 8))
plot_decision_regions(X_combined_std, Y_combined, clf=model)
plt.show()
