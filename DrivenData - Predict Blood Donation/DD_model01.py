# DrivenData Blood Donation Prediction Challenge

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, 1:5].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:, 1::].values

# EDA
# Seaborn plots - work only on Dataframes
sns.pairplot(train.iloc[:, 1:5])
''' It is observed that No. of Donations and Volume Donated are highly correlated
    Hence, removing one of the correlated variables - Volume Donated'''
X_train = np.delete(X_train, 2, axis = 1)
X_test = np.delete(X_test, 2, axis = 1)

# 2-D plot showing class labels
from matplotlib.colors import ListedColormap
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j, 1], X_train[y_train == j, 2],
                c = ListedColormap(('red', 'green'))(i), label = j)
# 3-D plot showing class labels
import pylab
from mpl_toolkits.mplot3d import Axes3D
fig = pylab.figure()
ax = Axes3D(fig)
for c, j in [('r', 0), ('b', 1)]:
    ax.scatter(X_train[y_train == j, 0], X_train[y_train == j, 1], X_train[y_train == j, 2], c = c)
plt.show()


# Adding new feature : Donation frequency
don_freq_train = ((X_train[:, 2] - X_train[:, 0]).astype(float) / X_train[:, 1]).reshape(576,1) 
don_freq_test = ((X_test[:, 2] - X_test[:, 0]).astype(float) / X_test[:, 1]).reshape(200,1)
X_train = np.append(X_train, don_freq_train, axis = 1)
X_test = np.append(X_test, don_freq_test, axis = 1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Model 1 : Regularized Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear')
classifier1.fit(X_train,y_train)

# Model 2 : SVM Classification
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'poly', degree = 5, probability = True, C = 50)
classifier2.fit(X_train, y_train)

# Model 3 : Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit(X_train, y_train)

# Model 4 : Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier4 = RandomForestClassifier(n_estimators = 300, criterion = "entropy")
classifier4.fit(X_train, y_train)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
CV_accuracy = accuracies.mean()
CV_std = accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [0.005, 0.01, 0.02, 0.05, 0.1]}
             ]
'''{'C' : [50, 75, 100],
               'kernel' : ['poly'], 
               'degree' : [7,9]}'''
'''{'C' : [0.05, 0.1, 0.5, 1, 5, 10, 50],
               'kernel' : ['rbf', 'linear',]}'''
'''{'n_estimators' : [100, 101, 300, 501]}  '''

grid_search = GridSearchCV(estimator = classifier1, 
                           param_grid = parameters,
                           scoring = "neg_log_loss",
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_

# Predicting the Test Set results
y_pred1 = classifier1.predict_proba(X_test)[:,1]
y_pred2 = classifier2.predict_proba(X_test)[:,1]
y_pred3 = classifier3.predict_proba(X_test)[:,1]
y_pred4 = classifier4.predict_proba(X_test)[:,1]

# Creating predictions from ensemble models
ensemble1 = (0.6*y_pred1) + (0.4*y_pred4)

# Writing the results to a csv file
np.savetxt('results.csv', ensemble1)
