# DrivenData Blood Donation Prediction Challenge - Neural Nets, XGBoost and others

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
sns.lmplot('Months since Last Donation', 'Number of Donations' , data = train, fit_reg = False)
sns.pairplot(train.iloc[:, 1:5])
''' It is observed that No. of Donations and Volume Donated are highly correlated
    Hence, removing one of the correlated variables - Volume Donated'''
X_train = np.delete(X_train, 2, axis = 1)
X_test = np.delete(X_test, 2, axis = 1)

# Adding new feature : Donation frequency
don_freq_train = ((X_train[:, 2] - X_train[:, 0]).astype(float) / X_train[:, 1]).reshape(576,1) 
don_freq_test = ((X_test[:, 2] - X_test[:, 0]).astype(float) / X_test[:, 1]).reshape(200,1)
X_train = np.append(X_train, don_freq_train, axis = 1)
X_test = np.append(X_test, don_freq_test, axis = 1)


# Applying KMeans to form suitable clusters and add cluster_id as a feature

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
X = np.append(X_train, X_test, axis = 0)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 25, max_iter = 300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Applying k-means
kmeans = KMeans(n_clusters=5, n_init = 25, max_iter = 300, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
''' Since this is a nominal feature, we perform OHE on it'''
# One Hot Encoding for the new cluster_id feature
from sklearn.preprocessing import OneHotEncoder
y_kmeans = y_kmeans.reshape(np.size(y_kmeans), 1)
onehotencoder = OneHotEncoder(categorical_features = [0])
y_kmeans = onehotencoder.fit_transform(y_kmeans).toarray()
# Removing one column to avoid dummy variable trap
y_kmeans = np.delete(y_kmeans, -1, axis = 1)
# Adding a new feature : 'cluster_id' in the training and test set
cid_train = y_kmeans[0:np.size(X_train,0),:]
cid_test = y_kmeans[np.size(X_train,0)::,:] 
X_train = np.append(X_train, cid_train, axis = 1)
X_test = np.append(X_test, cid_test, axis = 1)

''' It is observed that the 'cluster_id' feature has very little importance value while 
    running RF model. For other models also, it doesn't improve the score. Hence, this 
    seems to be an insignificant addition and should be dropped ''''


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Model 1 : XGBoost
from xgboost import XGBClassifier
classifier1 = XGBClassifier(max_depth = 3, learning_rate = 0.02,
                            n_estimators = 100, objective = "reg:linear", 
                            gamma = 0, base_score = 0.5, reg_lambda = 2, subsample = 0.2,
                            colsample_bytree = 0.6)

classifier1.fit(X_train, y_train, eval_metric = "logloss")

# Model 2 : ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm

# Initialising the ANN
classifier_NN = Sequential()

# Adding the input layer and one hidden layer
classifier_NN.add(Dense(units = 8, kernel_initializer = 'glorot_uniform', activation = 'relu', 
                     input_dim = 4))

# Adding the second hidden layer
classifier_NN.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))

# Adding the output layer
classifier_NN.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
'''sgd = SGD(lr = 0.005, momentum = 0.9, decay = 0.2, nesterov = False)'''
classifier_NN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
history = classifier_NN.fit(X_train, y_train, validation_split = 0, batch_size = 20, 
                            epochs = 150)

# Learning curves
'''# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10)
CV_accuracy = accuracies.mean()
CV_std = accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'reg_lambda' : [0.1, 0.5, 1, 2, 5, 10, 30, 50],
               'n_estimators' : [50, 75, 100, 300, 301],
               'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.5, 1],
               'max_depth' : [3, 4, 5, 6, 8, 10],
               'subsample' : [0.1, 0.2, 0.5, 0.75, 0.85, 1]}
             ]

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
y_pred_NN = classifier_NN.predict(X_test)

# Creating predictions from ensemble models
ensemble1 = ((0.25*y_pred1) + (0.75*y_pred_NN).T).T

# Writing the results to a csv file
np.savetxt('results.csv', ensemble1)
