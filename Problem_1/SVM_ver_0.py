import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC
import sklearn.metrics as skm
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

np.random.seed()
########################################################################################################################################################################
########################################################################################################################################################################

data = pd.read_csv("winequality-red.csv")                                      ## IMPORT DATA
print("Wine Quality Prediction  SVM")


y=data.quality
X=data.drop('quality', axis=1)                                                 ## Predict: Quality drop everything else as a feature 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)       ## make training set 75% and test set 25%



############################################################################### TRAINING & PREDICTING ###############################################################################
########################################################################################################################################################################

clf=SVC()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test) ## Predict the wine quality for the test set        ##PREDICTING



############################################################################### EVALUATION ################################################################################
###########################################################################################################################################################################


f1_score = skm.f1_score(y_test , y_pred, average='micro')                      ##F1 SCORE
print("F1 score is: " , f1_score*100 , "%\n")

recall_score= skm.recall_score(y_test,y_pred, average='micro')                 ##RECALL SCORE
print("Recall score is: " , recall_score*100 , "%\n")

precission = skm.precision_score(y_test , y_pred, average='micro')             ##PRECISION SCORE
print("Precision score is: " , recall_score*100 , "%\n")

############################################################################################################################################################################

param_grid = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01],'kernel': ['rbf', 'linear', 'sigmoid'] , 'decision_function_shape' :['ovr']}


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
print('\n')
print("Best Patameters: \n" , grid.best_estimator_)

SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, ## Grid Search Result
     decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False)

clf=SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test) ## Predict the wine quality for the test set        ##PREDICTING

print("Optimal Parameters SVM \n")
f1_score = skm.f1_score(y_test , y_pred, average='micro')                      ##F1 SCORE
print("F1 score is: " , f1_score*100 , "%\n")

recall_score= skm.recall_score(y_test,y_pred, average='micro')                 ##RECALL SCORE
print("Recall score is: " , recall_score*100 , "%\n")

precission = skm.precision_score(y_test , y_pred, average='micro')             ##PRECISION SCORE
print("Precision score is: " , recall_score*100 , "%\n")
