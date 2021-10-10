import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm

import numpy as np

from sklearn.svm import SVC

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


import sklearn.metrics as skm

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


np.random.seed(1047398)
########################################################################################################################################################################
########################################################################################################################################################################

data = pd.read_csv("winequality-red.csv")

y=data.quality
X=data.drop('quality', axis=1)                                                 ## Predict: Quality drop everything else as a feature 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)       ## make training set 75% and test set 25%
X_train_Copy=X_train.copy()

X_train_1=X_train.drop("pH", axis=1)                                            ## Training Set without pH 1_B_1
X_test_1=X_test.drop('pH', axis=1)

X_train_Copy["pH"]=X_train_Copy["pH"].sample(frac=0.66)                        ## Training Set without 33% of pH



Mean=X_train_Copy["pH"].mean(skipna="true")                                    ##Replace NAN values with the average of the column 
X_train_2=X_train_Copy.fillna(Mean)                                            ## 1_B_2         

########################################################################################################################################################################
########################################################################################################################################################################
## Logistic Regression 

LogReg_train_wpH=X_train_Copy.dropna(axis=0)                                   ## Training Set with pH (791,11)
LogReg_train_X=LogReg_train_wpH.drop("pH", axis=1)                             ##Train Set without ph (791,10)
LogReg_train_Y=(LogReg_train_wpH['pH'])                                        ##Train set y normal (791,1)
LogReg_train_Y_C=pd.cut(LogReg_train_Y, bins=5, labels=np.arange(5), right=False)  ## Train set with classes 

L=pd.cut(LogReg_train_Y, bins=5, labels=np.arange(5), right=False , retbins=True)    ##Return start and end point for each class in a tuple 


LogReg_test_X=X_train_Copy[pd.isnull(X_train_Copy['pH'])]                      ##Test Y without 33%
LogReg_test_X=LogReg_test_X.drop("pH", axis=1)                                 ##Without pH

LogReg = LogisticRegression(multi_class='multinomial',solver='newton-cg').fit(LogReg_train_X , LogReg_train_Y_C ) ##Train regrssion

pH_Log=pd.DataFrame(LogReg.predict(LogReg_test_X))  ##Make Prediction


Spaces=L[1] ## Convert tuple to list 
Averages=[]
for i in range (0,5):
    Average=(Spaces[i]+Spaces[i+1])*0.5    
    Averages.append(Average)              ## Calculate middle value of each class 

Averages = pd.DataFrame(Averages)               ## Convert into df 

pH_Log=pH_Log.replace( {0: Averages[0][0], 1: Averages[0][1] , 2:Averages[0][2] , 3:Averages[0][3] , 4:Averages[0][4] }) ## replace each predicted class with the middle number 

LogReg_test_X=LogReg_test_X.reset_index()  ##reset index in order to unite the predivted values with x_test 
LogReg_test_X['pH']=pH_Log  ##Add ph log 
LogReg_test_X=LogReg_test_X.set_index('index') ## make new index the old index 

Logistic_Regression_Data=LogReg_train_wpH.append(LogReg_test_X)  ##append xtest with xtrain
######################################################################################################################################################################
########################################################################################################################################################################
## K_means

KM_traindata_wpH=X_train_Copy.dropna(axis=0)                    ## Train set with pH (791)
KM_traindata=KM_traindata_wpH.drop("pH", axis=1)                ## Training set without pH

KM_Predict_Data_wpH=X_train_Copy[pd.isnull(X_train_Copy['pH'])]      ##Remove Nan values
KM_Predict_Data=KM_Predict_Data_wpH.drop("pH", axis=1)               ##Training set without nan elements (KM_Test Set )

distortions = []                                                             #Find Number of Clusters 
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='random',n_init=10, max_iter=300,tol=1e-04, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

# plot
# plt.plot(range(1, 15), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()

KMeans=KMeans(n_clusters=6 , init='random', algorithm='full').fit(KM_traindata)
Labels_fit=KMeans.labels_                                                      #Get Labels from the clustering (791,1)
Predicted_Labels=KMeans.predict(KM_Predict_Data)                               ##Get labelfor Test Data (408,1) Xtest kmeans 
Predicted_Labels=pd.DataFrame(Predicted_Labels)

#Calculate Average of each class

KM_traindata_wpH["pH Pred"] = Labels_fit                                       ##Make a new collumn named ph predicted 
df0 = KM_traindata_wpH[KM_traindata_wpH['pH Pred'] == 0]                       ##store in a dataframe the items that have phpredicted is 0
Mean_Cl0=df0["pH"].mean()                                                      ##Find their average pH
df1 = KM_traindata_wpH[KM_traindata_wpH['pH Pred'] == 1]                       ##Do the same for each cluster
Mean_Cl1=df1["pH"].mean()
df2 = KM_traindata_wpH[KM_traindata_wpH['pH Pred'] == 2]
Mean_Cl2=df2["pH"].mean()
df3 = KM_traindata_wpH[KM_traindata_wpH['pH Pred'] == 3]
Mean_Cl3=df3["pH"].mean()
df4 = KM_traindata_wpH[KM_traindata_wpH['pH Pred'] == 4]
Mean_Cl4=df4["pH"].mean()
df5 = KM_traindata_wpH[KM_traindata_wpH['pH Pred'] == 5]
Mean_Cl5=df5["pH"].mean()

KM_Predict_Data=KM_Predict_Data.reset_index()                                  ##Reset Index on the Missing values 
pH=Predicted_Labels.replace( {0: Mean_Cl0, 1: Mean_Cl1 , 2:Mean_Cl2 , 3:Mean_Cl3 , 4:Mean_Cl4 , 5:Mean_Cl5}) ##Replace elements with the averages 
KM_Predict_Data["pH"] = pH                                                     ## Add ph 
KM_Predict_Data=KM_Predict_Data.set_index('index')                             ##Make new index the old index

KM_data1=KM_traindata_wpH.drop("pH Pred", axis=1)                              ##Remove the collumn of the clusters     
KM_data= KM_data1.append(KM_Predict_Data)                                      ##Add the 66% of original xtrain


###########################################################################################################################

clf=SVC(C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
#clf=SVC(kernel = 'linear', random_state = 0, decision_function_shape='ovr')
clf.fit(KM_data,y_train)

y_pred=clf.predict(X_test) ## Predict the wine quality for the test set        ##PREDICTING



############################################################################### EVALUATION ################################################################################
###########################################################################################################################################################################




def test(xtrain , xtest , ytrain ,ytest):

    clf.fit(xtrain,ytrain)
    y_pred=clf.predict(xtest) ## Predict the wine quality for the test set        
    f1_score = skm.f1_score(ytest , y_pred, average='micro')                      
    print("F1 score is: " , f1_score*100 , "%")
    recall_score= skm.recall_score(ytest,y_pred, average='micro')                 
    print("Recall score is: " , recall_score*100 , "%")
    precission = skm.precision_score(ytest , y_pred, average='micro')             
    print("Precision score is: " , precission*100 , "%\n")
    return

print("Data Without pH \n")
No_pH=test(X_train_1, X_test_1 ,y_train ,y_test )
print("Data Mean pH \n")
Mean_pH=test(X_train_2, X_test , y_train ,y_test)
print("Logistic Regression Data\n")
Logistig_Regression=test(Logistic_Regression_Data, X_test , y_train ,y_test)
print("K Means data\n")
K_Means=test(KM_data, X_test , y_train ,y_test)

