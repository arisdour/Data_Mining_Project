import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense 

np.random.seed(1047398)



stemmer = SnowballStemmer("english")  ## Sbowball / Porter 
stop = stopwords.words('english')
data = pd.read_csv("onion-or-not.csv")   ## IMPORT DATA


############## Pre Processing / Data Cleaning  ##############

X=data['text'].apply(nltk.word_tokenize)  ## Tokenization
PreP_data = pd.DataFrame(X, columns = ['Unstemmed']) 
PreP_data['Unstemmed']=pd.Series(X)
PreP_data["Stemmed"] = X.apply(lambda x: [stemmer.stem(y) for y in x])         ## Stem every word.
PreP_data["StopWords"] = PreP_data['Stemmed'].apply(lambda x: [item for item in x if item not in stop])  ## Remove stopwords

PreP_data["StopWords"]=[" ".join(review) for review in PreP_data["StopWords"].values]     # Calculate TF-IDF Weights
v = TfidfVectorizer()
x = v.fit_transform(PreP_data["StopWords"])  ## TF-IDF Sparce Matrix 

################################### Training Sets #############################

df = pd.DataFrame(x.toarray())   ## Convert Sparce to DF 
df["label"]=data["label"]
y=df.label
X=df.drop('label', axis=1)                                                 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)     ## Create training and test sets 


################################### Neural Network ############################
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = Sequential()
model.add(Dense(30, input_dim=16844, activation='relu', ))
model.add(Dense(15, activation='relu', ))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc',f1_m,precision_m, recall_m] )
history=model.fit(X_train , y_train , epochs=32, batch_size=256) 



##################################### Metrics #################################


loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print("\n Metrics \n ")
print('Loss: %.2f' % (loss))

print('Accuracy: %.2f' % (accuracy*100))

print('F1 Score: %.2f' % (f1_score*100))

print('Precision: %.2f' % (precision*100))

print('Recall: %.2f' % (recall*100))

