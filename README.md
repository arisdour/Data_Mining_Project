# Data_Mining_Project

The main goal of the first problem is to Classify Wines Using the Red Wine Quality Dataset and Support Vectors Machines. 
In Version 0 the dataset is fitted in a SVM. 
In version 1 the dataset is slightly modified in order to examine different approaches for dealing with  missing data (Clustering , Regression Algorithms).

########################################################

PROBLEM 1 : Red Wine Classification Using Machine Learning

Classify a red wine, using 12 different Features, in 6 different Classes (3-8) . The algorythm used for classification is An SVM (Support Vector Machine).

- *** Version 0 ****

Split Dataset to Training Set and Test Set (75% -25%) and then calculate F1 Score , Precission , Recall.
Calculate the Hyperparameters for the SVM

Then Keep the Best results

- *** Version 1 *****

Remove 33% of pH Data from the original Dataset

Try to resolve the missing pH data problem by:
  Removing that feature completely
  Filling the missing Data With the mean value of pH
  Filling the missing data by using Logistic Regression
  Clustering using K-Means algorithm and then replacing the missing data with the average of the cluster that the wine belongs to
Calculate F1 Score , Precission , Recall.


########################################################

PROBLEM 2: Onion or Not 

Process the onion or not dataset in order to classify news Titles, using a Neural Network as: Real or Fake News 

Use the onion or not dataset and :
  Split the titles into Word Vectors
  Stem the words 
  Remove stop words 
  Calculate tf-idf

Feed the final array into a Neural Network 
Calculate F1, Score Precision and Recall
