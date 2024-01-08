# Sai S. Bushigampala
# Machine Learning
# CSCI 4371
# ML Project - Final code version

# Most of this is amended from the previous first experiment I did.

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
  
# fetch dataset 
national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset = fetch_ucirepo(id=887) 
  
# data (as pandas dataframes) 
X = national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.data.features 
y = national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.data.targets 
  
# metadata 
#print(national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.metadata) 
print("\n")
 
# variable information 
#print(national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.variables) 
print("\n")

# Comparison between Logistic Regression Model and KNN model where k=5
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

# Logistic Regression

logReg = LogisticRegression(random_state=0, max_iter=200).fit(X_train, Y_train.values.ravel())
logReg.predict(X_train)
score_CV_LR = cross_val_score(logReg, X_train, Y_train.values.ravel(), cv=5)
print("Score: Logistic Regression Cross Validation: ", score_CV_LR)
print("\n")

# KNN w/ k=5
neigh2 = KNeighborsClassifier(n_neighbors=5)
neigh2.fit(X_train, Y_train.values.ravel())
neigh2.predict(X_train)
score_CV_k5 = cross_val_score(neigh2, X_train, Y_train.values.ravel(), cv=5)
print("Score: K Nearest Neighbors with k=5 Cross Validation:", score_CV_k5)
print("\n")

lgrgAvg = sum(score_CV_LR) / 5
knnAvg = sum(score_CV_k5) / 5

print("Average Score of Log Reg:", lgrgAvg)  
print("Average Score of KNN with k=5:", knnAvg) 
print("\n")

# Logisitic Regression gives us a better average score, so I will use the Logisitic Regression Function from now on.
logReg = LogisticRegression(random_state=0, max_iter=200).fit(X_train, Y_train.values.ravel())
logReg.predict(X_train)
score_CV_LR = cross_val_score(logReg, X_train, Y_train.values.ravel(), cv=5)

# Evaluate with Test Set
logReg2 = LogisticRegression(random_state=0, max_iter=200).fit(X_test, Y_test.values.ravel())
y_test_pred = logReg2.predict(X_test)
score_LR2 = logReg2.score(X_test, Y_test.values.ravel())
print("Score: Logistic Regression  (Test Set): ", score_LR2)
print("\n")


print ("\n Testing Predictions", y_test_pred)



# Predictions that are wrong, Indexes/rows 2, 6, 8 - only examples I could find.


wrong = []
right = []

# Limited to 684 instead of full subset because as soon as i > 684, an out-of-bounds error persists. 
for i in range(0, 683):
    if(y_test_pred[i] == (Y_test.iloc[i].iloc[0])):
      right.append(i)
    else:
      wrong.append(i)

print("\n Wrong Class Indices: ", wrong)
print("\n Right Class Indices: ", right)

print("\n")
for i in range(0, 5):
    print("Prediction Class:", y_test_pred[wrong[i]])
    print("Actual Class: ", Y_test.iloc[wrong[i]].iloc[0])


# input attributes are the variables
print("\n")
print(national_health_and_nutrition_health_survey_2013_2014_nhanes_age_prediction_subset.variables) 
print("\n")
