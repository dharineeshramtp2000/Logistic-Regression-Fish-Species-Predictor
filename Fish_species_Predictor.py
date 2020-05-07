#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset. Here we use the dataset from Kaggle.
Dataset = pd.read_csv("Fish.csv")
X = Dataset.iloc[ : ,[1,2,3,5,6]].values
y = Dataset.iloc[ : , 0].values

#Now lets feature scale our independent variables from our famous scikit library
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the dataset into training and testing and performing a good shuffle of data
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X , y , test_size = 0.25, random_state= 42,shuffle= True)

#Now lets define our Logistic Regression model. 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000, random_state = 0, multi_class='ovr', solver='lbfgs')
classifier.fit(X_train, y_train)

#Now lets predict with our test set
y_pred = classifier.predict(X_test)

#Its time for evaluating our model.

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
acc_train = accuracy_score(y_train, classifier.predict(X_train))
f1_train = f1_score(y_train, classifier.predict(X_train), average= 'weighted')

print("Traing set results")
print("ACCURACY ---------------------->",acc_train)
print("F1 SCORE ---------------------->",f1_train)

#Now lets see how well is our model. So now lets evaluate with our test set
acc_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred, average= 'weighted')

print("Test set results")
print("ACCURACY ---------------------->",acc_test)
print("F1 SCORE ---------------------->",f1_test)

#Now lets have our famous Confusion Matrix to visually understand.
cm = confusion_matrix(y_test,y_pred)
print(cm)