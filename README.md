# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries which are used for the program.

2.Load the dataset.

3.Check for null data values and duplicate data values in the dataframe.

4.Apply logistic regression and predict the y output.

5.Calculate the confusion,accuracy and classification of the dataset.
   

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: TARANIKKA A
RegisterNumber:212223220115
/*

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
PLACEMENT DATA

![Screenshot 2023-08-31 104926](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/9eac3ff4-f343-4e73-a759-e4b6b081b884)

SALARY DATA

![Screenshot 2023-08-31 111210](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/df1d3c88-182e-4e28-9793-8874b4dcda51)

NULL FUNCTION

![Screenshot 2023-08-31 111320](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/af62f6e7-d646-43dd-94f1-a910be8ca881)

DUPLICATE()

![Screenshot 2023-08-31 111433](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/d86080d3-4c91-471f-a53c-86bbcbfff59d)

LABEL ENCODING

![Screenshot 2023-08-31 111736](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/09361796-0c3a-4a13-b39f-2b6bd7eb0c07)

X VALUE

![Screenshot 2023-08-31 111850](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/cd370f07-5dae-461f-b2af-01326bba611e)

Y VALUE

![Screenshot 2023-08-31 111910](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/e5ae64a0-9b65-4bf8-84bf-1e36d8447976)

ARRAY

![Screenshot 2023-08-31 110742](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/2dbf79b5-89ac-4f9f-8345-a421a096aa2e)

ACCURACY

![Screenshot 2023-08-31 112206](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/e8759ace-b23e-4548-b6ff-1e0259bc7988)

CONFUSION MATRIX

![Screenshot 2023-08-31 110750](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/82bd6e86-990f-47d1-a151-9c3a530de274)

CLASSIFICATION REPORT

![Screenshot 2023-08-31 110858](https://github.com/Preetha-Senthamilan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119390282/938d0f55-aaa2-4c17-b3f7-5f4975307713)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
