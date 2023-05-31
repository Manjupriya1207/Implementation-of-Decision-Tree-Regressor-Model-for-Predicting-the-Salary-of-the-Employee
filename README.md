# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import dataset and get data info
2. check for null values
3. Map values for position column
4. Split the dataset into train and test set
5. Import decision tree regressor and fit it for data
6. Calculate MSE,R2 and y predict. 
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Manjupriya P
RegisterNumber: 212220220024
*/
``
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])`

## Output:
1.data.head()

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/e1b5a3b2-38ab-4c0d-a330-92d8ccc954e0)

2.data.info()

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/44d83a51-0c03-4500-a1aa-1a0bf07784b5)

3.isnull() and sum()

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/0e8e113c-982a-4c03-8022-574e47c76388)

4.data.head() for salary

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/1015f472-caa3-493f-a076-486754605812)

5.MSE Value

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/fc724bb1-b4a1-4bb8-9bc2-71abcd3a174b)

6.r2 value

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/0da82738-3309-4401-8906-b8829787ea1a)

7.data prediction

![image](https://github.com/Manjupriya1207/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113583090/0749ca93-cb21-42db-91d0-35ab796517be)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
