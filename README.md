# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and get data info
2. check for null values
3. Map values for position column
4. Split the dataset into train and test set
5. Import decision tree regressor and fit it for data
6. Calculate MSE,R2 and y predict

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vigneshkumar V
RegisterNumber: 212220220054
*/
```
import pandas as pd

data=pd.read_csv('/content/Salary.csv')

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

x.head()

y=data[["Salary"]]

y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=200)

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])
*/

## Output:
1.DATA.HEAD():

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/d900f06c-f87e-4d0c-aa65-35783fa60ece)

2.DATA.INFO()

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/67fb95d4-bc5f-4897-b4a3-1c90914ca855)

3.ISNULL()AND SUM():

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/148c14e5-3776-4f9b-90a7-7c525eb5655b)

4.DATA.HEAD() FOR SALARY:

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/ebec4ee3-e0d8-4355-996e-6fcc35401389)

5.MSE VALUE:

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/009d3039-ba7b-4b7b-9526-6c478c386557)

6.R2 VALUE:

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/74f67fd4-93d2-44ee-80cb-0ccaa3c87917)

7.DATA PREDICTION:

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113573894/a52b1789-ca26-4581-97d8-67a4ed38b190)






## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
