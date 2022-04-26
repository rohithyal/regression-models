from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#linear regression necessities
from sklearn import linear_model 

#polynomial regression necessities
from sklearn.preprocessing import PolynomialFeatures



df = pd.read_csv('day.csv')
#Feature Selection
df = df.drop(columns=['dteday','instant'])

# Normalizing the data
from sklearn import preprocessing
x=df.drop(['cnt'],axis=1)
y=df['cnt']
x = preprocessing.normalize(x)

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
linearRegressor = LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predicted = linearRegressor.predict(x_test)
mse = mean_squared_error(y_test, y_predicted)
r = r2_score(y_test, y_predicted)
mae = mean_absolute_error(y_test,y_predicted)
print()
print("Mean Squared Error:",mse)
print("R score:",r)  #value to be compared 
print("Mean Absolute Error:",mae)


#Polynomial Regression
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x_train)
x_poly_test = polynomial_features.fit_transform(x_test)
model = LinearRegression()
model.fit(x_poly, y_train)
y_predicted_p = model.predict(x_poly_test)
mse = mean_squared_error(y_test, y_predicted_p)
r = r2_score(y_test, y_predicted_p)
mae = mean_absolute_error(y_test,y_predicted_p)
print()
print("Mean Squared Error:",mse)
print("R score:",r)
print("Mean Absolute Error:",mae)


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)
y_predicted_d = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_predicted_d)
r = r2_score(y_test, y_predicted_d)
mae = mean_absolute_error(y_test,y_predicted_d)
print()
print("Mean Squared Error:",mse)
print("R score:",r)
print("Mean Absolute Error:",mae)
