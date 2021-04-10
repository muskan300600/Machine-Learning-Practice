import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error , mean_absolute_error
#sklearn provides some built in datasets so we will be importing those for practice
#linear_model will be our first machine learning model

diabetes = datasets.load_diabetes()
#loading the dataset

#print(diabetes.keys())
'''loads all the features present in the dataset'''
#['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']

#print(diabetes.data)
'''loads all the data available in the dataset'''

#print(diabetes.DESCR)
'''describes the dataset'''

diabetes_X = diabetes.data
'''loads the entire dataset'''

#diabetes_X = diabetes.data[:,np.newaxis,2]
'''loads the second column in the dataset only. So if you wish to display only a single feature use this
Slicing features from dataset
'''

diabetes_X_train = diabetes_X[:-30]
#Slicing of the features training set
diabetes_X_test = diabetes_X[-20:]
#Slicing of the features test set

diabetes_Y_train = diabetes.target[:-30]
#Slicing of the labels training set
diabetes_Y_test = diabetes.target[-20:]
#Slicing of the labels test set

#Creating the model
model = linear_model.LinearRegression()
#Fitting the model
model.fit(diabetes_X_train,diabetes_Y_train)
#Predicting
diabetes_Y_predict = model.predict(diabetes_X_test)

#checking for accuracy by calculating mean squared error
print("The Mean Squared error is: ", mean_squared_error(diabetes_Y_test,diabetes_Y_predict))
print("The Mean Absolute error is: ", mean_absolute_error(diabetes_Y_test,diabetes_Y_predict))

#Calculating weights and intercept
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

#Plotting incase of a single feature
#plt.scatter(diabetes_X_test,diabetes_Y_test) #Scattered plot
#plt.plot(diabetes_X_test,diabetes_Y_predict) #Line plot
#plt.show()











