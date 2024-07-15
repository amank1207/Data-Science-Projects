# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:20:22 2020

@author: amank
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os

os.chdir(r"C:\Users\amank\Downloads")

# setting dimensions for plot
sns.set(rc={'figure.figsize':(11.7,8.27)})

cars_data=pd.read_csv('cars_sampled.csv')

cars=cars_data.copy()
cars.info()
cars.describe()

pd.set_option('display.float_format',lambda x: '%3f' %x)
cars.describe()

pd.set_option('display.max_columns',500)
cars.describe()

#Dropping unwanted colums
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)

cars.drop_duplicates(keep='first',inplace=True)

cars.isnull().sum()

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)

price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)

power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#working range of data
cars=cars[(cars.yearOfRegistration<=2018)
         &(cars.yearOfRegistration>=1950)
         &(cars.price<=150000)
         &(cars.price>=100)
         &(cars.powerPS<=500)
         &(cars.powerPS>=10)]


# combining yearOfReg & yearOfMonth same and adding in new column for age of car
cars['monthOfRegistration']/=12

cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

#Dropping yearOfRegistration & monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)


sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#Age Vs Price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)

#powerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)


cars['seller'].value_counts() #can be dropped since, it is less significant as almost all are of same type
cars['offerType'].value_counts() #can be dropped since, it is less significant as all are of same type

cars['abtest'].value_counts()
sns.boxplot(x='abtest', y='price',data=cars)
#can be dropped since, price for both types are almost same

cars['vehicleType'].value_counts()
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType', y='price',data=cars) #CANNOT be dropped

cars['gearbox'].value_counts()
sns.boxplot(x='gearbox', y='price',data=cars) #CANNOT be dropped

cars['model'].value_counts()
sns.countplot(x='model',data=cars)
sns.boxplot(x='model', y='price',data=cars) #CANNOT be dropped

cars['kilometer'].value_counts()
sns.boxplot(x='kilometer', y='price',data=cars) #CANNOT be dropped

cars['fuelType'].value_counts()
sns.boxplot(x='fuelType', y='price',data=cars) #CANNOT be dropped

cars['brand'].value_counts()
sns.boxplot(x='brand', y='price',data=cars) #CANNOT be dropped

cars['notRepairedDamage'].value_counts()
sns.boxplot(x='notRepairedDamage', y='price',data=cars) #CANNOT be dropped

#Removing insignificant variables
col=['seller','abtest','offerType']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#Correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)

#descending order correlation against price
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


'''
Building Linear Regression Model & Random Forest model
on 2 different data sets -
1. data with deleting missing values
2. data with imputing missing values
'''

cars_omit=cars.dropna(axis=0)

#converting categorical variables into dummy variables(divides each value in separate row with data 1 or 0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#separating input & output features
x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

#plotting variable price
prices=pd.DataFrame({"1.Before":y1,"2.After":np.log(y1)})
prices.hist()
#so, it is better to use log value for prices as it gives better distribution view

y1=np.log(y1)

#splitting data into test & train
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)



# BASE MODEL FOR OMMITED DATA
base_pred=np.mean(y_test)
base_pred=np.repeat(base_pred,len(y_test))

#finding RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)
# the model we build, should have RMSE lower than this base model





'''
Linear Regression with Omitted Data
'''
lgr=LinearRegression(fit_intercept=True) 

#Model
model_lin1=lgr.fit(x_train,y_train)

#predicting model on test set
cars_predictions_lin1=lgr.predict(x_test)

#computing MSE & RMSE
lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)            #much lower than the base model

#R-squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)         # almost same variability is aptured, so model is good

#Regression diagnostics - Residual(actual minus predicted) plot analysis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True,fit_reg=False)
residuals1.describe()          # we can see there is not much of difference since mean is small




'''
RANDOM FOREST WITH OMITTED DATA
'''
rf=RandomForestRegressor(n_estimators=100,max_features='auto',
                         max_depth=100,min_samples_split=10,
                         min_samples_leaf=4,random_state=4)


#Model
model_rf1=rf.fit(x_train,y_train)

#predicting model on test set
cars_predictions_rf1=rf.predict(x_test)

#computing MSE & RMSE
lin_mse1=mean_squared_error(y_test,cars_predictions_rf1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)            #much lower than the base model

#R-squared value
r2_rf_test1=model_rf1.score(x_test,y_test)
r2_rf_train1=model_rf1.score(x_train,y_train)
print(r2_rf_test1,r2_rf_train1)


########################################################
############################################################
###############################################################
############################################################
########################################################


'''
MODEL BUILDING WITH IMPUTED DATA
'''

cars_imputed=cars.apply(lambda x:x.fillna(x.median())\
                        if x.dtype=='float' else\
                        x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()


#separating input & output features
x2=cars_imputed.drop(['price'],axis='columns',inplace=False)
y2=cars_imputed['price']


y2=np.log(y2)

#splitting data into test & train
x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(x_train2.shape,x_test2.shape,y_train2.shape,y_test2.shape)



# BASE MODEL FOR OMMITED DATA
base_pred2=np.mean(y_test2)
base_pred2=np.repeat(base_pred2,len(y_test2))

#finding RMSE
base_root_mean_square_error2=np.sqrt(mean_squared_error(y_test2,base_pred2))
print(base_root_mean_square_error2)
# the model we build, should have RMSE lower than this base model





'''
Linear Regression with Omitted Data
'''
lgr2=LinearRegression(fit_intercept=True) 

#Model
model2_lin1=lgr2.fit(x_train2,y_train2)

#predicting model on test set
cars_predictions_lin1=lgr.predict(x_test)

#computing MSE & RMSE
lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)            #much lower than the base model

#R-squared value
r2_lin_test1=model_lin1.score(x_test,y_test)
r2_lin_train1=model_lin1.score(x_train,y_train)
print(r2_lin_test1,r2_lin_train1)         # almost same variability is aptured, so model is good

#Regression diagnostics - Residual(actual minus predicted) plot analysis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True,fit_reg=False)
residuals1.describe()          # we can see there is not much of difference since mean is small




