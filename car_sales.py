# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:23:25 2020

@author: MANOJ KUMAR SHRIVAST
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

car_df=pd.read_csv('Car_Purchasing_Data.csv',encoding='ISO-8859-1')

car_df.head()

car_df.tail()

sns.pairplot(car_df)

X=car_df.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)

y=car_df['Car Purchase Amount']

X.shape

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)

y=y.values.reshape(-1,1)
y_scaled=scaler.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled)

import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(25,input_dim=5,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(1,activation='linear'))

model.summary()

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
epochs_hist=model.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)

epochs_hist.history.keys()

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation loss')
plt.xlabel('Epochs')

#Gender ,Age,Annual Salary,Credit Card Debt,Net Worth

y_predict=model.predict(X_test)

from keras.metrics import accuracy




