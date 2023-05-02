# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:53:35 2023

@author: casper
"""

# Reading data and choosing dependent and independent variable from csv
import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")    

x = data.iloc[:,3:-1]
y = data.iloc[:,-1]

# Encoding the data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x.iloc[:,1] = le.fit_transform(x.iloc[:,1])

le2 = LabelEncoder()
x.iloc[:,2] = le2.fit_transform(x.iloc[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
    
    )
x = ohe.fit_transform(x)
x = x[:,1:]

# Spliting the data to train and split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

# Scaling the independent
sc = StandardScaler()
x_train = sc.fit_transform(x_train) 
x_test = sc.fit_transform(x_test)

# Yapay Sinir Ağı
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu", input_dim=11))
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])

classifier.fit(x_train,y_train, epochs=500)
y_pred = classifier.predict(x_test)







