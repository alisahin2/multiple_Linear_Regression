# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 23:49:32 2020

@author: Alisahin2
"""

# salary = dependent variable
# experience , age = independent variable

# b0 b1 b2 ?
# find the min(MSE)
#MSE = mean square error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv", sep = ";")


x = df.iloc[:, [0,2]].values #this row describe: select the all rows and select the 0. and 2. columns
y = df.salary.values.reshape(-1,1)

#%%
multiple_lr = LinearRegression()
multiple_lr.fit(x,y)

print("b0: " , multiple_lr.intercept_) # bias
print("b1,b2: " , multiple_lr.coef_) # b1 and b2

#predict
multiple_lr.predict(np.array([[10,35],[5,35]]))

# if experience = 10 and age = 35 => salary = 11046.35815877
# if experience = 5 and age = 35 => salary = 3418.85455609

















