import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor   
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
import pickle

df = pd.read_csv("ipl.csv")
df.head()

print("Shape:",df.shape)
print("Dtype:",df.dtypes)
print(df.columns)

df.isna().sum()

df.duplicated().sum()

print("Before removing first 6 overs:",df.shape)
df_without_six = df[df['overs'] >= 6.0]
print("After removing first 6 overs:",df_without_six.shape)

Y_without_six = df_without_six['total']
print(Y_without_six)

X_without_six = df_without_six.drop(columns=['total','mid','venue','batsman','bowler','striker','non-striker','date'])
print(X_without_six.columns)
print(X_without_six.shape)

encoded_X_without_six =pd.get_dummies(data=X_without_six)
encoded_X_without_six.head(2)

encoded_X_without_six.shape

train_x,test_x,train_y,test_y=train_test_split(encoded_X_without_six,Y_without_six,test_size=0.2)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

model_dt =DecisionTreeRegressor()
model_dt.fit(train_x,train_y)

test_x_pred = model_dt.predict(test_x)

print("Mean Absolute Error (MAE):",mae(test_y,test_x_pred))
print("Mean Squared Error (MSE):",mse(test_y,test_x_pred))

print("Root Mean Squared Error (RMSE):",np.sqrt(mse(test_y,test_x_pred)))

pickle.dump( model_dt , open('model.pkl', 'wb'))