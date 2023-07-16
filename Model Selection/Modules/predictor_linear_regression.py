import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# read data
path = 'C:/Users/Dorrin/Simulation_Project'
df = pd.read_csv(path+'/'+'data.csv')
df = pd.DataFrame(df)
df.head(100)
data = df.values

### functions
def age(born):
    if born=='#NULL!':
        return 55
    splt = (str.split(born,'-'))
    born = int(splt[2])

    return 2023 - (1900 + born)
def salary(string):
    num =  string.replace('$','').replace(' ','').replace(',','') # remove $ symbol
    return int(num)

# calculate the age according to bdate
df['Age'] = df['bdate'].apply(age)
# salary (str to int)
df['salary'] = df['salary'].apply(salary)

print(df.head(5))


### arange data (train/test - input/output)
x = df.iloc[:, [1, 3, 4, 6, 7]].values
y = df.iloc[:, [5]].values
#print(x.shape)
#print(y.shape)

### split data to train-test
Xtrain, Xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=0)

### linear regression
reg = LinearRegression().fit(Xtrain, ytrain)
prd_train = reg.predict(Xtrain)
prd_test = reg.predict(Xtest)

rmse_train = sqrt(mean_squared_error(ytrain, prd_train)) #RMSE
mae_train = mean_absolute_error(ytrain, prd_train) #MAE
rmse_test = sqrt(mean_squared_error(ytest, prd_test)) #RMSE
mae_test = mean_absolute_error(ytest, prd_test) #MAE

print('RMSE train = ',rmse_train)
print('RMSE test = ',rmse_test)
print('MAE train = ',mae_train)
print('MAE test = ',mae_test)
