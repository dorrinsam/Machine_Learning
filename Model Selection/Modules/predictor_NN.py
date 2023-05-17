import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.neural_network import MLPRegressor

# read data
path = 'C:/Users/OMID/OneDrive/Freelance/Project 4'
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

### Preprocessing >>> to increase the performance of the predictors
norm_input = preprocessing.MinMaxScaler()
norm_output = preprocessing.MinMaxScaler()
norm_input.fit(Xtrain)
norm_output.fit(ytrain)

# rescale data
Xtrain_scaled = norm_input.transform(Xtrain)
ytrain_scaled = norm_output.transform(ytrain.reshape(-1, 1))
Xtest_scaled = norm_input.transform(Xtest)
ytest_scaled = norm_output.transform(ytest.reshape(-1, 1))

### neural network
epochs = 50 # iterations
af = "relu" # activation function
s = 32 # neurons
reg = MLPRegressor(random_state=4, max_iter=epochs, hidden_layer_sizes=s,activation=af).fit(Xtrain_scaled,ytrain_scaled)
print('Train prediction Accuracy=', reg.score(Xtrain_scaled, ytrain_scaled))
print('Test prediction Accuracy=',reg.score(Xtest_scaled, ytest_scaled))
#print(regr.score(Xtest, ytest))

### Predict
prd_train = reg.predict(Xtrain_scaled) # use the scaled data to predict using NN
prd_test = reg.predict(Xtest_scaled)

### rescale data to attain the original data
prd_train = norm_output.inverse_transform(prd_train.reshape(-1, 1)) # actual salary predicted in dollar
prd_test = norm_output.inverse_transform(prd_test.reshape(-1, 1))
###

rmse_train = sqrt(mean_squared_error(ytrain, prd_train)) #RMSE
mae_train = mean_absolute_error(ytrain, prd_train) #MAE
rmse_test = sqrt(mean_squared_error(ytest, prd_test)) #RMSE
mae_test = mean_absolute_error(ytest, prd_test) #MAE

### 
print('RMSE train = ',rmse_train)
print('RMSE test = ',rmse_test)
print('MAE train = ',mae_train)
print('MAE test = ',mae_test)