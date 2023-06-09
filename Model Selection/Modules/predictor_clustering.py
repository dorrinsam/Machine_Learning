import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

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

def get_scores(y_true, y_prd):

    rmse = sqrt(mean_squared_error(y_true, y_prd)) #RMSE
    mae = mean_absolute_error(y_true, y_prd) #MAE

    print('RMSE = ',rmse)
    print('MAE = ',mae)

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

### clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(Xtrain)
labels = kmeans.labels_
X0train = Xtrain[labels==0, :]
X1train = Xtrain[labels==1, :]
X2train = Xtrain[labels==2, :]
X3train = Xtrain[labels==3, :]

y0train = ytrain[labels==0, :]
y1train = ytrain[labels==1, :]
y2train = ytrain[labels==2, :]
y3train = ytrain[labels==3, :]

### predict clusters for TEST data
labels = kmeans.predict(Xtest)
X0test = Xtest[labels==0, :]
X1test = Xtest[labels==1, :]
X2test = Xtest[labels==2, :]
X3test = Xtest[labels==3, :]

y0test = ytest[labels==0, :]
y1test = ytest[labels==1, :]
y2test = ytest[labels==2, :]
y3test = ytest[labels==3, :]

### Train on each cluster
reg0 = LinearRegression().fit(X0train, y0train)
reg1 = LinearRegression().fit(X1train, y1train)
reg2 = LinearRegression().fit(X2train, y2train)
reg3 = LinearRegression().fit(X3train, y3train)

### train data scores
prd0train = reg0.predict(X0train)
prd1train = reg1.predict(X1train)
prd2train = reg2.predict(X2train)
prd3train = reg3.predict(X3train)
print('Train cluster 0:')
get_scores(y0train, prd0train)
print('Train cluster 1:')
get_scores(y1train, prd1train)
print('Train cluster 2:')
get_scores(y2train, prd2train)
print('Train cluster 3:')
get_scores(y3train, prd3train)

### test data scores
prd0test = reg0.predict(X0test)
prd1test = reg1.predict(X1test)
prd2test = reg2.predict(X2test)
prd3test = reg3.predict(X3test)
print('Test cluster 0:')
get_scores(y0test, prd0test)
print('Test cluster 1:')
get_scores(y1test, prd1test)
print('Test cluster 2:')
get_scores(y2test, prd2test)
print('Test cluster 3:')
get_scores(y3test, prd3test)