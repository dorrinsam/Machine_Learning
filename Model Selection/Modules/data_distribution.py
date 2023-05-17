import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

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
X = df.iloc[:, [1, 3, 4, 6, 7]].values
y = df.iloc[:, [5]].values

#sns.displot(df["salary"], kind="kde", fill=True)
#sns.displot(df["educ"], kind="kde", fill=True)
#sns.displot(df["jobcat"], fill=True)
#sns.displot(df["prevexp"], kind="kde", fill=True)
sns.displot(df["Age"], kind="kde", fill=True)

plt.show()