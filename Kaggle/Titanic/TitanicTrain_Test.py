import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv', usecols=[1,2,3,4,5,6,7])
df1 = pd.read_csv('test.csv', usecols = [0,1,2,3,4,5,6])

#Clean Training Data
x = df['Age'][df['Name'].str.contains(r'Mr\b')].fillna(value = 0).tolist()
mr = np.mean(x)

x = df['Age'][df['Name'].str.contains(r'Mrs\b')].fillna(value = 0).tolist()
mrs = np.mean(x)

x = df['Age'][df['Name'].str.contains(r'Master\b')].fillna(value = 0).tolist()
master = np.mean(x)

x = df['Age'][df['Name'].str.contains(r'Miss\b')].fillna(value = 0).tolist()
miss = np.mean(x)

x = df['Age'][df['Name'].str.contains(r'Dr\b')].fillna(value = 0).tolist()
dr = np.mean(x)

df.loc[(df['Age'].isnull()) & (df['Name'].str.contains(r'Mr\b')), 'Age'] =  df['Age'][(df['Age'].isnull()) & (df['Name'].str.contains(r'Mr\b'))].fillna(value = mr)
df.loc[(df['Age'].isnull()) & (df['Name'].str.contains(r'Mrs\b')), 'Age'] = df['Age'][(df['Age'].isnull()) & (df['Name'].str.contains(r'Mrs\b'))].fillna(value = mrs)
df.loc[(df['Age'].isnull()) & (df['Name'].str.contains(r'Master\b')),'Age'] = df['Age'][(df['Age'].isnull()) & (df['Name'].str.contains(r'Master\b'))].fillna(value = master)
df.loc[(df['Age'].isnull()) & (df['Name'].str.contains(r'Miss\b')), 'Age'] = df['Age'][(df['Age'].isnull()) & (df['Name'].str.contains(r'Miss\b'))].fillna(value = miss) 
df.loc[(df['Age'].isnull()) & (df['Name'].str.contains(r'Dr\b')),'Age'] = df['Age'][(df['Age'].isnull()) & (df['Name'].str.contains(r'Dr\b'))].fillna(value = dr) 

df['Sex'] = df['Sex'].astype('category')
df['S_code'] = df['Sex'].cat.codes

df.drop(['Name','Sex'], axis = 1, inplace =True)

#Clean Test Data
x = df1['Age'][df1['Name'].str.contains(r'Mr\b')].fillna(value = 0).tolist()
mr = np.mean(x)

x = df1['Age'][df1['Name'].str.contains(r'Mrs\b')].fillna(value = 0).tolist()
mrs = np.mean(x)

x = df1['Age'][df1['Name'].str.contains(r'Master\b')].fillna(value = 0).tolist()
master = np.mean(x)

x = df1['Age'][df1['Name'].str.contains(r'Miss\b')].fillna(value = 0).tolist()
miss = np.mean(x)

x = df1['Age'][df1['Name'].str.contains(r'Dr\b')].fillna(value = 0).tolist()
dr = np.mean(x)

x = df1['Age'][df1['Name'].str.contains(r'Ms\b')].fillna(value = 0).tolist()
ms = np.mean(x)

df1.loc[(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Mr\b')), 'Age'] =  df1['Age'][(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Mr\b'))].fillna(value = mr)
df1.loc[(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Mrs\b')), 'Age'] = df1['Age'][(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Mrs\b'))].fillna(value = mrs)
df1.loc[(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Master\b')),'Age'] = df1['Age'][(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Master\b'))].fillna(value = master)
df1.loc[(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Miss\b')), 'Age'] = df1['Age'][(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Miss\b'))].fillna(value = miss) 
df1.loc[(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Dr\b')),'Age'] = df1['Age'][(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Dr\b'))].fillna(value = dr)
df1.loc[(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Ms\b')),'Age'] = df1['Age'][(df1['Age'].isnull()) & (df1['Name'].str.contains(r'Ms\b'))].fillna(value = ms)

df1['Sex'] = df1['Sex'].astype('category')
df1['S_code'] = df1['Sex'].cat.codes
df1.drop(['Name', 'Sex'], axis = 1, inplace =True)
p_id = df1.values[:,0]
df1.drop(['PassengerId'], axis = 1, inplace = True)


"""
Train Data
   Survived  Pclass   Age  SibSp  Parch  S_code
0         0       3  22.0      1      0       1
1         1       1  38.0      1      0       0
2         1       3  26.0      0      0       0
3         1       1  35.0      1      0       0
4         0       3  35.0      0      0       1

Test Data
   Pclass   Age  SibSp  Parch  S_code
0       3  34.5      0      0       1
1       3  47.0      1      0       0
2       2  62.0      0      0       1
3       3  27.0      0      0       1
4       3  22.0      1      1       0
"""


X = df.values[:,1:6]
y = df.values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

clf = RandomForestClassifier()

tModel = clf.fit(X_train, y_train)
y_pred = tModel.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
"""
#Split Data
X_train = df.values[:,1:6]
y_train = df.values[:, 0]

X_test = df1.values[:, :5]

clf = RandomForestClassifier()
tModel = clf.fit(X_train, y_train)
y_pred = tModel.predict(X_test)

print(y_pred)

df2 = pd.DataFrame(data = np.c_[p_id, y_pred], columns = ['PassengerId','Survived']);

print(df2.head())
df2 = df2.astype('int32')
df2.to_csv('result.csv', sep=',', index=False)
"""