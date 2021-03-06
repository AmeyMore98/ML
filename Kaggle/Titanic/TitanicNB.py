import pandas as pd
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper
from sklearn import feature_selection
import seaborn as sns
import os

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df = pd.read_csv('titanic_train.csv')

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

df.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'

encodable_columns = ['Sex', 'Embarked', 'Pclass']
feature_defs = [(col_name, preprocessing.LabelEncoder()) for col_name in encodable_columns]
mapper = DataFrameMapper(feature_defs)
mapper.fit(df)
df[encodable_columns] = mapper.transform(df)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
sns.FacetGrid(df, row="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

def convert_family_size(size):
	if size == 1:
		return 'Single'
	elif size <=3:
		return 'Small'
	elif size <= 6:
		return 'Medium'
	else:
		return 'Large'

df['Family_Size'] = df['FamilySize'].map(convert_family_size)
sns.factorplot(x="Family_Size", hue="Survived", data=df, kind="count", size=6)

df1 = df.drop(['PassengerId', 'Name', 'Cabin','Ticket','Survived', 'FamilySize'], axis=1)

features = ['Sex', 'Embarked', 'Pclass', 'Family_Size']
df2 = pd.get_dummies(df1, columns=features)
y = df['Survived']

best_k = feature_selection.SelectKBest(feature_selection.chi2, k=10)
best_k.fit(df2, y)
print(best_k.get_support())
print(best_k.scores_)
new_feature = df2.columns[best_k.get_support()]
X = df2[new_feature]
print(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

gnb = GaussianNB()

model = gnb.fit(X_train, y_train)

y_pred = model.predict(y_test.reshape(-1,1))

print("Accuracy:", accuracy_score(y_test, y_pred) * 100)