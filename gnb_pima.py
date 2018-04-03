import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score

def pxy(x, mean, var):
	p = np.exp((-(x-mean)**2) / (2*(var))) / np.sqrt(2*np.pi*(var))
	return p

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df = pd.read_csv('pima_diabetes.csv', header = None, names = ['Times_Pregnant', 'Glucose_conct', 'Blood_pressure', 'S_K_T', 'serum_insulin', 'BMI', 'Diabetes_pedigree', 'Age', 'Class'])
#print(df.head())

train = df.iloc[:536, :]
test = df.iloc[536:, :]

X_test = test.values[:, :8]
y_test = test.values[:, 8]

X_mean = train.groupby('Class').mean().values.tolist()
X_var = train.groupby('Class').var().values.tolist()

#print(X_mean,'\n', X_var)

Pos_p = train['Class'][train['Class'] == 1].count() / train['Class'].count()
Neg_p = train['Class'][train['Class'] == 0].count() / train['Class'].count()

#print(Pos_p,'\n', Neg_p)
pred = []
for example in X_test:
	N_prob = Neg_p
	P_prob = Pos_p
	for i in range(len(example)):
		N_prob *= pxy(example[i], X_mean[0][i], X_var[0][i])
		P_prob *= pxy(example[i], X_mean[1][i], X_var[1][i])
	pred.append(1 if P_prob > N_prob else 0)

print("ACC: ", accuracy_score(y_test, pred) * 100)



