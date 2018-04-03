import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

def predict(X, theta):
	h_theta = np.matmul(X, theta)
	return h_theta

def compute_cost(X, Y, h_theta):
	m = len(Y)
	return (np.matmul(X.transpose(),((h_theta - Y)**2)) / m)

def gradient_descent(X, Y, alpha, iterations, theta):
	J_theta_hist = []
	n = np.size(X,1)
	cost = np.ones(shape=(n,1))
	#print('cost:',cost)
	for i in range(iterations):
		h_theta = predict(X, theta)
		#print('h_theta for:', h_theta)
		cost = compute_cost(X, Y, h_theta)
		#print('Cost:', cost)
		theta = theta - (alpha * cost.astype(float))
		J_theta_hist.append(theta)

	#print('J_theta_hist:', J_theta_hist)
	return J_theta_hist[-1]

os.chdir(r'C:\Users\Amay\Desktop\Sem6\ML\Datasets')

df = pd.read_csv('pima_diabetes.csv')
X = df.values[:, :8]
y = df.values[:, 8]

"""X = np.array([[1],[2],[3],[4]])
Y = np.array([[2],[3],[4],[5]]).astype(float)
y_test = np.array([[1,6],[1,7],[1,8],[1,9]]).astype(float)"""

#adding X0
X = np.insert(X, 0, 1, axis = 1).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
#print('X:',X)
#print('Y:',Y)

theta = np.random.randn(len(X[0]),1)
theta = theta.astype(float)
#print('Theta:',theta)

iterations = 10
alpha = 0.0001

theta = gradient_descent(X_train, y_train, alpha, iterations, theta)

pred = predict(X_test, theta)
print("ACC:",accuracy_score(y_test, pred))

# lr = LinearRegression()
# model = lr.fit(X, Y)
# pred = model.predict(y_test)
# print('Sklearn:',pred)