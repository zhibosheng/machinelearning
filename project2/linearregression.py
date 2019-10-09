import numpy as np 
import pandas as pd 
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import time
#https://www.geeksforgeeks.org/linear-regression-python-implementation/


def linearRegression(feature_vector,label):
	length = len(feature_vector)
	feature_vector_m = np.mean(feature_vector)
	label_m = np.mean(label)
	SS_xy = np.sum([label[i]*feature_vector[i] for i in range(len(label))]) - length*feature_vector_m*label_m
	SS_xx = np.sum([feature_vector[i][0]**2 for i in range(len(feature_vector))]) - length*feature_vector_m*feature_vector_m
	w = SS_xy/SS_xx
	b = label_m - w*feature_vector_m
	return w,b

def predict(X_test,w,b):
	result = []
	for ele in X_test:
		result.append(w*ele+b)
	return result

def getError(Y_test,pred):
	return sum([abs(Y_test[i] - pred[i]) for i in range(len(Y_test))])/len(pred)
dataset = pd.read_csv("./house/kc_house_data.csv")
space = dataset['sqft_living']
price = dataset['price']
x = np.array(space).reshape(-1, 1)
y = np.array(price)
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=1/3, random_state=0)
t0 = time.time()
w,b = linearRegression(X_train,Y_train)
t1 = time.time()
print(t1-t0)
print(w)
print(b)
pred = predict(X_test,w,b)
error = getError(Y_test,pred)
print(error)
plt.scatter(X_test, Y_test, color= 'red')
plt.plot(X_test, pred, color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()