import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from subprocess import check_output
import time


df = pd.read_csv('./cancer/data.csv')

df_std = StandardScaler().fit_transform(df.drop(['id','diagnosis','Unnamed: 32'], axis = 1))
label = np.array([0 for x in range(len(df_std))])

print(len(df_std))
for x in range(len(label)):
	if df["diagnosis"][x]== "M":
		label[x] = 1
	else:
		label[x] = -1

X_train, X_test, Y_train, Y_test = train_test_split(df_std, label,test_size=1/3, random_state=0)
print(len(X_train[0]))
epochs = 1
learning_rate = 0.00001
w =[0 for x in range(30)]
t0 = time.time()
while epochs < 100:
	for i in range(len(X_train)):
		pred = 0
		for j in range(30):
			pred += w[j]*X_train[i][j] 
		
		if pred*Y_train[i] < 1:
			for j in range(30):
				w[j] = w[j] + learning_rate*(Y_train[i]*X_train[i][j]-0.02*w[j])
		else:
			for j in range(30):
				w[j] = w[j] - learning_rate*(0.02*w[j])
	epochs += 1

t1 = time.time()
print("time is :" + str(t1-t0))

TPcount = 0
TNcount = 0
FPcount = 0
FNcount = 0

for i in range(len(X_test)):
	pred = 0
	for j in range(30):
		pred += w[j]*X_test[i][j] 
	if pred*Y_test[i]>=0:
		if pred >= 0:
			TPcount += 1
		else:
			TNcount += 1
	else:
		if pred >= 0:
			FPcount += 1
		else:
			FNcount += 1 
confusion_matrix = [[TPcount,FNcount],[FPcount,TNcount]]
print("The confusion_matrix is")
print([TPcount,FNcount])
print([FPcount,TNcount])