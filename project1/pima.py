import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import time
def timer(func):
	def newfunc(*args, **kws):
		t0 = time.time()
		result = func(*args, **kws)
		t1 = time.time()
		print(t1 - t0)
	return newfunc

diabetes_data = pd.read_csv("./pima/diabetes.csv")
print(diabetes_data.head())
print(diabetes_data.describe())
'''p = diabetes_data.hist(figsize = (12,12))
plt.show()
plt.figure(figsize=(12,12))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')
plt.show()'''
diabetes_data_copy = diabetes_data.copy(deep = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(diabetes_data_copy.isnull().sum())
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].mean(), inplace = True)
X = preprocessing.normalize(diabetes_data_copy.drop(["Outcome"],axis = 1))
y = diabetes_data_copy.Outcome

#print(y)
@timer
def KNN(data,label,K):
	predictlist =[]
	for i in range(len(data)):
		distancelist = []
		for j in range(len(data)):
			if i != j:
				distance = Chebychev(data[i],data[j])
				distancelist.append((j,distance))
		resultlist = sorted(distancelist, key = lambda x :x[1])[:K]
		predict = np.around((sum([label[key[0]] for key in resultlist])/K))
		predictlist.append(predict)
	return predictlist

def counting(predictlist,label):
	total = len(predictlist)
	count = 0
	for i in range(total):
		if predictlist[i] == label[i]:
				count += 1
	print(count)
	return count/total



def Manhattan(data1,data2):
	return sum(abs(data1-data2))
def Euclidean(data1,data2):
	return np.linalg.norm(data1-data2)
def Chebychev(data1,data2):
	return max(abs(data1-data2))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/5,random_state=42, stratify=y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

test_scores = []
train_scores = []

for i in range(1,6):
	train_predict = KNN(X_train,y_train,i)
	test_predict = KNN(X_test,y_test,i)
	train_scores.append(counting(train_predict,y_train))
	test_scores.append(counting(test_predict,y_test))
print(train_scores)
print(test_scores)
test_predict = KNN(X_test,y_test,2)
print(len(test_predict))
print(len(y_test))
p = confusion_matrix(y_test,test_predict)
pd.crosstab(y_test, test_predict)
print(p)
'''cnf_matrix = metrics.confusion_matrix(y_test, test_predict)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()'''
'''
for i in range(1,6):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

print(train_scores)
print(test_scores)'''
'''
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,6),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,6),test_scores,marker='o',label='Test Score')
plt.show()'''