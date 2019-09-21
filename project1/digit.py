import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
df_train = pd.read_csv("./digit/train.csv")
print(df_train.shape)
df_test = pd.read_csv("./digit/test.csv")
print(df_test.shape)
y_train = np.array(df_train['label'])
del df_train['label']
X_train = np.array(df_train)
X_test = np.array(df_test)
X_train = X_train / 255
X_test = X_test / 255
'''for i in range(21,30):
	plt.imshow(X_train[i].reshape((28, 28)))
	plt.show()'''
X_train,X_vali, y_train, y_vali = train_test_split(X_train,y_train,test_size = 0.2,random_state = 0)
ans_k = 0

k_range = range(1,8)
scores = []
def KNN(X_train,y_train,X_vali,K):
    predictlist =[]
    for i in range(len(X_vali)):
        distancelist = []
        count = [0,0,0,0,0,0,0,0,0,0]
        for j in range(len(X_train)):
            distance = sum(abs(X_train[j] - X_vali[i]))
            distancelist.append((j,distance))
        resultlist = sorted(distancelist, key = lambda x :x[1])[:K]
        for ele in resultlist:
            index = ele[0]
            count[y_train[index]] += 1
        maxnum = max(count)
        predict = count.index(maxnum)
        predictlist.append(predict)
        if i % 1000 == 0:
            print(predictlist)
    return predictlist

for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    '''knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_vali)'''
    y_pred = KNN(X_train,y_train,X_vali,k)
    accuracy = accuracy_score(y_vali,y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali, y_pred))  
    print(confusion_matrix(y_vali, y_pred))  
    
    print("Complete time: " + str(end-start) + " Secs.")
print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
p = confusion_matrix(y_train,y_pred)
print (y_pred[134])
plt.imshow(X_test[134].reshape((28, 28)))
plt.show()