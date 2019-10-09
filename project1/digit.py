import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df_train = pd.read_csv("./digit/train.csv")
print(df_train.shape)
df_test = pd.read_csv("./digit/test.csv")
print(df_test.shape)
y_train = df_train['label'].to_numpy()
del df_train['label']
X_train = df_train.to_numpy()
X_test = df_test.to_numpy()
X_train = X_train/255
#X_test = X_test/255
for i in range(20):
	plt.imshow(X_train[i].reshape((28, 28)))
	plt.title(y_train[i])
	plt.show()
