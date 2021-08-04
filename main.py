#import  libraries
import  pandas as pd
import numpy as np

#Reading the dataset
dataset=pd.read_csv('social_network_ads.csv')
#dependent variable
y=dataset.iloc[:,-1].values

#independent variable
x=dataset.iloc[:,:-1].values


#Diving the dataset into training and testing set
from sklearn.model_selection  import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size = 0.1)

#feature scaling scaling the data into a scale so that none of feature get dominant by other features
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

#Logistic regressiomn
from sklearn.linear_model import LogisticRegression
classifer= LogisticRegression( )
classifer.fit(x_train,y_train)
#Logistic regressiomn predict the result
y_pred = classifer.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Visualising the Training set results
import matplotlib.pyplot  as plt
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[: , 0].max() + 1, step = 0.01),
  np.arange(start = X_set[: , 1].min() - 1, stop = X_set[: , 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
  alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# # Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[: , 0].max() + 1, step = 0.01),
#   np.arange(start = X_set[: , 1].min() - 1, stop = X_set[: , 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#   alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#   plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#     c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Classifier (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()