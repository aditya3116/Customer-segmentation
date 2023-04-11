import pandas as pd
import numpy as np
import pickle 
data=pd.read_excel(r"C:\Users\Jay Rathod\Downloads\Mall_Customers.xlsx",index_col=0,header=0)
data=data[['Annual Income (k$)', 'Spending Score (1-100)','Clusters']]
data.Clusters=data.Clusters.replace({"Careless":0,"Standard":1,"Target":2,"Sensible":3,"Careful":4})

X=data.values[:,:-1]
Y=data.values[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=9,metric="euclidean")
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfmlr_knn=confusion_matrix(Y_test,Y_pred)
print("Confusion matrix by KNN  method:\n",cfmlr_knn)
print()
class_ralr_knn=classification_report(Y_test,Y_pred)
print("Classfication rep by KNN method:\n",class_ralr_knn)
print()
acc_s_knn=accuracy_score(Y_test,Y_pred)
print("Accuracy score by KNN method:",acc_s_knn)



filename = r'knn.sav'
pickle.dump(knn, open(filename,"wb"))
loaded_model = pickle.load(open(filename,"rb"))
pickle.dump(knn,open("mall.pkl","wb"))
model = pickle.load(open("mall.pkl","rb"))

