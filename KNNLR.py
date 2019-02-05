import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv("trainingset.csv")
td = pd.read_csv("testset.csv")

df.drop(['rowIndex'],1, inplace=True)
td.drop(['rowIndex'],1, inplace=True)

td = np.array(td)
X = np.array(df.drop(['ClaimAmount'],1))
Y = np.array(df['ClaimAmount'])

y_classify = np.where(Y > 0, 1, 0, )


X_train, X_test, Y_train, Y_test = train_test_split(X,y_classify,test_size=0.25, shuffle=False)

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)
CV_mae = cross_val_score(estimator = clf, X=X_train, y=Y_train, cv = 10)
CV_mae = np.mean(CV_mae)

predictTest = clf.predict(td)
predictTrain = clf.predict(X)

rows = []
claimX_t = []
claimX = []
claimY = [] 

print(accuracy_score(predictTrain,y_classify))
for x in range(len(td)):
    #if clf.predict(X[x].reshape(1,-1)) == 1 and Y[x] > 0:
    if predictTest[x] == 1:
        claimX_t.append(td[x])
        rows.append(x)


#train for regression
for x in range(len(X)):
    if predictTrain[x] == 1 and Y[x] > 0:
        claimX.append(X[x])
        claimY.append(Y[x])
        

claimX = np.array(claimX).reshape(len(claimX),18)
claimY = np.array(claimY).reshape(len(claimY),)

claimX_t = np.array(claimX).reshape(len(claimX),18)
X_t_test_rr = pd.DataFrame(claimX_t)

X_train, X_test, Y_train, Y_test = train_test_split(claimX,claimY,test_size=0.25, shuffle=False)

reg = LinearRegression().fit(X_train, Y_train)
testPredict = reg.predict(X_test)

#testRidge = Ridge(alpha=(10 ** lambdaValues[RRMinIndex]))
#testRidge.fit(X_train_rr, Y_train_rr)

#testPredict = testRidge.predict(X_t_test_rr)


testPredict.reshape(len(testPredict),)
emptyRows = [0]*30000

count =0 
for x in rows:
    emptyRows[x] = testPredict[count]
    count = count+1

emptyRows = np.array(emptyRows, dtype=np.float64)
submission = emptyRows
    
output = pd.DataFrame({'ClaimAmount': submission})
output.to_csv("submission.csv", index_label="rowIndex")
