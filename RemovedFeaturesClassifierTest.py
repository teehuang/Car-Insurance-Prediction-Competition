import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score


td = pd.read_csv("testset.csv")
td.drop(['rowIndex'],1, inplace=True)

df = pd.read_csv("trainingset.csv")
df.drop(['rowIndex'],1, inplace=True)
df.drop(['feature1'],1, inplace=True)
df.drop(['feature2'],1, inplace=True)
df.drop(['feature6'],1, inplace=True)
df.drop(['feature8'],1, inplace=True)
df.drop(['feature10'],1, inplace=True)

td = np.array(td)
X = np.array(df.drop(['ClaimAmount'],1))
Y = np.array(df['ClaimAmount'])

y_classify = np.where(Y > 0, 1, 0, )

X_train, X_test, Y_train, Y_test = train_test_split(X,y_classify,test_size=0.25, shuffle=False)

clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, Y_train)
#CV_mae = cross_val_score(estimator = clf, X=X_train, y=Y_train, cv = 10)
#print(np.mean(CV_mae))
#CV_mae = np.mean(CV_mae)

testPred = clf.predict(X_test)
testPred = np.array(testPred)
testPred.reshape(len(testPred),)
Y_test = np.array(Y_test)
#
#testPred.reshape(-1,1)
#Y_test.reshape(-1,1)
#
#mae = mean_absolute_error(Y_test, testPred)
#
#count = 0
#for x in range(len(testPred)):
#    if testPred[x] == Y_test[x]:
#        count = count + 1
#
#f1 = f1_score(Y_test, testPred)
#print("{} neighors with success {}".format(j, success))

#predictTest = clf.predict(td)
#predictTrain = clf.predict(X)

rows = []
claimX_t = []
claimX = []
claimY = []

df = pd.read_csv("trainingset.csv")
df.drop(['rowIndex'],1, inplace=True)

X = np.array(df.drop(['ClaimAmount'],1))
Y = np.array(df['ClaimAmount'])

y_classify = np.where(Y > 0, 1, 0, )

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X,y_classify,test_size=0.25, shuffle=False)

#the next step is to pull the X values from the Y test

for x in range(len(td)):
    if predictTest[x] == 1:
        claimX_t.append(td[x])
        rows.append(x)
    

#train for regression
for x in range(len(X)):
    if predictTrain[x] == 1 and Y[x] > 0:
        claimX.append(X[x])
        claimY.append(Y[x])






#FOR K NEAREST NEIGHBORS

#15 neighors with success 0.9525142857142858
#17 neighors with success 0.9525714285714286
#19 neighors with success 0.9526285714285714
#21 neighors with success 0.9526857142857142
#23 neighors with success 0.9526857142857142
#25 neighors with success 0.9526857142857142
#27 neighors with success 0.9526857142857142

#for j in range(15,30,2):
#    clf = neighbors.KNeighborsClassifier(n_neighbors=j)
#    clf.fit(X_train, Y_train)
#    CV_mae = cross_val_score(estimator = clf, X=X_train, y=Y_train, cv = 10)
#    CV_mae = np.mean(CV_mae)
#    
#    testPred = clf.predict(X_test)
#    testPred = np.array(testPred)
#    testPred.reshape(len(testPred),)
#    Y_test = np.array(Y_test)
#    
#    testPred.reshape(-1,1)
#    Y_test.reshape(-1,1)
#    
#    mae = mean_absolute_error(Y_test, testPred)
#    
#    count = 0
#    for x in range(len(testPred)):
#        if testPred[x] == Y_test[x]:
#            count = count + 1
#    
#    success = count/len(testPred)
#    print("{} neighors with success {}".format(j, success))




