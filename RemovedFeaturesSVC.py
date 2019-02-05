import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC

def load(filepath):
    raw_data = pd.read_csv(filepath)
    #Create a data frame for column storage
    processed_columns = pd.DataFrame({})
    for col in raw_data:
        #col_datatype = raw_data[col].dtype
        #Check the column for dtype object or unique value < 20
        if raw_data[col].dtype == 'object' or raw_data[col].nunique() < 20:
            df = pd.get_dummies(raw_data[col], prefix=col)
            processed_columns = pd.concat([processed_columns, df], axis=1)
        else:
            processed_columns = pd.concat([processed_columns, raw_data[col]], axis=1)
    return processed_columns

#data = load("data_lab5.csv")

df = load("trainingset.csv")
df.drop(['rowIndex'],1, inplace=True)
df.drop(['feature1'],1, inplace=True)
df.drop(['feature2'],1, inplace=True)
df.drop(['feature6'],1, inplace=True)
df.drop(['feature8'],1, inplace=True)
df.drop(['feature10'],1, inplace=True)

td = load("testset.csv")
td.drop(['rowIndex'],1, inplace=True)
td.drop(['feature1'],1, inplace=True)
td.drop(['feature2'],1, inplace=True)
td.drop(['feature6'],1, inplace=True)
td.drop(['feature8'],1, inplace=True)
td.drop(['feature10'],1, inplace=True)

#td = np.array(td)
X = np.array(df.drop(['ClaimAmount'],1))
Y = np.array(df['ClaimAmount'])

y_classify = np.where(Y > 0, 1, 0, )

total = []

cValue = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1,5,10,100,1000]

for c in cValue:
    X_train, X_test, Y_train, Y_test = train_test_split(X,y_classify,test_size=0.25, shuffle=False)
    
    clf = SVC(C=c)
    clf.fit(X_train, Y_train)
    CV_mae = cross_val_score(estimator = clf, X=X_train, y=Y_train, cv = 10)
    CV_mae = np.mean(CV_mae)
    
    #predictTest = clf.predict(td)
    predictTrain = clf.predict(X_test)
    
    count = 0
    numberOfY = 0
    for x in range(len(predictTrain)):
        if predictTrain[x] == 1 and Y_test[x] == 1:
            count = count + 1
        if Y_test[x] == 1:
            numberOfY = numberOfY + 1
    
    print("{} c. accuracy = {}".format(c,count/numberOfY))

#cv1 = np.array([0.95219958, 0.95239002, 0.95239002, 0.95219048, 0.95219048, 0.95219048,
# 0.952, 0.95237188, 0.95218137, 0.95237188])
#
#cv0p1 = np.array([0.95219958, 0.95219958, 0.95219958, 0.95219048, 0.95219048, 0.95219048, 0.95219048,
#                  0.95237188, 0.95237188, 0.95237188])
#
#cv0ppp1 = np.array([0.95219958, 0.95219958, 0.95219958, 0.95219048, 0.95219048, 0.95219048,
#                    0.95219048, 0.95237188, 0.95237188, 0.95237188])

#print(np.mean(cv0ppp1))
#
#print(np.mean(cv0p1))
#
#print(np.mean(cv1))

#cValue = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

#for c in cValue:
#print("CVMae for c = {} is {}".format(c,np.mean(CV_mae)))
    
#1 neighbors. accuracy = 0.9335428571428571
#3 neighbors. accuracy = 0.9404571428571429
#5 neighbors. accuracy = 0.9472
#7 neighbors. accuracy = 0.9498285714285715
#9 neighbors. accuracy = 0.9514285714285714
#11 neighbors. accuracy = 0.9522285714285714
#13 neighbors. accuracy = 0.9525142857142858
#15 neighbors. accuracy = 0.9526857142857142
#17 neighbors. accuracy = 0.9526857142857142
#19 neighbors. accuracy = 0.9526857142857142
#21 neighbors. accuracy = 0.9526857142857142
#23 neighbors. accuracy = 0.9526857142857142
#25 neighbors. accuracy = 0.9526857142857142
#27 neighbors. accuracy = 0.9526857142857142
#29 neighbors. accuracy = 0.9526857142857142


#testPredict = clf.predict(td)
#CVMae for c = 0.0001 is [0.95219958 0.95219958 0.95219958 0.95219048 0.95219048 0.95219048
# 0.95219048 0.95237188 0.95237188 0.95237188]
#CVMae for c = 0.001 is [0.95219958 0.95219958 0.95219958 0.95219048 0.95219048 0.95219048
# 0.95219048 0.95237188 0.95237188 0.95237188]
#CVMae for c = 0.01 is [0.95219958 0.95219958 0.95219958 0.95219048 0.95219048 0.95219048
# 0.95219048 0.95237188 0.95237188 0.95237188]
#CVMae for c = 0.1 is [0.95219958 0.95219958 0.95219958 0.95219048 0.95219048 0.95219048
# 0.95219048 0.95237188 0.95237188 0.95237188]
#CVMae for c = 1 is [0.95219958 0.95239002 0.95239002 0.95219048 0.95219048 0.95219048
# 0.952      0.95237188 0.95218137 0.95237188]
#CVMae for c = 5 is [0.95143782 0.95067606 0.95143782 0.9512381  0.952      0.95180952
# 0.95085714 0.95027624 0.95122881 0.95084778]
#CVMae for c = 10 is [0.94953342 0.9489621  0.94972386 0.94952381 0.95047619 0.95028571
# 0.94952381 0.94932368 0.95008573 0.95065727]
#CVMae for c = 1e-06 is 0.9522476288932946
#CVMae for c = 1e-05 is 0.9522476288932946
#CVMae for c = 0.0001 is 0.9522476288932946
#CVMae for c = 0.001 is 0.9522476288932946
#CVMae for c = 0.01 is 0.9522476288932946
#CVMae for c = 0.1 is 0.9522476288932946

