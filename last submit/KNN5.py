import numpy as np
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

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

td = np.array(td)
X = np.array(df.drop(['ClaimAmount'],1))
Y = np.array(df['ClaimAmount'])

y_classify = np.where(Y > 0, 1, 0, )

X_train, X_test, Y_train, Y_test = train_test_split(X,y_classify,test_size=0.25, shuffle=False)

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,Y_train)
CV_mae = cross_val_score(estimator = clf, X=X_train, y=Y_train, cv = 10)

testPredict = clf.predict(td)

#testPredict.reshape(len(testPredict),)
#emptyRows = [0]*30000

submission = testPredict
 
output = pd.DataFrame({'ClaimAmount': submission})
output.to_csv("submission.csv", index_label="rowIndex")

