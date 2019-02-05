import pandas as pd
import featureSelection
import kNN
from sklearn.neighbors import KNeighborsClassifier

# Summary
# --------------------------------------------------
# kNN to classify features into binary claim amounts


# Step 1 - Set Tuning Parameters
# --------------------------------------------------
k_neighbours = 3


# Step 2 - Setup Data
# --------------------------------------------------
training_data = pd.read_csv('./data/trainingset.csv')
training_data = training_data.drop(['rowIndex'], axis=1)
training_data_features = training_data.drop(['ClaimAmount'], axis=1)
training_data_labels = training_data.loc[:, ['ClaimAmount']]

test_data = pd.read_csv('./data/testset.csv')
submission = test_data.loc[:, ['rowIndex']]
test_data_features = test_data.drop(['rowIndex'], axis=1)

training_data_labels_binary = featureSelection.convert_claim_amount_binary(training_data_labels)


# Step 3 - Find MAE
# --------------------------------------------------
kNN.knn_mae(training_data, k_neighbours)


# Step 4 - Predict Test Labels
# --------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=k_neighbours)
knn_model.fit(training_data_features, training_data_labels_binary.values.ravel())
predicted_validation_labels = knn_model.predict(test_data_features)
test_data_labels_binary = pd.DataFrame(data=predicted_validation_labels)



# Step 5 - Print a CSV file
# --------------------------------------------------
#submission_csv = pd.concat([submission, test_data_labels_binary], axis=1, join='inner')
#submission_csv.columns = ["rowIndex", "ClaimAmount"]
#submission_csv.to_csv("testsetassessment_3_1.csv", index=False)

