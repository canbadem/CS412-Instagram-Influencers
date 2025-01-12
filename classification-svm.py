import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack  # To combine sparse matrices

import common

print("=====================================")
print("Classification SVM")
print("=====================================")

#Set up vectorizer using 5000 features for posts and 500 features for bios
#posts have a weight of 1 and bios have a weight of 0.2

bioWeight = 0.2
postWeight = 1
bioFeatures = 500
postFeatures = 5000

x_combined_train, y_train, x_combined_test, df_tfidf, train_usernames, test_usernames, x_post_train, x_bio_train, x_post_test, x_bio_test, combined_feature_names = common.VectorizeDatasetSeperately(postFeatures,bioFeatures,postWeight,bioWeight, common.UserAggregatorPosts)

x_train, x_val, y_train, y_val = train_test_split(df_tfidf, y_train, test_size=0.2, stratify=y_train)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

model = SVC(kernel='linear', C=1, random_state=42)

doParamSearch = False

if doParamSearch:
    param_grid = {'C': [0.1, 0.5, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=10)
    grid_search.fit(x_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_

model.fit(x_train, y_train)

#@title Train Data
y_train_pred = model.predict(x_train)

print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred, zero_division=0))

#@title Validation Data
y_val_pred = model.predict(x_val)

print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

#run test data
test_unames = []
with open(common.test_classification_path, "rt") as fh:
  for line in fh:
    test_unames.append(line.strip())

print(test_unames[:5])

x_test_combined = []


for uname in test_unames:
    try:
        # Find the index in the test data for the username
        index = test_usernames.index(uname)

        # Extract caption and bio features for the current username
        caption_features = x_post_test[index]  # Caption features (sparse matrix from x_post_test)
        bio_features = x_bio_test[index]  # Bio features (sparse matrix from x_bio_test)

        # Apply weights to both caption and bio features and combine using hstack
        combined_features = hstack([caption_features.multiply(postWeight), 
                                    bio_features.multiply(bioWeight)])

        # Append the combined features to x_test_combined
        x_test_combined.append(combined_features)

    except Exception as e:
        try:
            # If username is not in test_usernames, check in train_usernames
            index = train_usernames.index(uname)

            # Extract caption and bio features for the current username
            caption_features = x_post_train[index]  # Caption features (sparse matrix from x_post_train)
            bio_features = x_bio_train[index]  # Bio features (sparse matrix from x_bio_train)

            # Apply weights to both caption and bio features and combine using hstack
            combined_features = hstack([caption_features.multiply(postWeight), 
                                        bio_features.multiply(bioWeight)])

            # Append the combined features to x_test_combined
            x_test_combined.append(combined_features)

        except Exception as e:
            # If username not found in both, print the username
            print(uname)

if "screenname" in test_unames:
    test_unames.remove("screenname")

from scipy.sparse import vstack
import json

# Convert list of sparse matrices to a single sparse matrix
x_test_combined_matrix = vstack(x_test_combined)

# Convert to dense array if needed for the model
x_test_df = pd.DataFrame(x_test_combined_matrix.toarray(), columns=combined_feature_names)

# Make predictions
predictions = model.predict(x_test_df)

# Create dictionary of predictions
prediction_dict = {}
for username, pred in zip(test_unames, predictions):
    prediction_dict[username] = pred

# Save predictions to JSON file
with open('prediction-classification-round.json', 'w', encoding='utf-8') as f:
    json.dump(prediction_dict, f, ensure_ascii=False, indent=4)

# Optional: Print some statistics
print(f"Number of predictions made: {len(predictions)}")
print("\nPrediction distribution:")
print(pd.Series(predictions).value_counts())