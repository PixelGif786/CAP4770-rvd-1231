import pandas as pd
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

bach = pd.read_csv('bach.csv')

label_encoder = preprocessing.LabelEncoder()

for col in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','bass']:
    # Encode the labels in the current column
    bach[col] = label_encoder.fit_transform(bach[col])

bach = bach.drop(bach.columns[0], axis=1)

bFeatures = bach[['event_number','C','C#','D','D#','E','F','F#','G','G#','A','A#','B','bass','meter']]
bLabels = bach['chord_label']

# Split the data into training and test sets
bach_features_train, bach_features_test, bach_label_train, bach_label_test = train_test_split(bFeatures, bLabels, test_size=0.2, random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
}

# Create a grid search object
grid_search = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(criterion='entropy'),
    param_grid=param_grid,
    cv=10, # use 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1, # use all available CPU cores
)

# Fit the grid search object to the training data
grid_search.fit(bach_features_train, bach_label_train)

# Print the best hyperparameters and the corresponding accuracy score
print(grid_search.best_params_)
print(grid_search.best_score_)

# Train a new decision tree classifier with the best hyperparameters
clf1 = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
)
clf1.fit(bach_features_train, bach_label_train)

# Evaluate the accuracy of the classifier on the test set
bach_predictions = clf1.predict(bach_features_test)
print(accuracy_score(bach_label_test, bach_predictions))

