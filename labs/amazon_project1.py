import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_df = pd.read_csv('http://34.82.108.37/amazonReviewsTrain.csv.zip', low_memory=False)

train_df.dropna(inplace=True)
train_df = train_df.infer_objects()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['reviewText'], train_df['rating'], random_state=0)

# Convert review text to matrix of word counts
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_val_counts = vectorizer.transform(X_val)

# Train logistic regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_counts, y_train)

# Evaluate classifier on validation set
y_val_pred = clf.predict(X_val_counts)
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred, average='macro'))
print("Recall:", recall_score(y_val, y_val_pred, average='macro'))
print("F1 score:", f1_score(y_val, y_val_pred, average='macro'))

# Load test data
test_df = pd.read_csv('http://34.82.108.37/amazonReviewsTest.csv')

# Convert 'verified' column to binary feature
test_df['verified'] = test_df['verified'].map({'Y': 1, 'N': 0})

# Convert review text to matrix of word counts
test_df_counts = vectorizer.transform(test_df['reviewText'])

# Make predictions on test data
test_df_pred = clf.predict(test_df_counts)

# Save predictions to text file
with open('amazonReviewsTestPredictions.txt', 'w') as f:
    for pred in test_df_pred:
        f.write(str(pred) + '\n')

