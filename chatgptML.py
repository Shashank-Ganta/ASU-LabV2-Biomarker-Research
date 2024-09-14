import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Define threshold for binarization
THRESHOLD = 2.0

# Load data from csv file
data = pd.read_csv("C:\\Users\\shash\\VS Code Programs\\Python\\ResearchWork\\Gastric Cancer Data\\gastricSBATCHFiles\\chatgptMLData.csv", index_col=0)
print("run")
data.transpose()
# Binarize data based on threshold
data = np.where(data >= THRESHOLD, 1, 0)

# Define number of folds
NUM_FOLDS = 5

# Initialize sensitivity and specificity arrays
sensitivity = np.zeros(data.shape[0])
specificity = np.zeros(data.shape[0])

# Initialize array to store biomarker scores
biomarker_scores = np.zeros(data.shape[0])
# Get the indices of the top 30 biomarkers
top_biomarkers = np.argsort(biomarker_scores)[-10:]

# Split data into train and test sets for each fold
skf = StratifiedKFold(n_splits=NUM_FOLDS)
predictions = []
true_labels = []
for train_index, test_index in skf.split(data, np.zeros(data.shape[0])):
    train_data = data[train_index][:, top_biomarkers]
    train_labels = np.zeros((len(train_index), 2))
    train_labels[:, 0] = 1
    train_labels[train_index == test_index[0], :] = [0, 1]

    test_data = data[test_index][:, top_biomarkers]
    test_label = 0
    if np.sum(data[test_index]) > 0:
        test_label = 1

    # Train Random Forest model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(train_data, train_labels)

    # Make predictions on test set
    predictions.append(clf.predict_proba(test_data)[:, 1])
    true_labels.append(test_label)

# Concatenate predictions and true labels for all folds
predictions = np.concatenate(predictions)
true_labels = np.concatenate(true_labels)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, predictions)
roc_auc = auc(fpr, tpr)

# Print AUC
print("AUC:", roc_auc)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
