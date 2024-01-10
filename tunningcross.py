import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

import torch
import torch.nn as nn

dataset = pd.read_json(r'data/word2vec_data.json', lines=True)

x_train, x_test, y_train, y_test = train_test_split(dataset['word2vec'], dataset['category'], test_size=0.2, random_state=42)

x_train = [x for x in x_train]
x_test = [x for x in x_test]

y_train = [y for y in y_train]
y_test = [y for y in y_test]

# XGBoost Classifier Tunning
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

xgboost = XGBClassifier()

grid = {
    'n_estimators': [32, 64, 128, 256],
    'max_depth': [1, 3, 5, 7],
    'grow_policy': [0, 1],
    'booster' : ["bgtree", "bglinear", "dart"]
}

# grid_test = {
#     'n_estimators': [32, 64],
# }

tune = GridSearchCV(xgboost, grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=1)

tune.fit(x_train, y_train_encoded)
best_params = tune.best_params_

xgboost = XGBClassifier(**best_params)
xgboost.fit(x_train, y_train_encoded)

xgboost.fit(x_train, y_train_encoded)
xgboost_predictions = xgboost.predict(x_test)
xgboost_accuracy = accuracy_score(y_test_encoded, xgboost_predictions)
xgboost_precision = precision_score(y_test_encoded, xgboost_predictions, average='weighted')
xgboost_recall = recall_score(y_test_encoded, xgboost_predictions, average='weighted')
xgboost_f1 = f1_score(y_test_encoded, xgboost_predictions, average='weighted')
print("Best XGBoost Accuracy:", xgboost_accuracy, "with parameters", best_params)

x_data = dataset['word2vec']
y_data = dataset['category']

x_data = np.array([x for x in x_data])
y_data = np.array([y for y in y_data])

label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(y_data)

skfold = StratifiedKFold(n_splits=10, shuffle=True)
accuracies = []

for i, (train_mask, test_mask) in enumerate(skfold.split(x_data, y_data)):
    x_train, x_test = x_data[train_mask], x_data[test_mask]
    y_train, y_test = y_data[train_mask], y_data[test_mask]

    xgboost = XGBClassifier(**best_params)
    xgboost.fit(x_train, y_train)

    y_CV_pred = xgboost.predict(x_test)

    accuracy = accuracy_score(y_test, y_CV_pred)
    accuracies.append(accuracy)

print("Cross-Validated XGBoost Accuracy:", np.mean(accuracies))