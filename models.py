import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier

dataset = pd.read_json(r'data/word2vec_data.json', lines=True)

x_train, x_test, y_train, y_test = train_test_split(dataset['word2vec'], dataset['category'], test_size=0.2, random_state=42)

x_train = [x for x in x_train]
x_test = [x for x in x_test]

y_train = [x for x in y_train]
y_test = [x for x in y_test]

# Random Classifier 
rand = DummyClassifier(strategy = "uniform")
rand.fit(x_train, y_train)
rand_predictions = rand.predict(x_test)
rand_accuracy = accuracy_score(y_test, rand_predictions)
rand_precision = precision_score(y_test, rand_predictions, average='weighted')
rand_recall = recall_score(y_test, rand_predictions, average='weighted')
rand_f1 = f1_score(y_test, rand_predictions, average='weighted')
print("Random Classifier Accuracy:", rand_accuracy)

# Majority Classifier 
majority = DummyClassifier(strategy = "most_frequent")
majority.fit(x_train, y_train)
majority_predictions = majority.predict(x_test)
majority_accuracy = accuracy_score(y_test, majority_predictions)
majority_precision = precision_score(y_test, majority_predictions, average='weighted')
majority_recall = recall_score(y_test, majority_predictions, average='weighted')
majority_f1 = f1_score(y_test, majority_predictions, average='weighted')
print("Majority Classifier Accuracy:", majority_accuracy)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=128)
knn.fit(x_train, y_train)
knn_predictions = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')
print("KNN Accuracy:", knn_accuracy)

# Naive Bayes Classifier
bayes = GaussianNB()
bayes.fit(x_train, y_train)
bayes_predictions = bayes.predict(x_test)
bayes_accuracy = accuracy_score(y_test, bayes_predictions)
bayes_precision = precision_score(y_test, bayes_predictions, average='weighted')
bayes_recall = recall_score(y_test, bayes_predictions, average='weighted')
bayes_f1 = f1_score(y_test, bayes_predictions, average='weighted')
print("Naive Bayes Accuracy:", bayes_accuracy)

# Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth = 10)
tree.fit(x_train, y_train)
tree_predictions = tree.predict(x_test)
tree_accuracy = accuracy_score(y_test, tree_predictions)
tree_precision = precision_score(y_test, tree_predictions, average='weighted')
tree_recall = recall_score(y_test, tree_predictions, average='weighted')
tree_f1 = f1_score(y_test, tree_predictions, average='weighted')
print("Decision Tree Accuracy:", tree_accuracy)

# Hard Voting Classifier
hard_voting = VotingClassifier(estimators=[('rc', rand), ('mc', majority), ('tree', tree), ('knn', knn), ('nb', bayes)], voting='hard')
hard_voting.fit(x_train, y_train)
hard_voting_predictions = hard_voting.predict(x_test)
hard_voting_accuracy = accuracy_score(y_test, hard_voting_predictions)
hard_voting_precision = precision_score(y_test, hard_voting_predictions, average='weighted')
hard_voting_recall = recall_score(y_test, hard_voting_predictions, average='weighted')
hard_voting_f1 = f1_score(y_test, hard_voting_predictions, average='weighted')
print("Hard Voting Accuracy:", hard_voting_accuracy)

# Soft Voting Classifier
soft_voting = VotingClassifier(estimators=[('rc', rand), ('mc', majority), ('tree', tree), ('knn', knn), ('nb', bayes)], voting='soft')
soft_voting.fit(x_train, y_train)
soft_voting_predictions = soft_voting.predict(x_test)
soft_voting_accuracy = accuracy_score(y_test, soft_voting_predictions)
soft_voting_precision = precision_score(y_test, soft_voting_predictions, average='weighted')
soft_voting_recall = recall_score(y_test, soft_voting_predictions, average='weighted')
soft_voting_f1 = f1_score(y_test, soft_voting_predictions, average='weighted')
print("Soft Voting Accuracy:", soft_voting_accuracy)

# Bagging Classifier
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=16, n_jobs=-1)
bagging.fit(x_train, y_train)
bagging_predictions = bagging.predict(x_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
bagging_precision = precision_score(y_test, bagging_predictions, average='weighted')
bagging_recall = recall_score(y_test, bagging_predictions, average='weighted')
bagging_f1 = f1_score(y_test, bagging_predictions, average='weighted')
print("Bagging Accuracy:", bagging_accuracy)

# Random Forest Classifier
rf = RandomForestClassifier(verbose=0, n_jobs=-1) # TURN OFF/ON VERBOSE
rf.fit(x_train, y_train)
rf_predictions = rf.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
print("Random Forest Accuracy:", rf_accuracy)

# Ada Boost Classifier
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=1)
adaboost.fit(x_train, y_train)
adaboost_predictions = adaboost.predict(x_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)
adaboost_precision = precision_score(y_test, adaboost_predictions, average='weighted')
adaboost_recall = recall_score(y_test, adaboost_predictions, average='weighted')
adaboost_f1 = f1_score(y_test, adaboost_predictions, average='weighted')
print("Ada Boost Accuracy:", adaboost_accuracy)

# XGBoost Classifier
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)

xgboost = XGBClassifier(n_estimators=512, max_depth=3, n_jobs=-1)
xgboost.fit(x_train, y_train_encoded)
xgboost_predictions = xgboost.predict(x_test)
xgboost_accuracy = accuracy_score(y_test_encoded, xgboost_predictions)
xgboost_precision = precision_score(y_test_encoded, xgboost_predictions, average='weighted')
xgboost_recall = recall_score(y_test_encoded, xgboost_predictions, average='weighted')
xgboost_f1 = f1_score(y_test_encoded, xgboost_predictions, average='weighted')
print("XGBoost Accuracy:", xgboost_accuracy)

models = ["Random", "Majority", "KNN", "NaiveBayes", "DecisionTree", "HardVoting", "SoftVoting", "Bagging", "RandomForest", "AdaBoost", "XGBoost"]
accuracy_scores = [rand_accuracy, majority_accuracy, knn_accuracy, bayes_accuracy, tree_accuracy, hard_voting_accuracy, soft_voting_accuracy, bagging_accuracy, rf_accuracy, adaboost_accuracy, xgboost_accuracy]
precision_scores = [rand_precision, majority_precision, knn_precision, bayes_precision, tree_precision, hard_voting_precision, soft_voting_precision, bagging_precision, rf_precision, adaboost_precision, xgboost_precision]
recall_scores = [rand_recall, majority_recall, knn_recall, bayes_recall, tree_recall, hard_voting_recall, soft_voting_recall, bagging_recall, rf_recall, adaboost_recall, xgboost_recall]
f1_scores = [rand_f1, majority_f1, knn_f1, bayes_f1, tree_f1, hard_voting_f1, soft_voting_f1, bagging_f1, rf_f1, adaboost_f1, xgboost_f1]

plt.plot(accuracy_scores)
plt.show()

plt.plot(precision_scores)
plt.show()

plt.plot(recall_scores)
plt.show()

plt.plot(f1_scores)
plt.show()


