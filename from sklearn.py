from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
import graphviz



X_train = pd.read_csv(Path("data/training_x.csv"))
y_train = pd.read_csv(Path("data/training_y.csv"))
y_train = y_train.squeeze()
X_test = pd.read_csv(Path("data/test_x.csv"))
y_test = pd.read_csv(Path("data/test_y.csv"))
y_test = y_test.squeeze()
X_valid = pd.read_csv(Path("data/validation_x.csv"))
y_valid = pd.read_csv(Path("data/validation_y.csv"))

rf = RandomForestClassifier()
param_dist = {'n_estimators': randint(50,500)}

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=20, 
                                 cv=5)

rand_search.fit(X_train, y_train)

rf_preds = rand_search.predict(X_test)

print(f"Random forest Results:\n{classification_report(y_test, rf_preds)}", sep="\n\n")


rf_preds = rand_search.predict(X_valid)

print(f"Random forest Results:\n{classification_report(y_valid, rf_preds)}", sep="\n\n")

