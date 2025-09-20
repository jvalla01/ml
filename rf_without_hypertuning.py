#random forest without hypertuning
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
from joblib import Parallel, delayed
import joblib

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
rf.fit(X_train, y_train)


rf_preds = rf.predict(X_test)

print(f"Random forest Results:\n{classification_report(y_test, rf_preds)}", sep="\n\n")


rf_preds = rf.predict(X_valid)

print(f"Random forest Results:\n{classification_report(y_valid, rf_preds)}", sep="\n\n")


#save the model
# Save the model as a pickle in a file
joblib.dump(rf, 'RF_nh.pkl')

# save the pictures of trees
for i in range(3):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=X_train.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
            # Save as PNG
        filename = f"tree_nh_{i}"
        graph.render(filename=filename, format='png', cleanup=True)
        print(f"Saved tree {i} as {filename}")


"""
Random forest Results:
              precision    recall  f1-score   support

           0       0.92      0.92      0.92        77
           1       0.91      0.91      0.92        77

    accuracy                           0.92       154
   macro avg       0.92      0.92      0.92       154
weighted avg       0.92      0.92      0.92       154

Random forest Results:
              precision    recall  f1-score   support

           0       0.92      0.94      0.93        70
           1       0.95      0.93      0.94        85

    accuracy                           0.94       155
   macro avg       0.93      0.94      0.94       155
weighted avg       0.94      0.94      0.94       155

Best hyperparameters: {'max_depth': 12, 'n_estimators': 317}
"""