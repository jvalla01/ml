# following datacamp tutorial exactly for first function, then doing same method with MLP model

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
import graphviz


def testing_ml():
    """
    Testing three models as per the tutorial exactly
    """

    X_train = pd.read_csv(Path("data/training_x.csv"))
    y_train = pd.read_csv(Path("data/training_y.csv"))
    X_test = pd.read_csv(Path("data/test_x.csv"))
    y_test = pd.read_csv(Path("data/test_y.csv"))
    

    # Instnatiating the models 
    logistic_regression = LogisticRegression()
    svm = SVC()
    tree = DecisionTreeClassifier()

    # Training the models 
    logistic_regression.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    # Making predictions with each model
    log_reg_preds = logistic_regression.predict(X_test)
    svm_preds = svm.predict(X_test)
    tree_preds = tree.predict(X_test)


    # Store model predictions in a dictionary
    # this makes it easier to iterate through each model
    # and print the results. 
    model_preds = {
        "Logistic Regression": log_reg_preds,
        "Support Vector Machine": svm_preds,
        "Decision Tree": tree_preds
    }

    for model, preds in model_preds.items():
        print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")

    return

testing_ml()

"""
Logistic Regression Results:
              precision    recall  f1-score   support

           0       0.76      0.92      0.83        77
           1       0.90      0.70      0.79        77

    accuracy                           0.81       154
   macro avg       0.83      0.81      0.81       154
weighted avg       0.83      0.81      0.81       154

Support Vector Machine Results:
              precision    recall  f1-score   support

           0       0.89      0.87      0.88        77
           1       0.87      0.90      0.88        77

    accuracy                           0.88       154
   macro avg       0.88      0.88      0.88       154
weighted avg       0.88      0.88      0.88       154

Decision Tree Results:
              precision    recall  f1-score   support

           0       0.83      0.84      0.84        77
           1       0.84      0.83      0.84        77

    accuracy                           0.84       154
   macro avg       0.84      0.84      0.84       154
weighted avg       0.84      0.84      0.84       154
"""

# The results aren't too bad but not incredible


def testing_nn():
    """
    """

    X_train = pd.read_csv(Path("data/training_x.csv"))
    y_train = pd.read_csv(Path("data/training_y.csv"))
    y_train = y_train.squeeze()
    X_test = pd.read_csv(Path("data/test_x.csv"))
    y_test = pd.read_csv(Path("data/test_y.csv"))
    y_test = y_test.squeeze()


    model = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

    nn_preds = model.predict(X_test)

    print(f"Neural Network Results:\n{classification_report(y_test, nn_preds)}", sep="\n\n")

    return

testing_nn()


"""
Neural Network Results:
              precision    recall  f1-score   support

           0       0.80      0.87      0.83        77
           1       0.86      0.78      0.82        77

    accuracy                           0.82       154
   macro avg       0.83      0.82      0.82       154
weighted avg       0.83      0.82      0.82       154
"""
def testing_rf():
    """
    """

    X_train = pd.read_csv(Path("data/training_x.csv"))
    y_train = pd.read_csv(Path("data/training_y.csv"))
    y_train = y_train.squeeze()
    X_test = pd.read_csv(Path("data/test_x.csv"))
    y_test = pd.read_csv(Path("data/test_y.csv"))
    y_test = y_test.squeeze()

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)

    print(f"Random forest Results:\n{classification_report(y_test, rf_preds)}", sep="\n\n")

    return

testing_rf()

"""
Random forest Results:
              precision    recall  f1-score   support

           0       0.91      0.91      0.91        77
           1       0.91      0.91      0.91        77

    accuracy                           0.91       154
   macro avg       0.91      0.91      0.91       154
weighted avg       0.91      0.91      0.91       154
"""

def checking_rf():
    """
    """

    X_train = pd.read_csv(Path("data/training_x.csv"))
    y_train = pd.read_csv(Path("data/training_y.csv"))
    y_train = y_train.squeeze()
    X_test = pd.read_csv(Path("data/validation_x.csv"))
    y_test = pd.read_csv(Path("data/validation_y.csv"))
    y_test = y_test.squeeze()

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)

    print(f"Random forest Results:\n{classification_report(y_test, rf_preds)}", sep="\n\n")

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
        filename = f"tree_{i}.png"
        graph.render(filename=filename, format='png', cleanup=True)
        print(f"Saved tree {i} as {filename}")

    return

#checking_rf()

"""
Random forest Results:
              precision    recall  f1-score   support

           0       0.93      0.94      0.94        70
           1       0.95      0.94      0.95        85

    accuracy                           0.94       155
   macro avg       0.94      0.94      0.94       155
weighted avg       0.94      0.94      0.94       155

Saved tree 0 as tree_0.png
Saved tree 1 as tree_1.png
Saved tree 2 as tree_2.png
"""