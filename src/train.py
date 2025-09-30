import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from src.dataloader import load_dataset
from src.utils import ensure_dir
import joblib


def train_model(random_state: int=42):
    # Load the dataset
    X, y, meta = load_dataset()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # Initialize the Logistic Regression model
    model = LogisticRegression(max_iter=200)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    print(f"ROC AUC Score: {roc_auc}")

    ensure_dir("artifacts/models")
    joblib.dump(model, "artifacts/models/logistic_regression_model.joblib")

    return model, X_test, y_test, meta, {"roc_auc": roc_auc, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}

## from notebook ML Studio
def notebook_train_model():
    ## Load dataset
    print("Updated method")
    X,y,meta = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    ## set regularization hyperparameter 
    reg = 0.01  
    
    ##train a logistic regression model
    print('Training a logistic regression model with regularization rate of', reg)
    model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
    
    ## Calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print(f"Accuracy:", acc)
    
    ## Calculate AUC

    # y_scores = model.predict_proba(X_test)
    # auc = roc_auc_score(y_test, y_scores[:,1])
    # print(f"AUC:", auc)
    
    ## add safe guards for multiclass classification
    auc = None
    if len(np.unique(y_test)) == 2:
        y_scores = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_scores[:,1])
        print(f"AUC:", auc)
    else:
        print("Skipping AUC calculation for multiclass classification/non-binary target")

    ## record best model
    ensure_dir("artifacts/models")
    joblib.dump(model, "artifacts/models/logistic_regression_model.joblib")
    
    return model, X_test, y_test, meta, {"accuracy": acc, "auc": auc}
