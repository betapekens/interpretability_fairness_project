from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import numpy
import pandas as pd
import os


def train(X, y, val_split: bool=False, **model_params):
    """
    Train an XGBoost classifier on the given DataFrame.

    Parameters:
    X (pd.DataFrame): features
    y (pd.Series): labels

    Returns:
    XGBClassifier: The trained XGBoost classifier.
    """
    print("Training ...")
    model = XGBClassifier(**model_params)
    if val_split:
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("roc-auc score: ", roc_auc_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        return model
    model.fit(X, y)
    return model




def cross_validation(model, X, y):
    """
    Perform cross-validation on a model using ROC AUC as the scoring metric.

    Parameters:
    model: The machine learning model to be cross-validated.
    df (pd.DataFrame): The DataFrame with features and labels.

    Prints:
    The individual ROC AUC scores and the mean cross-validation score.
    """
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", )
    print(f"The individual scores are:{scores}")
    print(f"The mean cross validation score is:{scores.mean()}")


def save_model(model, path):
    """
    Save a machine learning model to a file using Pickle.

    Parameters:
    model: The machine learning model to be saved.
    path (str): The file path where the model will be saved.    
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    """
    Load a machine learning model from a file using Pickle.

    Parameters:
    path (str): The file path from which the model will be loaded.

    Returns:
    model: The loaded machine learning model.
    """
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model
