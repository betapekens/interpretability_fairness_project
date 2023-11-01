from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import numpy
import pandas as pd


def train(df):
    """
    Train an XGBoost classifier on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with features and labels.

    Returns:
    XGBClassifier: The trained XGBoost classifier.
    """
    print("Training ...")
    X = df.drop("Class/ASD", axis=1)
    y = df["Class/ASD"]
    model = XGBClassifier(max_depth=5, tree_method="hist", device="cuda",
                          objective="binary:logistic",eval_metric="auc",
                          random_state=42)
    model.fit(X, y)
    return model


def cross_validation(model, df):
    """
    Perform cross-validation on a model using ROC AUC as the scoring metric.

    Parameters:
    model: The machine learning model to be cross-validated.
    df (pd.DataFrame): The DataFrame with features and labels.

    Prints:
    The individual ROC AUC scores and the mean cross-validation score.
    """
    X = df.drop("Class/ASD", axis=1)
    y = df["Class/ASD"]
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
    with open(path, 'wb') as file:
        pickle.dump(model, file)