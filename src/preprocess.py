from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from typing import List

def label_encoder(df, cat_cols = None):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The DataFrame containing categorical columns to be encoded.

    Returns:
    pd.DataFrame: The DataFrame with categorical columns encoded.
    """
    print("Encoding variables ...")
    if cat_cols is None:
        cat_cols = ["gender", "jaundice", "autism", "used_app_before", "age_desc"]
    for c in cat_cols:
        encoder = LabelEncoder()
        df[c] = encoder.fit_transform(df[c])
    return df


def onehot_encode(
    df: pd.DataFrame, categorical_cols: List[str] = None
) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["country_of_res", "ethnicity", "relation"]
    dicts = df[categorical_cols].to_dict(orient="records")
    dv = DictVectorizer()
    categorical_df = dv.fit_transform(dicts)
    cat_df = pd.DataFrame(categorical_df.todense(), columns=dv.get_feature_names_out(categorical_df))
    encoded_df = pd.concat([df, cat_df], axis=1)
    encoded_df = encoded_df.drop(columns=categorical_cols)
    return encoded_df.dropna(subset=["Class/ASD"])


def scale(df):
    """
    Scale continuous variables in a DataFrame using Min-Max scaling.

    Parameters:
    df (pd.DataFrame): The DataFrame containing continuous variables to be scaled.

    Returns:
    pd.DataFrame: The DataFrame with continuous variables scaled.
    """
    df = df.drop("age_desc", axis=1)
    continuous_var = df.select_dtypes(['int64']).columns
    scaler = MinMaxScaler()
    df[continuous_var] = scaler.fit_transform(df[continuous_var])
    return df


def impute(df, nan_vals="?"):
    return df.replace(nan_vals, np.nan)


def feature_engineering(df):
    pass
    return df


def preprocess(df):
    rename_mapping = {
        "austim": "autism",
        "contry_of_res": "country_of_res",
    }
    df.rename(rename_mapping, axis=1, inplace=True)
    df = label_encoder(df)
    df = onehot_encode(df)
    # df = scale(df)
    df = impute(df)

    X = df.drop("Class/ASD", axis=1).copy()
    y = df["Class/ASD"]
    return X, y