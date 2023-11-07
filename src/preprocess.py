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
        cat_cols = ["jaundice", "autism", "used_app_before", "age_desc"]
    for c in cat_cols:
        encoder = LabelEncoder()
        df[c] = encoder.fit_transform(df[c])
    return df


def onehot_encode(
    df: pd.DataFrame, categorical_cols: List[str] = None
) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = ["country_of_res", "ethnicity", "relation", "gender"]
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


def preprocess_text(df, text_cols=None):
    if text_cols is None:
        text_cols = ["ethnicity", "country_of_res"]
    for col in text_cols:
        df[col] = df[col].str.lower().str.strip()
        df[col] = df[col].str.replace(" ", "_")
    return df


def feature_engineering(df):
    df["sum_scores"] = df.filter(regex="_Score").sum(axis=1)
    return df


def replace_infreq_vals(df, columns=None, threshold=10):
    if columns is None:
        columns = ["country_of_res", "ethnicity"]
    for colname in columns:
        category_counts = df[colname].value_counts()
        infrequent_categories = category_counts[category_counts < threshold].index
        df[colname] = df[colname].apply(lambda x: 'other' if x in infrequent_categories else x)
    return df


def preprocess(df):
    rename_mapping = {
        "austim": "autism",
        "contry_of_res": "country_of_res",
    }
    df.rename(rename_mapping, axis=1, inplace=True)
    df = preprocess_text(df)
    df = replace_infreq_vals(df)
    df = impute(df)
    df = label_encoder(df)
    df = onehot_encode(df)
    df = feature_engineering(df)
    # df = scale(df)
    X = df.drop("Class/ASD", axis=1).copy()
    y = df["Class/ASD"]
    return X, y