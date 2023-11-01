from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

def encoder(df):
    """
    Encode categorical columns in a DataFrame using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The DataFrame containing categorical columns to be encoded.

    Returns:
    pd.DataFrame: The DataFrame with categorical columns encoded.
    """
    print("Encoding variables ...")
    cat = df.select_dtypes(['object']).columns 
    encoding_dict = dict()
    for c in cat:
        encoder = LabelEncoder()
        df[c] = encoder.fit_transform(df[c])
    return df


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