import pandas as pd

def load(path):
    """
    Load data from a CSV file and return it as a DataFrame.
    """
    print("Loading data ...")
    df = pd.read_csv(path)
    return df