import pandas as pd


def get_dataframe_from_csv(file_name):
    df = pd.read_csv(file_name)
    return df


def save_dataframe_to_csv(df, file_name, headers=[]):
    df.to_csv(file_name, headers=headers)
