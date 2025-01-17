import os

import pandas as pd


def load_data(filename='data/INVEST_clean.csv'):
    df = pd.read_csv(filename, sep=',')
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_benchmark_data(index_code, directory='data/INVEST_IRESS'):
    """
       Loads and returns a dataframe containing benchmark data
    """
    df = pd.read_csv(os.path.join(directory, index_code + '.csv'), delimiter=';')
    return df.reindex(index=df.index[::-1])