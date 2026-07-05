import pandas as pd

def read_csv():
    df = pd.read_csv('data/dataset02.csv')
    df = df[:10000]
    return df
