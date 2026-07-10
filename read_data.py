import pandas as pd

def read_csv():
    df = pd.read_csv('data/dataset02.csv')
    return df

def read_data_merged():
    df1 = pd.read_csv('data/dataset01.csv')
    df2 = pd.read_csv('data/dataset02.csv')
    df2['label'] = df2['label'].astype(int)
    df3 = pd.read_csv('data/dataset03.csv')
    df4 = pd.read_csv('data/dataset04.csv')
    df5 = pd.read_csv('data/dataset05.csv')

    merged_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

    return merged_df

read_data_merged()