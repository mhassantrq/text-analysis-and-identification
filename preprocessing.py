from read_data import read_csv
from collections import defaultdict


def train_test_split(data, train_size=0.8):
    generated_rows = data[data['label'] == 1]
    written_rows = data[data['label'] == 0]
    
    # generated_rows = generated_rows.sample(frac=1)
    # written_rows = written_rows.sample(frac=1)

    train_data = []
    test_data = []

    train_size_generated = int(len(generated_rows) * train_size)
    train_size_written = int(len(written_rows) * train_size)

    train_data_generated = generated_rows.iloc[:train_size_generated]
    train_data_written = written_rows.iloc[:train_size_written]

    test_data_generated = generated_rows.iloc[train_size_generated:]
    test_data_written = written_rows.iloc[train_size_written:]

    train_data_generated = train_data_generated['text']
    train_data_written = train_data_written['text']

    test_data_generated = test_data_generated['text']
    test_data_written = test_data_written['text']

    return train_data_generated, train_data_written, test_data_generated, test_data_written

def lower_case(data):
    for i in range(len(data)):
        data.iloc[i] = data.iloc[i].lower()
    return data


data = read_csv()

train_data_generated, train_data_written, test_data_generated, test_data_written = train_test_split(data)

train_data_generated = lower_case(train_data_generated)
train_data_written = lower_case(train_data_written)

test_data_generated = lower_case(test_data_generated)
test_data_written = lower_case(test_data_written)
