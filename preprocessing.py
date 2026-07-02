import read_data
from collections import defaultdict


def split_train_test(data, train_size=0.8):
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

def split_features(data):
    # will act as a basic tokenizer
    for i in range(len(data)):
        data.iloc[i] = data.iloc[i].split()
    return data

def bag_of_words(data):
    bag = defaultdict(int)
    for doc in data:
        for w in doc:
            bag[w] += 1    
    return bag


data = read_data.read_csv()

train_data_generated, train_data_written, test_data_generated, test_data_written = split_train_test(data)

train_data_generated = lower_case(train_data_generated)
train_data_written = lower_case(train_data_written)

test_data_generated = lower_case(test_data_generated)
test_data_written = lower_case(test_data_written)

train_data_generated = split_features(train_data_generated)
train_data_written = split_features(train_data_written)

test_data_generated = split_features(test_data_generated)
test_data_written = split_features(test_data_written)



tg = bag_of_words(train_data_generated)
print(tg['the'])