import joblib
import numpy as np

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
    return data.str.lower()

def split_features(data):
    return data.str.split()

def top_words_svm(model, vect, count):
    words = vect.get_feature_names_out()

    weights = model.coef_[0]

    gen_index = np.argsort(weights)[-count:]
    wri_index = np.argsort(weights)[:count]

    gen_top = {}
    wri_top= {}

    print('generated')
    for i in gen_index:
        gen_top[words[i]] = weights[i]

    print('written')
    for i in wri_index:
        wri_top[words[i]] = weights[i]

    return gen_top, wri_top

def words_contribution_svm(model, vect, text):
    text_vec = vect.transform([text])
    words = vect.get_feature_names_out()
    weights = model.coef_[0]

    wrds_contr = []
    wrds_pos = text_vec.nonzero()[1]

    for i in wrds_pos:
        wrds_contr.append({
            'word': words[i],
            'tfidf': text_vec[0, i],
            'weight': weights[i],
            'contr': text_vec[0,i] * weights[i],
        })

    wrds_contr.sort(key= lambda x: x['contr'])

    return wrds_contr

