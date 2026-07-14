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
    feature_names = vect.get_feature_names_out()

    weights = model.coef_[0]

    gen_index = np.argsort(weights)[-count:]
    wri_index = np.argsort(weights)[:count]

    print('generated')
    for i in gen_index:
        print(feature_names[i], weights[i])

    print('written')
    for i in wri_index:
        print(feature_names[i], weights[i])

svm_model = joblib.load('models/svm.pkl')
tfidf_vec = joblib.load('models/svm_tfidf_vectorizer.pkl')

top_words_svm(svm_model, tfidf_vec, 10)