import read_data, preprocessing
from collections import defaultdict
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

def from_scratch(data):
    train_data_generated, train_data_written, test_data_generated, test_data_written = preprocessing.split_train_test(data)

    train_data_generated = preprocessing.lower_case(train_data_generated)
    train_data_written = preprocessing.lower_case(train_data_written)
    test_data_generated = preprocessing.lower_case(test_data_generated)
    test_data_written = preprocessing.lower_case(test_data_written)

    train_data_generated = preprocessing.split_features(train_data_generated)
    train_data_written = preprocessing.split_features(train_data_written)
    test_data_generated = preprocessing.split_features(test_data_generated)
    test_data_written = preprocessing.split_features(test_data_written)

    vocab = defaultdict(int)
    train_gen_bow = defaultdict(int)
    train_wri_bow = defaultdict(int)
    cond_gen = defaultdict(int)
    cond_wri = defaultdict(int)
    tgc=twc=tp=fp=tn=fn=0

    prior_gen = len(train_data_generated) / (len(train_data_generated) + len(train_data_written))
    prior_wri = len(train_data_written) / (len(train_data_generated) + len(train_data_written))

    for doc in train_data_generated:
        for w in doc:
            vocab[w] +=1
            train_gen_bow[w] +=1
            tgc+=1

    for doc in train_data_written:
        for w in doc:
            vocab[w] +=1
            train_wri_bow[w] +=1
            twc+=1

    for w in vocab:
        cond_gen[w] = ((train_gen_bow[w] + 1) / (tgc + len(vocab)))
        cond_wri[w] = ((train_wri_bow[w] + 1) / (twc + len(vocab)))

    for doc in test_data_generated:
        prob_gen = math.log(prior_gen)
        prob_wri = math.log(prior_wri)
        for w in doc:
            if w in cond_gen:
                prob_gen += math.log(cond_gen[w])
            if w in cond_wri:
                prob_wri += math.log(cond_wri[w])
        if prob_gen>=prob_wri:
            tp+=1
        else:
            fn+=1

    for doc in test_data_written:
        prob_gen = math.log(prior_gen)
        prob_wri = math.log(prior_wri)
        for w in doc:
            if w in cond_gen:
                prob_gen += math.log(cond_gen[w])
            if w in cond_wri:
                prob_wri += math.log(cond_wri[w])
        if prob_wri>=prob_gen:
            tn+=1
        else:
            fp+=1

    print(tp, fp, fn, tn)

    acc = (tp+tn)/(tp+fp+fn+tn)
    print(acc)

def from_sklearn(data):
    train_X, test_X, train_y, test_y = train_test_split(data.text, data.label, test_size=0.1)

    vectrizer = CountVectorizer(lowercase=True)
    train_c = vectrizer.fit_transform(train_X)
    test_c = vectrizer.transform(test_X)

    nb = MultinomialNB()
    nb.fit(train_c, train_y)

    # joblib.dump(nb, 'models/nb.pkl')
    # joblib.dump(vectrizer, 'models/nb_vectorizer.pkl')
    score = nb.score(test_c, test_y)
    print(score)


data = read_data.read_csv()
#from_scratch(data)
#from_sklearn(data)

