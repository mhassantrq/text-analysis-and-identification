import read_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

def naive_bayes(data):
    train_X, test_X, train_y, test_y = train_test_split(data.text, data.label, test_size=0.2)

    vectrizer = CountVectorizer()
    train_c = vectrizer.fit_transform(train_X)
    test_c = vectrizer.transform(test_X)

    nb = MultinomialNB()
    nb.fit(train_c, train_y)

    # joblib.dump(nb, 'models/nb.pkl')
    # joblib.dump(vectrizer, 'models/nb_vectorizer.pkl')
    score = nb.score(test_c, test_y)
    print(score)


data = read_data.read_data_merged()
naive_bayes(data)

