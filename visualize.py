import read_data
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt


def stopwords_pie(data):
    others=stopw=0
    for doc in data:
        for w in doc.split():
            if w in ENGLISH_STOP_WORDS:
                stopw+=1
            else:
                others+=1
    labels = ['Stop Words', 'Other Words']
    wrds=[stopw,others]
    plt.pie(wrds, labels=labels)
    plt.show()





df = read_data.read_csv()
stopwords_pie(df['text'])