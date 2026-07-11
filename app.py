from flask import Flask, render_template, request
import joblib


app = Flask(__name__, template_folder='templates')

nb_model = joblib.load('models/nb.pkl')
count_vec = joblib.load('models/count_vectorizer.pkl')
svm_model = joblib.load('models/svm.pkl')
tfidf_vec = joblib.load('models/svm_tfidf_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    file = request.files.get('file')
    
    if request.method == 'POST':
        if file and file.name.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            text = request.form['text']
        #vec = count_vec.transform([text])
        vec = tfidf_vec.transform([text])
#        prb = nb_model.predict_proba(vec)[0]
        pred = svm_model.predict(vec)[0]
        predictions = {
            'pred': pred
        }

        # predictions = {
        #     'pred' : pred,
        #     'written' : prb[0],
        #     'generated' : prb[1],
        # }

    return render_template('index.html', predictions=predictions)

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/graphs/')
def graphs():
    return render_template('graphs.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


