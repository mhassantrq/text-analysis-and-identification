from flask import Flask, render_template, request
import joblib


app = Flask(__name__, template_folder='templates')

nb_model = joblib.load('models/nb.pkl')
count_vec = joblib.load('models/nb_vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None

    if request.method == 'POST':

        text = request.form['text']
        vec = count_vec.transform([text])
        prb = nb_model.predict_proba(vec)[0]
        pred = nb_model.predict(vec)[0]

        predictions = {
            'pred' : pred,
            'written' : prb[0],
            'generated' : prb[1],
        }

    return render_template('index.html', predictions=predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


