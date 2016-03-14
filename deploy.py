from flask import Flask
from flask import request
from flask import render_template
from sklearn.externals import joblib
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("myform.html")

@app.route('/', methods=['POST'])
def my_form_post():

    title = request.form['title']
    description = request.form['description']
    model = joblib.load('./ListUp/ListupNLP_v2.pkl')
    count_vect = joblib.load('./ListUp/vect.pkl')
    tfidf_transformer = joblib.load('./ListUp/tfidf.pkl')
    docs_new = [title + " "+ description]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = model.predict(X_new_tfidf)
    decison_function = model.decision_function(X_new_tfidf)
    confidence = 1/(1+np.exp(-np.amax(decison_function)))
    if confidence > 0.8:
        return predicted
    else:
        return 'Lets be honest, cant predict this' 

if __name__ == '__main__':
    app.run()


 