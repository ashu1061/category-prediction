from flask import Flask
from flask import request
from flask import render_template
from sklearn.externals import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("myform.html")

@app.route('/', methods=['POST'])
def my_form_post():

    title = request.form['title']
    description = request.form['description']
    model = joblib.load('./ListUp/SVM_GridSearch.pkl')
    products = pd.read_pickle('./ListUp/products_pandas')
    count_vect = CountVectorizer(ngram_range=(1,2))
    X_train_counts = count_vect.fit_transform(products.description)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    docs_new = [title + " "+ description]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = model.predict(docs_new)
    return predicted[0] 

if __name__ == '__main__':
    app.run()


 