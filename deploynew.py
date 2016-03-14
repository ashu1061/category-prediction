from __future__ import division
from flask import Flask
from flask import request
from flask import render_template
import joblib
import sklearn
import os
from bs4 import BeautifulSoup
import re



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
    def review_to_words( raw_review ):
        review_text = BeautifulSoup(raw_review).get_text()     
        letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
        words = letters_only.lower().split()                                               
        more_meaningful_words=[]
        for words in words:
            if len(words) < 3:
                continue
            else:
                more_meaningful_words.append(words)         
        return( " ".join( more_meaningful_words ))
    def stem_words(text):
        lemma = joblib.load('./ListUp/lemma.pkl')
        stemmed_words =[lemma.lemmatize(word) for word in text.split(" ")]
        return( " ".join( stemmed_words ))
    docs_new = [stem_words(review_to_words(title + " "+ description))]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = model.predict(X_new_tfidf)
    decison_function = model.decision_function(X_new_tfidf)
    confidence = 1/(1+np.exp(-np.amax(decison_function)))
    if confidence > 0.8:
        return 'The predicted class is %s with confidence of %d' %(predicted[0], confidence*100)
    else:
        return 'Lets be honest, cant predict this as confidence is %s' %(round(confidence*100, 2))

if __name__ == '__main__':
    app.run()
