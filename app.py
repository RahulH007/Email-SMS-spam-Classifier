import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

ps=PorterStemmer()

def transform_text(text):
     text=text.lower()
     text=nltk.word_tokenize(text)
     y=[]
     for i in text:
         if i.isalnum():
             y.append(i)
     text = y[:]
     y.clear()
     for i in text:
         if i not in stopwords.words('english') and i not in string.punctuation:
             y.append(i)

     text=y[:]
     y.clear()
     for i in text:
         y.append(ps.stem(i))
     return " ".join(y)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("EMAIL/SMS SPAM DETECTION")

input_sms=st.text_area("Enter the message")

if st.button("Predict"):
    t_text=transform_text(input_sms)
    vect=vectorizer.transform([t_text])
    result=model.predict(vect)[0]
    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")


