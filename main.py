import string
import nltk
import streamlit as st
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# UI
st.title("Email/SMS Spam Classifier")

sms = st.text_area("Enter your message")

if st.button('Predict'):

    # 1. preprocess
    transformed_text = transform_text(sms)

    # 2. vectorize
    vector_input = vectorizer.transform([transformed_text])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")


