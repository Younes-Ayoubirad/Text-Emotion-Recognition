import pickle
import streamlit as st
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras import preprocessing
from tensorflow.keras import models

model = models.load_model("sense_recognizer.keras")

with open("tokenizer.h5", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_text_len.txt", 'r') as f:
    max_len = int(f.read())


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    punctuation_obj = str.maketrans('', '', punctuation)
    all_text_list = []
    word_list = (
        word.translate(punctuation_obj)  # Remove punctuation
        for line in text
        for word in word_tokenize(line)  # Tokenize line-by-line
    )
    # Filter stopwords and join cleaned text
    filtered_words = [word for word in word_list if word.lower() not in stop_words]
    all_text_list.append(' '.join(filtered_words))
    return all_text_list


def predict_class(news):
    processed_text = preprocess_text(news)
    encoded_text = tokenizer.texts_to_sequences(processed_text)
    padded_text = preprocessing.sequence.pad_sequences(encoded_text, maxlen=max_len, padding='post')
    pred = model.predict(padded_text)
    if pred == 0:
        return "negative"
    else:
        return "positive"


st.title('Text Emotion Recognition with AI')

text = st.text_area('Enter your text: ')

btn = st.button('Detect Emotion')


if btn:
    out_text = predict_class(text)
    st.success("Text Emotion is :  " + out_text)
