from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tkinter import *


model = models.load_model("sense_recognizer.keras")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_text_len.txt", 'r') as f:
    max_len = int(f.read())

stop_words = set(stopwords.words('english'))
punctuation_obj = str.maketrans('', '', punctuation)


def preprocess_text(text):
    word_list = word_tokenize(text)
    word_list = [word.translate(punctuation_obj) for word in word_list]
    filtered_words = [word for word in word_list if word.lower() not in stop_words]
    preprocessed_text = ' '.join(filtered_words)
    return preprocessed_text


def predict_class(news):
    preprocessed_text = preprocess_text(news)
    encoded_text = tokenizer.texts_to_sequences([preprocessed_text])[0]
    padded_text = preprocessing.sequence.pad_sequences([encoded_text], maxlen=max_len, padding='post')
    pred = model.predict(padded_text)[0][0]
    if float(pred) > 0.5:
        return "positive"
    else:
        return "negative"




def send():
    msg = InputBox.get("1.0", 'end-1c').strip()
    InputBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        res = predict_class(msg)
        ChatLog.insert(END, 'Text Emotion is: ' + res + '\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def on_key_release(event):
    ctrl = (event.state & 0x4) != 0

    if event.keycode == 88 and ctrl and event.keysym.lower() == 'x':
        event.widget.event_generate('<<Cut>>')

    if event.keycode == 86 and ctrl and event.keysym.lower() == 'v':
        event.widget.event_generate('<<Paste>>')

    if event.keycode == 67 and ctrl and event.keysym.lower() == 'c':
        event.widget.event_generate('<<Copy>>')


base = Tk()
base.title('Text Emotion Recognition')
base.geometry('400x500')
base.resizable(width=False, height=False)

ChatLog = Text(base, bd=1, bg='gray', height=100, width=200, font='Arial')
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

AssignButton = Button(base, font=('Verdana', 14, 'bold'), text='Assign',
                      width=10, height=5, bd=0, bg='#32de97', activebackground='#3c9d9b', fg='#ffffff',
                      command=send)

InputBox = Text(base, bd=0, bg='gray', width=200, height=300, font='Arial')
# InputBox.bind_all("<Key>", on_key_release, '+')
InputBox.bind('<KeyRelease>', on_key_release, '+')

scrollbar.place(x=380, y=5, height=400)
InputBox.place(x=5, y=5, height=400, width=375)

ChatLog.place(x=5, y=410, height=85, width=280)
AssignButton.place(x=290, y=410, height=85, width=105)

base.mainloop()
