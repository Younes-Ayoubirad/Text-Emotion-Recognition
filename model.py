import pickle
print("import packages")
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
import random

# # Set environment variables for TensorFlow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow.keras as tk

print("\tdone")

# Preload stopwords and punctuation objects
stop_words = set(stopwords.words('english'))
punctuation_obj = str.maketrans('', '', punctuation)


# Function to preprocess text files
def preprocess_text_files(folder_path):
    all_text_list = []
    max_len = 0

    path = 'txt_sentoken/' + folder_path
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            word_list = (
                word.translate(punctuation_obj)  # Remove punctuation
                for line in f
                for word in word_tokenize(line)  # Tokenize line-by-line
            )
            # Filter stopwords and join cleaned text
            filtered_words = [word for word in word_list if word.lower() not in stop_words]
            all_text_list.append(' '.join(filtered_words))
            max_len = max(max_len, len(filtered_words))

    return all_text_list, max_len


# Process negative and positive datasets
print("process text files")
neg_path = "neg/"
pos_path = "pos/"
negative_word_list, neg_max_len = preprocess_text_files(neg_path)
positive_word_list, pos_max_len = preprocess_text_files(pos_path)
print("\tdone")

# Determine the maximum text length
max_text_len = max(neg_max_len, pos_max_len)
with open("max_text_len.txt", 'w', encoding='utf-8') as f:
    f.write(str(max_text_len))


# shuffle data
random.shuffle(negative_word_list)
random.shuffle(positive_word_list)

x_train = negative_word_list[:800] + positive_word_list[:800]
y_train = [0] * 800 + [1] * 800

x_test = negative_word_list[800:] + positive_word_list[800:]
y_test = [0] * 200 + [1] * 200

y_train = np.array(y_train)
y_test = np.array(y_test)

# Tokenize and pad sequences
print("preprocess data")
tokenizer = tk.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(x_train)

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

x_train_encoded = tokenizer.texts_to_sequences(x_train)
x_test_encoded = tokenizer.texts_to_sequences(x_test)

x_train_padded = tk.preprocessing.sequence.pad_sequences(x_train_encoded, maxlen=max_text_len, padding='post')
x_test_padded = tk.preprocessing.sequence.pad_sequences(x_test_encoded, maxlen=max_text_len, padding='post')
print("\tdone")

# from tensorflow.keras import models
# model = models.load_model('sense_recognizer.keras')
# print(model.predict(x_train_padded[1]))


# Model architecture
model = tk.models.Sequential([
    tk.layers.Input(shape=(max_text_len,)),
    tk.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128),
    tk.layers.Bidirectional(tk.layers.LSTM(64, return_sequences=True)),
    tk.layers.Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu'),
    tk.layers.MaxPooling1D(pool_size=2),
    tk.layers.Flatten(),
    tk.layers.Dropout(0.5),
    tk.layers.Dense(64, activation='relu'),
    tk.layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(
    optimizer=tk.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train_padded, y_train, validation_data=(x_test_padded, y_test), epochs=10, batch_size=32)
model.save('sense_recognizer.keras')
