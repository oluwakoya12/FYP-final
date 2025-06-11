import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def load_and_clean_data(filepath):
    """Load and clean the raw data for 3-class sentiment"""
    df = pd.read_csv(filepath)
    df['sentiment'] = df['Rating'].apply(lambda x: 2 if x == 3 else (1 if x >= 4 else 0))
    df['cleaned_review'] = df['Review'].apply(clean_text)
    return df



def preprocess_data(df, test_size=0.2, max_words=20000, max_len=200):
    """Tokenize and split data into train/test sets"""
    X = df['cleaned_review'].values
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer