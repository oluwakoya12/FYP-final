import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Attention Layer
class AttentionLayer(Layer):
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.attention = Dense(1, activation='tanh')

    def call(self, inputs):
        attention_scores = self.attention(inputs)
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = tf.expand_dims(attention_weights, axis=-1) * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# Model Builder

def build_bigru_attention_model(vocab_size, embedding_dim, max_length, gru_units=64, dense_units=64, dropout_rate=0.3, l2_lambda=0.001):
    inputs = Input(shape=(max_length,))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, mask_zero=True)(inputs)
    gru_out = Bidirectional(GRU(units=gru_units, return_sequences=True, kernel_regularizer=l2(l2_lambda)))(embedding)
    gru_out = Dropout(dropout_rate)(gru_out)
    context_vector, attention_weights = AttentionLayer(units=gru_units*2)(gru_out)
    dense = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_lambda))(context_vector)
    dense = Dropout(dropout_rate)(dense)
    outputs = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# Load and clean data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan))
    df = df.dropna(subset=['sentiment'])
    df['cleaned_review'] = df['Review'].apply(clean_text)
    return df

# Preprocess data
def preprocess_data(df, test_size=0.2, max_words=20000, max_len=200):
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

# Train and evaluate

def train_and_evaluate(X_train, X_test, y_train, y_test, vocab_size, max_length):
    model = build_bigru_attention_model(vocab_size, 128, max_length)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'), ReduceLROnPlateau(factor=0.1, patience=2, monitor='val_loss')]
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=20, callbacks=callbacks)
    results = model.evaluate(X_test, y_test)
    print(f"Test Loss: {results[0]:.4f}\nTest Accuracy: {results[1]:.4f}\nPrecision: {results[2]:.4f}\nRecall: {results[3]:.4f}")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.tight_layout()
    plt.show()
    return model, history

# Main
if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_and_clean_data("new_reviews.csv")
    print(df[['Review', 'cleaned_review', 'Rating', 'sentiment']].head())
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, tokenizer = preprocess_data(df)
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    print("\nTraining model...")
    vocab_size = len(tokenizer.word_index) + 1
    max_length = X_train.shape[1]
    model, history = train_and_evaluate(X_train, X_test, y_train, y_test, vocab_size, max_length)

    # Predictions
    print("\nSample predictions...")
    sample_texts = ["This product is amazing!", "Horrible experience.", "It was just okay."]
    sample_cleaned = [clean_text(text) for text in sample_texts]
    sample_seq = tokenizer.texts_to_sequences(sample_cleaned)
    sample_pad = pad_sequences(sample_seq, maxlen=max_length, padding='post')
    predictions = model.predict(sample_pad)
    for text, pred in zip(sample_texts, predictions):
        label = "Positive" if pred > 0.5 else "Negative"
        print(f"Review: {text}\nPredicted Sentiment: {label} ({pred[0]:.4f})\n")