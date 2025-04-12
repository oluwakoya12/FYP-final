import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.data_preprocessing import load_and_clean_data, preprocess_data, clean_text
from src.model_training import build_bigru_attention_model, train_model, AttentionLayer
from src.feature_analysis import (
    extract_features, calculate_sentiment_scores,
    analyze_feature_sentiment, plot_feature_impact
)
from src.utils import plot_training_history, evaluate_model, predict_sentiment
import pickle
import os


def train_and_save_model():
    # Load and preprocess data
    df = load_and_clean_data("data/raw/new_reviews.csv")
    X_train, X_test, y_train, y_test, tokenizer = preprocess_data(df)

    # Build and train model
    vocab_size = len(tokenizer.word_index) + 1
    max_length = X_train.shape[1]
    model = build_bigru_attention_model(vocab_size, 128, max_length)
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate and save
    metrics = evaluate_model(model, X_test, y_test)
    print("\nModel Evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    plot_training_history(history)

    # Save model and tokenizer
    os.makedirs("models", exist_ok=True)
    model.save("models/bigru_model")  # SavedModel format (recommended)
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    return model, tokenizer


def analyze_user_data(user_csv_path, tokenizer, model, max_length=200):
    """Analyze new user-provided data"""
    user_df = pd.read_csv(user_csv_path)
    user_df['cleaned_review'] = user_df['Review'].apply(clean_text)

    # Predict sentiment
    sequences = tokenizer.texts_to_sequences(user_df['cleaned_review'])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded)
    user_df['predicted_sentiment'] = np.where(predictions > 0.5, 'Positive', 'Negative')

    # Feature analysis
    feature_keywords = {
        'battery': ['battery', 'charge', 'charging', 'power'],
        'screen': ['screen', 'display', 'resolution', 'touch'],
        'camera': ['camera', 'photo', 'picture', 'selfie'],
        'performance': ['speed', 'fast', 'slow', 'performance', 'lag'],
        'price': ['price', 'cost', 'expensive', 'cheap'],
        'design': ['design', 'look', 'appearance', 'size'],
        'customer_service': ['service', 'support', 'help', 'return', 'warranty']
    }

    all_keywords = [kw for sublist in feature_keywords.values() for kw in sublist]
    feature_df = extract_features(user_df['cleaned_review'], all_keywords)

    for feature in feature_keywords:
        keywords = feature_keywords[feature]
        feature_df[feature] = feature_df[keywords].any(axis=1).astype(int)

    feature_df = feature_df[list(feature_keywords.keys())]
    sentiment_scores = calculate_sentiment_scores(user_df['cleaned_review'])
    results_df, _ = analyze_feature_sentiment(feature_df, sentiment_scores)

    print("\n=== Sentiment Summary ===")
    print(user_df['predicted_sentiment'].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

    print("\n=== Key Feature Insights ===")
    print("Most positive impact features:")
    print(results_df.head(3))
    print("\nMost negative impact features:")
    print(results_df.tail(3))

    plot_feature_impact(results_df)
    return user_df, results_df


if __name__ == "__main__":
    # Train model or load existing
    if not os.path.exists("models/bigru_model"):
        print("Training new model...")
        model, tokenizer = train_and_save_model()
    else:
        print("Loading existing model...")
        model = tf.keras.models.load_model("models/bigru_model")  # SavedModel loading (no custom_objects needed)
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    # Analyze new user data
    user_csv = input("Enter path to user CSV file: ")
    analyze_user_data(user_csv, tokenizer, model)
