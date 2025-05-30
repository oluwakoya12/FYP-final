# core/models.py

import os
import tensorflow as tf
import pickle
from pathlib import Path
from src.model_training import AttentionLayer


class SentimentModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentModel, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        """Load model and tokenizer from disk"""
        model_path = Path("models/bigru_model.keras")
        tokenizer_path = Path("models/tokenizer.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        print("Loading pre-trained model...")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.max_length = 200
        print("Model and tokenizer loaded successfully.")


# Global instance
sentiment_model = SentimentModel()


def get_model():
    """Return the loaded model and tokenizer"""
    return sentiment_model.model, sentiment_model.tokenizer
