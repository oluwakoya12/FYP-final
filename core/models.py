import tensorflow as tf
import pickle
from pathlib import Path
from src.model_training import AttentionLayer
from src.data_preprocessing import load_and_clean_data, preprocess_data
from src.model_training import build_bigru_attention_model, train_model
from src.utils import evaluate_model
import os


class SentimentModel:
    model = None
    tokenizer = None
    max_length = 200

    @classmethod
    def get_model(cls):
        """Get or initialize model"""
        if cls.model is None:
            cls.load_models()
        return cls.model, cls.tokenizer

    @classmethod
    def load_models(cls):
        """Load ML models from disk or train new ones"""
        model_path = Path("models/bigru_model")
        tokenizer_path = Path("models/tokenizer.pkl")

        if model_path.exists() and tokenizer_path.exists():
            # Load existing models
            # Change this line in load_models():
            cls.model = tf.keras.models.load_model(
                "models/bigru_model.keras",  # Match the extension you used
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            with open(tokenizer_path, "rb") as f:
                cls.tokenizer = pickle.load(f)
        else:
            # Train new models if they don't exist
            cls.train_new_model()

    @classmethod
    def train_new_model(cls):
        """Train and save new model"""
        print("Training new model...")
        df = load_and_clean_data("data/raw/new_reviews.csv")
        X_train, X_test, y_train, y_test, tokenizer = preprocess_data(df)

        vocab_size = len(tokenizer.word_index) + 1
        max_length = X_train.shape[1]
        model = build_bigru_attention_model(vocab_size, 128, max_length)
        model, history = train_model(model, X_train, y_train, X_test, y_test)

        metrics = evaluate_model(model, X_test, y_test)
        print("\nModel Evaluation:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        os.makedirs("models", exist_ok=True)
        model.save("models/bigru_model.keras")  # or .h5 instead of just directory
        with open("models/tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

        cls.model = model
        cls.tokenizer = tokenizer
        cls.max_length = max_length


def load_models():
    """Initialize models (for API startup)"""
    SentimentModel.load_models()