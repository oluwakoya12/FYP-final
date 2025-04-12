import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(history):
    """Plot training history metrics"""
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


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    results = model.evaluate(X_test, y_test)
    metrics = {
        'Test Loss': results[0],
        'Test Accuracy': results[1],
        'Precision': results[2],
        'Recall': results[3]
    }
    return metrics


def predict_sentiment(model, tokenizer, texts, max_length):
    """Predict sentiment for new texts"""
    cleaned_texts = [clean_text(text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded)
    return predictions