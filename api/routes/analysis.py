from fastapi import APIRouter, UploadFile, File, HTTPException
from core.models import get_model, sentiment_model
from src.data_preprocessing import clean_text
from src.feature_analysis import (
    extract_features, calculate_sentiment_scores, analyze_feature_sentiment
)
from api.schemas import AnalysisResult
from tempfile import NamedTemporaryFile
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import pickle
import logging
import os

router = APIRouter()
logger = logging.getLogger(__name__)

FEATURE_KEYWORDS = {
    'battery': ['battery', 'charge', 'charging', 'power'],
    'screen': ['screen', 'display', 'resolution', 'touch'],
    'camera': ['camera', 'photo', 'picture', 'selfie'],
    'performance': ['speed', 'fast', 'slow', 'performance', 'lag'],
    'price': ['price', 'cost', 'expensive', 'cheap'],
    'design': ['design', 'look', 'appearance', 'size'],
    'customer_service': ['service', 'support', 'help', 'return', 'warranty']
}


async def process_uploaded_file(file: UploadFile):
    """Handle CSV file upload and validation"""
    try:
        with NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path)
        required_columns = {'Title', 'Rating', 'Review'}

        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain these columns: {required_columns}"
            )

        return df
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=400, detail="Error processing uploaded file")
    finally:
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_reviews(file: UploadFile = File(...)):
    """Analyze uploaded reviews for sentiment and features"""
    try:
        print("🔄 Starting analysis endpoint...")

        model, tokenizer = get_model()
        print("✅ Model and tokenizer loaded.")

        # Load label encoder and create label map
        label_encoder_path = Path("models/label_encoder.pkl")
        if not label_encoder_path.exists():
            raise HTTPException(status_code=500, detail="Label encoder file missing.")

        with open(label_encoder_path, "rb") as f:
            le = pickle.load(f)
        label_map = {i: label for i, label in enumerate(le.classes_)}
        print("✅ Label encoder loaded.")

        df = await process_uploaded_file(file)
        print(f"✅ File processed. Number of rows: {len(df)}")

        df['cleaned_review'] = df['Review'].apply(clean_text)
        print("✅ Reviews cleaned.")

        sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
        padded = pad_sequences(sequences, maxlen=sentiment_model.max_length, padding='post')
        print(f"✅ Text tokenized and padded. Shape: {padded.shape}")

        predictions = model.predict(padded)
        print("✅ Predictions made.")

        predicted_classes = np.argmax(predictions, axis=1)
        df['predicted_sentiment'] = [label_map[i] for i in predicted_classes]
        print("✅ Sentiment labels assigned.")

        all_keywords = [kw for sublist in FEATURE_KEYWORDS.values() for kw in sublist]
        feature_df = extract_features(df['cleaned_review'], all_keywords)
        print("✅ Features extracted from reviews.")

        for feature, keywords in FEATURE_KEYWORDS.items():
            feature_df[feature] = feature_df[keywords].any(axis=1).astype(int)
        print("✅ Feature presence columns added.")

        feature_df = feature_df[list(FEATURE_KEYWORDS.keys())]

        sentiment_scores = calculate_sentiment_scores(df['cleaned_review'])
        print("✅ Sentiment scores calculated.")

        results_df, _ = analyze_feature_sentiment(feature_df, sentiment_scores)
        print("✅ Feature sentiment analysis complete.")

        sentiment_dist = df['predicted_sentiment'].value_counts(normalize=True).to_dict()
        print("✅ Sentiment distribution calculated.")

        print("✅ Returning response.")
        return {
            "sentiment_distribution": sentiment_dist,
            "feature_impacts": results_df.to_dict('records'),
            "sample_predictions": df.head(3)[['Review', 'predicted_sentiment']].to_dict('records'),
            "raw_data": df.to_dict('records')[:100]
        }

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail="Error during analysis")
