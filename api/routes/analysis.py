from fastapi import APIRouter, UploadFile, File, HTTPException
from core.models import SentimentModel
from core.models import get_model
from src.data_preprocessing import clean_text
from src.feature_analysis import (
    extract_features, calculate_sentiment_scores,
    analyze_feature_sentiment
)
from api.schemas import AnalysisResult
from tempfile import NamedTemporaryFile
import pandas as pd
import numpy as np
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences

router = APIRouter()
logger = logging.getLogger(__name__)

# Your existing feature keywords
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
    """Your existing file processing logic"""
    try:
        # Create a temporary file
        with NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Read the CSV
        df = pd.read_csv(tmp_path)
        required_columns = {'Title', 'Rating', 'Review', 'Customer Name', 'Date', 'Customer Location'}

        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain these columns: {required_columns}"
            )

        return df
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing uploaded file")
    finally:
        # Clean up temporary file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass

@router.post("/analyze")
async def analyze_reviews(file: UploadFile = File(...)):
    model = get_model()  # Get the singleton instance
    # Use model.model and model.tokenizer in your processing

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_reviews(file: UploadFile = File(...)):
    """Endpoint that uses your existing analysis logic"""
    try:
        # Load models
        model, tokenizer = SentimentModel.get_model()

        # Process file
        df = await process_uploaded_file(file)

        # Your existing analyze_user_data logic
        df['cleaned_review'] = df['Review'].apply(clean_text)
        sequences = tokenizer.texts_to_sequences(df['cleaned_review'])
        padded = pad_sequences(sequences, maxlen=SentimentModel.max_length, padding='post')
        predictions = model.predict(padded)
        df['predicted_sentiment'] = np.where(predictions > 0.5, 'Positive', 'Negative')

        # Feature analysis
        all_keywords = [kw for sublist in FEATURE_KEYWORDS.values() for kw in sublist]
        feature_df = extract_features(df['cleaned_review'], all_keywords)

        for feature in FEATURE_KEYWORDS:
            keywords = FEATURE_KEYWORDS[feature]
            feature_df[feature] = feature_df[keywords].any(axis=1).astype(int)

        feature_df = feature_df[list(FEATURE_KEYWORDS.keys())]
        sentiment_scores = calculate_sentiment_scores(df['cleaned_review'])
        results_df, _ = analyze_feature_sentiment(feature_df, sentiment_scores)

        # Prepare response
        sentiment_dist = df['predicted_sentiment'].value_counts(normalize=True).to_dict()

        return {
            "sentiment_distribution": sentiment_dist,
            "feature_impacts": results_df.to_dict('records'),
            "sample_predictions": df.head(3)[['Review', 'predicted_sentiment']].to_dict('records'),
            "raw_data": df.to_dict('records')[:100]  # Limit to first 100 records
        }

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))