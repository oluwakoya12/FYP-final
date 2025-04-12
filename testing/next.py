import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# 1. Feature Extraction from Reviews
def extract_features(reviews, feature_keywords):
    """
    Extract product features mentioned in reviews using keyword matching
    """
    vectorizer = CountVectorizer(vocabulary=feature_keywords, binary=True)
    feature_matrix = vectorizer.fit_transform(reviews)
    feature_df = pd.DataFrame(feature_matrix.toarray(), columns=feature_keywords)
    return feature_df


# 2. Sentiment Scoring
def calculate_sentiment_scores(reviews):
    """
    Calculate sentiment polarity for each review
    """
    sentiments = []
    for review in reviews:
        blob = TextBlob(review)
        sentiments.append(blob.sentiment.polarity)
    return np.array(sentiments)


# 3. Feature-Sentiment Analysis
def analyze_feature_sentiment(feature_df, sentiment_scores):
    """
    Perform linear regression to analyze feature-sentiment relationships
    """
    # Initialize results storage
    coef_dict = defaultdict(list)

    # Individual feature analysis
    print("Individual Feature Impact:")
    for feature in feature_df.columns:
        X = feature_df[feature].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, sentiment_scores)

        coef_dict['Feature'].append(feature)
        coef_dict['Coefficient'].append(model.coef_[0])
        coef_dict['R-squared'].append(model.score(X, sentiment_scores))

        print(f"{feature}: Coefficient = {model.coef_[0]:.4f}, R² = {model.score(X, sentiment_scores):.4f}")

    # Multiple regression with all features
    print("\nMultiple Regression Analysis:")
    full_model = LinearRegression()
    full_model.fit(feature_df, sentiment_scores)

    # Create results dataframe
    results_df = pd.DataFrame(coef_dict)
    results_df = results_df.sort_values('Coefficient', ascending=False)

    return results_df, full_model


# 4. Visualization
def plot_feature_impact(results_df):
    """
    Visualize feature impact on sentiment
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=results_df, palette='viridis')
    plt.title('Impact of Product Features on Sentiment')
    plt.xlabel('Regression Coefficient')
    plt.ylabel('Product Feature')
    plt.tight_layout()
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Load your data (replace with your actual data loading)
    df = pd.read_csv("../new_reviews.csv")

    # Clean reviews (basic cleaning)
    df['cleaned_review'] = df['Review'].str.lower()
    df['cleaned_review'] = df['cleaned_review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Define product features to analyze (customize based on your product)
    feature_keywords = {
        'battery': ['battery', 'charge', 'charging', 'power'],
        'screen': ['screen', 'display', 'resolution', 'touch'],
        'camera': ['camera', 'photo', 'picture', 'selfie'],
        'performance': ['speed', 'fast', 'slow', 'performance', 'lag'],
        'price': ['price', 'cost', 'expensive', 'cheap'],
        'design': ['design', 'look', 'appearance', 'size'],
        'customer_service': ['service', 'support', 'help', 'return', 'warranty']
    }

    # Flatten the keyword dictionary for CountVectorizer
    all_keywords = []
    for feature, keywords in feature_keywords.items():
        all_keywords.extend(keywords)

    # 1. Extract features
    print("Extracting features from reviews...")
    feature_df = extract_features(df['cleaned_review'], all_keywords)

    # Map back to original feature categories
    for feature in feature_keywords:
        keywords = feature_keywords[feature]
        feature_df[feature] = feature_df[keywords].any(axis=1).astype(int)

    # Keep only the main features (not individual keywords)
    feature_df = feature_df[list(feature_keywords.keys())]

    # 2. Calculate sentiment scores
    print("\nCalculating sentiment scores...")
    sentiment_scores = calculate_sentiment_scores(df['cleaned_review'])

    # 3. Analyze feature-sentiment relationship
    print("\nAnalyzing feature-sentiment relationships...")
    results_df, model = analyze_feature_sentiment(feature_df, sentiment_scores)

    # 4. Visualize results
    print("\nVisualizing results...")
    plot_feature_impact(results_df)

    # Show strongest positive and negative influencers
    print("\nKey Findings:")
    print("Most positive impact features:")
    print(results_df.head(3))

    print("\nMost negative impact features:")
    print(results_df.tail(3))