import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def extract_features(reviews, feature_keywords):
    """Extract product features using keyword matching"""
    vectorizer = CountVectorizer(vocabulary=feature_keywords, binary=True)
    feature_matrix = vectorizer.fit_transform(reviews)
    feature_df = pd.DataFrame(feature_matrix.toarray(), columns=feature_keywords)
    return feature_df


def calculate_sentiment_scores(reviews):
    """Calculate sentiment polarity scores"""
    return np.array([TextBlob(review).sentiment.polarity for review in reviews])


def analyze_feature_sentiment(feature_df, sentiment_scores):
    """Perform linear regression analysis"""
    coef_dict = defaultdict(list)

    # Individual feature analysis
    for feature in feature_df.columns:
        X = feature_df[feature].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, sentiment_scores)

        coef_dict['Feature'].append(feature)
        coef_dict['Coefficient'].append(model.coef_[0])
        coef_dict['R-squared'].append(model.score(X, sentiment_scores))

    # Multiple regression
    full_model = LinearRegression()
    full_model.fit(feature_df, sentiment_scores)

    results_df = pd.DataFrame(coef_dict)
    return results_df.sort_values('Coefficient', ascending=False), full_model


def plot_feature_impact(results_df):
    """Visualize feature impact"""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=results_df, palette='viridis')
    plt.title('Impact of Product Features on Sentiment')
    plt.tight_layout()
    plt.show()