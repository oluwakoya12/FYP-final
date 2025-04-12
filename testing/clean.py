import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load CSV
df = pd.read_csv('new_reviews.csv')  # adjust path as needed

# Clean review text
def clean_text(text):
    text = str(text).lower()                       # Lowercase
    text = re.sub(r'\d+', '', text)                # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)            # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()       # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['Cleaned_Review'] = df['Review'].apply(clean_text)

# Map ratings to sentiment
def get_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating <= 2:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Rating'].apply(get_sentiment)

# Show sentiment distribution
sentiment_counts = df['Sentiment'].value_counts()
print("Sentiment Distribution:\n", sentiment_counts)

# Plot
sentiment_counts.plot(kind='bar', title='Sentiment Distribution', color=['red', 'gray', 'green'])
plt.ylabel('Number of Reviews')
plt.show()
