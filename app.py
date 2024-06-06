import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go

# Initialize NLP tools
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Custom stopwords setup, removing typical negations that are significant in sentiment analysis
custom_stopwords = set(stopwords.words('english')) - {"no", "not"}

# Contraction mapping for text processing
contraction_mapping = {
    "didn't": "did not", "don't": "do not", "aren't": "are not", "couldn't": "could not",
}

# Function definitions
def preprocess_data(data):
    """
    Preprocess the CSV file data by handling state extraction, contraction expansion, stopwords removal, and lemmatization.
    """
    data['state'] = data['property_name'].str.extract(r'(\w\w)$').str.upper().fillna('N/A')
    data = expand_and_clean_reviews(data)
    return data

def expand_and_clean_reviews(data):
    """
    Expands reviews into sentences, cleans, and processes text for sentiment analysis.
    """
    data['review'] = data.apply(lambda row: f"{row['public_review']} {row['private_feedback']}".strip(), axis=1)
    data = data.dropna(subset=['review'])
    data = data.drop_duplicates(subset=['review'])

    # Expand reviews into sentences
    data['sentences'] = data['review'].str.split(r'\.\s+')
    data = data.explode('sentences').dropna(subset=['sentences'])
    data['sentences'] = data['sentences'].apply(lambda x: clean_text(x))
    return data

def clean_text(text):
    """
    Cleans text by expanding contractions, removing punctuation, and applying lemmatization.
    """
    # Expand contractions
    for contraction, expanded in contraction_mapping.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text).lower()
    # Tokenization and removal of custom stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]
    # Lemmatization using spaCy
    lemmatized_text = ' '.join([token.lemma_ for token in nlp(' '.join(filtered_tokens))])
    return lemmatized_text

def analyze_sentiment(data):
    """
    Analyzes sentiment of processed text.
    """
    sia = SentimentIntensityAnalyzer()
    data['sentiment_score'] = data['sentences'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data

def plot_sentiment_trends(data):
    """
    Plots sentiment trends over time.
    """
    fig = go.Figure()
    for state in data['state'].unique():
        state_data = data[data['state'] == state]
        state_data = state_data.groupby('checkout_date').mean().reset_index()
        fig.add_trace(go.Scatter(x=state_data['checkout_date'], y=state_data['sentiment_score'],
                                 mode='lines+markers', name=state))
    fig.update_layout(title='Sentiment Analysis by State Over Time',
                      xaxis_title='Date',
                      yaxis_title='Average Sentiment Score',
                      legend_title='State')
    return fig

# Streamlit UI
st.title('Review Data Sentiment Analysis')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)
    data = analyze_sentiment(data)
    st.write('Processed Data', data.head())

    # Plotting
    sentiment_plot = plot_sentiment_trends(data)
    st.plotly_chart(sentiment_plot, use_container_width=True)
