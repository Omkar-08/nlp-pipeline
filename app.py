import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading language model for the spaCy POS tagger")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

sia = SentimentIntensityAnalyzer()
custom_stopwords = set(stopwords.words('english')) - {"no", "not"}

def load_data(uploaded_files):
    dataframes = []
    for uploaded_file in uploaded_files:
        dataframes.append(pd.read_csv(uploaded_file))
    return pd.concat(dataframes, ignore_index=True)

def preprocess_data(data):
    data['state'] = data['property_name'].str.extract(r'(\w\w)$').str.upper()
    return data

def filter_data(data, state=None, review_type=None):
    if state:
        data = data[data['state'] == state]
    if review_type == 'Public':
        data['review'] = data['public_review'].fillna('')
    elif review_type == 'Private':
        data['review'] = data['private_feedback'].fillna('')
    elif review_type == 'All':
        data = data.assign(review=data.apply(lambda x: f"{x['public_review']} {x['private_feedback']}", axis=1))
    data = data.dropna(subset=['review'])
    return data

def expand_and_clean_reviews(data):
    data['segment'] = data['review'].str.split(r'\.\s+')
    data = data.explode('segment')
    data['cleaned_segment'] = data['segment'].apply(lambda text: clean_and_lemmatize(text))
    return data.dropna(subset=['cleaned_segment'])

def clean_and_lemmatize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    return ' '.join([token.lemma_ for token in nlp(text)])

def categorize_and_sentiment(data, categories):
    data['category'] = data['cleaned_segment'].apply(lambda x: categorize(x, categories))
    data['sentiment_score'] = data['cleaned_segment'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data

def categorize(text, categories):
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'Other'

# Categories and their keywords
categories = {
    "communication": ["communicate", "contact", "responsive"],
    "cleanliness": ["clean", "sanitize", "dirty"],
    "location": ["location", "area", "place"],
}

# Streamlit UI
st.title('Review Analysis Tool')

uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=['csv'])
state_filter = st.text_input("Filter by state abbreviation (optional):")
review_type = st.selectbox("Review Type", ('Public', 'Private', 'All'))

if uploaded_files:
    data = load_data(uploaded_files)
    data = preprocess_data(data)
    data = filter_data(data, state=state_filter.upper(), review_type=review_type)

    if st.button('Analyze Reviews'):
        data = expand_and_clean_reviews(data)
        data = categorize_and_sentiment(data, categories)
        st.write(data[['segment', 'cleaned_segment', 'category', 'sentiment_score']].head())

        # Visualization
        fig = px.histogram(data, x='sentiment_score', color='category', nbins=30, title="Sentiment Distribution by Category")
        st.plotly_chart(fig)

        category_counts = data['category'].value_counts()
        st.bar_chart(category_counts)
