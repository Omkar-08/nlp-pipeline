import streamlit as st
import pandas as pd
import os
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import plotly.graph_objects as go
import tarfile
import urllib.request


MODEL_URL = "https://github.com/Abdullah-Kazi/nlp-pipeline/raw/1d34f53ef652286efa7e563a47c30825db20b745/en_core_web_sm-3.7.1.tar.gz"
MODEL_PATH = "en_core_web_sm-3.7.1.tar.gz"
EXTRACT_PATH = "./models"

@st.cache(allow_output_mutation=True, show_spinner=True)
def download_and_setup_spacy_model():
    # Ensure the download directory exists
    if not os.path.exists(EXTRACT_PATH):
        os.makedirs(EXTRACT_PATH)

    # Download the model tar file if not present
    if not os.path.exists(os.path.join(EXTRACT_PATH, MODEL_PATH)):
        urllib.request.urlretrieve(MODEL_URL, os.path.join(EXTRACT_PATH, MODEL_PATH))

    # Define the model directory inside the extract path
    model_directory = os.path.join(EXTRACT_PATH, "en_core_web_sm-3.7.1")

    # Extract the model if the directory doesn't exist
    if not os.path.exists(model_directory):
        with tarfile.open(os.path.join(EXTRACT_PATH, MODEL_PATH), "r:gz") as tar:
            tar.extractall(path=EXTRACT_PATH)

    # Check if the config file exists
    if not os.path.exists(os.path.join(model_directory, "config.cfg")):
        st.error("Model configuration file not found after extraction.")
        st.stop()

    # Load the model from the extracted path
    try:
        nlp_model = spacy.load(model_directory)
    except Exception as e:
        st.error(f"Failed to load spaCy model from {model_directory}: {e}")
        st.stop()

    return nlp_model

# Load the model
nlp = download_and_setup_spacy_model()


def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
download_nltk_resources()

def clean_text(text):
    contraction_mapping = {
        "didn't": "did not", "don't": "do not", "aren't": "are not", "couldn't": "could not",
    }
    text = text.lower()
    for contraction, expanded in contraction_mapping.items():
        text = re.sub(r'\b' + contraction + r'\b', expanded, text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized = [token.lemma_ for token in nlp(' '.join(tokens)) if token.text not in custom_stopwords]
    return ' '.join(lemmatized)

def process_reviews(data):
    data['state'] = data['property_name'].str.extract(r'(\w\w)$').str.upper().fillna('N/A')
    data['full_review'] = data['public_review'].fillna('') + ' ' + data['private_feedback'].fillna('')
    data['cleaned_text'] = data['full_review'].apply(clean_text)
    sia = SentimentIntensityAnalyzer()
    data['sentiment'] = data['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return data

def plot_sentiment(data):
    fig = go.Figure()
    for state in data['state'].unique():
        state_data = data[data['state'] == state]
        fig.add_trace(go.Scatter(
            x=state_data['checkout_date'], y=state_data['sentiment'],
            mode='lines+markers', name=state
        ))
    fig.update_layout(
        title='Sentiment Analysis by State',
        xaxis_title='Checkout Date',
        yaxis_title='Sentiment Score',
        legend_title='State'
    )
    return fig

st.title('Review Sentiment Analysis App')
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_columns = ['checkout_date', 'property_name', 'public_review', 'private_feedback']
    if all(column in data.columns for column in required_columns):
        processed_data = process_reviews(data)
        st.write("Processed Data", processed_data.head())
        sentiment_plot = plot_sentiment(processed_data)
        st.plotly_chart(sentiment_plot)
    else:
        st.error("Uploaded file is missing one or more required columns: 'checkout_date', 'property_name', 'public_review', 'private_feedback'.")


