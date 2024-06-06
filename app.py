import streamlit as st
import pandas as pd
import os
from processing_code import master_nlp, plot_sentiment_trend_by_category  # assuming your provided code is saved as processing_code.py

# Initialization
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

st.title("Review Processing App")

# File uploader
uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=['csv'])
data_frames = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        data_frames.append(df)
    
    if st.button("Process Files"):
        # Combine all dataframes if multiple files
        if len(data_frames) > 1:
            data = pd.concat(data_frames, ignore_index=True)
        else:
            data = data_frames[0]

        # Processing data
        processed_data = master_nlp(data)
        st.write("Processed Data", processed_data.head())

        # Plotting
        plots = plot_sentiment_trend_by_category(processed_data)
        for category, plot_html in plots.items():
            st.write(f"{category.capitalize()} Sentiment Trend")
            st.components.v1.html(plot_html, height=600)

else:
    st.warning("Please upload at least one CSV file to proceed.")

