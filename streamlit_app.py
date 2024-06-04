import streamlit as st
import pandas as pd
import functions
from io import StringIO

# Display a title for the app
st.title("Data Review Application")

# File uploader allows user to add file
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
if uploaded_file is not None:
    # Read the file into a Pandas DataFrame
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    data = pd.read_csv(stringio)
    
    # Process the single DataFrame
    data = functions.process_single_csv_file(data)
    
    # Create the 'state' column
    data = functions.state(data)
    
    # User selects geography, review type, and display type
    geography = st.selectbox("Select Geography", ["national", "state"])
    review_type = st.selectbox("Select Review Type", ["Public", "Private", "All"])
    show_me = st.selectbox("What do you want to see?", ["nlp", "sentiment", "both"])
    
    selected_state = None
    if geography == "state":
        selected_state = st.selectbox("Select State", functions.states)
        if selected_state:
            data = functions.filter_by_state(data, selected_state)
    
    # Process files based on review type
    data = functions.process_files(data, review_type)
    
    # Master NLP function
    data = functions.master_nlp(data)
    
    # Show the DataFrame and plots based on user selection
    st.write("Processed Data", data.to_html(escape=False), unsafe_allow_html=True)
    
    if show_me == 'nlp' or show_me == 'both':
        plots_nlp = functions.plot_trend(data)
        for category, plot_html in plots_nlp.items():
            st.components.v1.html(plot_html, height=600)
    
    if show_me == 'sentiment' or show_me == 'both':
        plots_sentiment = functions.plot_sentiment_trend_by_category(data)
        for category, plot_html in plots_sentiment.items():
            st.components.v1.html(plot_html, height=600)
