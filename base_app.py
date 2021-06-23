"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from io import RawIOBase
from google.protobuf import message
import streamlit as st
import joblib,os
import base64

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Our dependencies
import nltk
import seaborn as sns
import re
from string import punctuation
import emoji
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# en_core_web_sm dependancy
from nltk import pos_tag
from nlppreprocess import NLP
nlp = NLP()

# Picture
image = "Images/pic.jpg"

# Cleaning data
def cleaner(col):

    # Emoji to Text
    col = emoji.demojize(col, delimiters=("", "")) 

    # Lower case-ing
    col = col.lower()

    # Replace urls
    url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    web = 'url'
    col = re.sub(url, web, col)
    
    # Removing Mention
    mentions = r'@[\w]*'
    rep = ''
    #col = re.sub(mentions, rep, col)

    # Removing Hasgtags
    hashtags = r'#[\w]*'
    rep = ''
    #col = re.sub(hashtags, rep, col)

    # Remove Puncuation
    col = ''.join(i for i in col if i not in punctuation)
    
    # Removing non-alphabets
    col = re.sub('[^a-z]', ' ',col)

    # Tokenisation & Lemmatisation
    lemmatizer = WordNetLemmatizer()
    col = [lemmatizer.lemmatize(token) for token in col.split(' ')]
    col = [lemmatizer.lemmatize(token, 'v') for token in col]
    col = ' '.join(col)
   
    return col

# Vectorizer
news_vectorizer = open("resources/lsvc_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer)

#@st.cache
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Climate Change Analysis")
    st.subheader("Text Classification on Twitter Messages")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home", "Prediction", "EDA", "Visuals", "Meet the team"]
    selection = st.sidebar.selectbox("Choose Option", options)


    # Building out the "Information" page
    if selection == "Home":

        #picture
        st.markdown(f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(image, "rb").read()).decode()}">
        </div>
        """,
            unsafe_allow_html=True)

        #empty space
        st.text("")

        # You can read a markdown file from supporting resources folder
        st.markdown("""
        Many companies are built around lessening one's environmental impact or carbon foot print. They offer products and services 
        that are environmentally friendly and sustainable, in line with their values and ideals. 
        """)

        st.markdown("""
        The purpose of this App is to determine how people perceive climante change and whether or not they belive it is a real threat or not. 
        This information would be used for market research purposses and how diffrenent product/service may be received.
        """)

        st.markdown("""
        We used machine learning to classify whether or not a person believes in climate change, based on their novel tweet data.
        This will provide accurate and robust solutions to companies in oder to segment their consumers according to multiple demographic and geographic categories 
        thus increasing their insights and implement future marketing strategies.
        """)

        st.subheader("Raw Twitter messages")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment','message','tweetid']]) # will write the df to the page



        # Building out the predication page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input

        # Option of classifiying text or dataset
        data_source = ["Select option","Classify Tweet", "Classify Dataset"]
        source_selection = st.selectbox('What to classify?', data_source)

        # Load Our Models
        def load_prediction_models(model_file):
            loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
            return loaded_models

        # Getting the predictions
        def get_keys(val,my_dict):
            for key,value in my_dict.items():
                if val == value:
                    return key

        ### Classify Tweet
        if source_selection == 'Classify Tweet':
            input_text = st.text_area('Enter Text (max. 120 characters)') ##user entering a single text to classify and predict
            all_ml_models = ["Select Option","LSVC"]
            model_choice = st.selectbox("Classify Tweet",all_ml_models)

            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
             
            if st.button('Classify'):
                st.text("Original test ::\n{}".format(input_text))
                text1 = cleaner(input_text) 
                vect_text = tweet_cv.transform([text1]).toarray()
                #first Model Prediction
                if model_choice == 'LSVC':
                    predictor = load_prediction_models("resources/lsvc_model.pkl")
                    prediction = predictor.predict(vect_text)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as:: {}".format(final_result))
        
        # data classification
        if source_selection == 'Classify Dataset':
            all_ml_models = ["Select Option","LSVC"]
            model_choice = st.selectbox("Classify Dataset",all_ml_models)

            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)

            uploaded_dataset = st.checkbox('See uploaded dataset')
            if uploaded_dataset:
                st.dataframe(text_input.head(25))
            
            col = st.text_area('Enter column to classify')

            if st.button("Classify"):
                st.text("Original test ::\n{}".format(text_input))
                text2 = text_input[col].apply(cleaner) ###passing the text through the 'cleaner' function
                vect_text = tweet_cv.transform([text2]).toarray()

                if model_choice == 'LSVC':
                    predictor = load_prediction_models("resources/lsvc_model.pkl")
                    prediction = predictor.predict(vect_text)

                #st.write(prediction)
                text_input['sentiment'] = prediction
                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweets Categorized as:: {}".format(final_result))

                
                csv = text_input.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)


    if selection == 'Visuals':
        # Number of Messages Per Sentiment
        st.write('Distribution of the sentiments')
        # Labeling the target
        raw['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in raw['sentiment']]
        
        # checking the distribution
        values = raw['sentiment'].value_counts()/raw.shape[0]
        labels = (raw['sentiment'].value_counts()/raw.shape[0]).index
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
        explode = (0.05, 0, 0, 0)
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode, colors=colors)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        st.markdown("""There are three class namely """)

               
    if selection == "Meet the team":
        st.subheader('**The Team**')
        st.markdown("""Meet our exceptionally talented "A" team🏆.""" )
        st.text("")
        st.markdown("""Jean-Luc van Zyl""" )
        st.markdown("""Noluthando Ntsangani""" )
        st.markdown("""Sung Hyu Kim""" )
        st.markdown("""Innocentia Pakati""")
        st.markdown("""Hlulani Nkonyani""" )




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
