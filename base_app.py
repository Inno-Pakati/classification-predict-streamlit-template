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
import streamlit as st
import joblib,os
import base64

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#logo
LOGO_IMAGE = "Images/download.jpg"

st.markdown(
    """
    <style>
    	.container {
        	display: flex;
    }
    	.logo-text {
        	font-weight:700 !important;
        	font-size:100px !important;
        	color: #f9a01b !important;
        	padding-top: 500px !important;
			padding-bottom: 800px !important;
    }
    	.logo-img {
        	float:right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Prediction", "EDA", "Visuals", "Meet the team"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Home":
		   #logo
		st.markdown(f"""
    	<div class="container">
    		<img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
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
			st.write(raw[['message']]) # will write the df to the page

	if selection == "EDA":
		st.info("This is the EDA page")
		st.markdown("""""" )

	if selection == "Meet the team":
		st.subheader('**The Team**')
		st.markdown("""
		Meet our exceptionally talented "A" team🏆.
		""" )
		st.text("")
		st.markdown("""Jean-Luc van Zyl""" )
		st.markdown("""Noluthando Ntsangani""" )
		st.markdown("""Sung Hyu Kim""" )
		st.markdown("""Innocentia Pakati""" )
		st.markdown("""Hlulani Nkonyani""" )


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
