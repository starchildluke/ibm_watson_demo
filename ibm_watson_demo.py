import streamlit as st
from ibm_watson import NaturalLanguageUnderstandingV1, ApiException
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, ConceptsOptions, CategoriesOptions, EmotionOptions, SemanticRolesOptions, SentimentOptions
import pandas as pd

# NLP dictionaries

entities_dict = {
	"Name": [],
	"Type": [],
	"Confidence": [],
	"Sentiment Score": [],
	"Label": [],
	"Relevance": [],
	"Sadness": [],
	"Joy": [],
	"Fear": [],
	"Disgust": [],
	"Anger": []
}

keywords_dict = {
	"Keyword": [],
	"Count": [],
	"Sentiment Score": [],
	"Label": [],
	"Relevance": [],
	"Sadness": [],
	"Joy": [],
	"Fear": [],
	"Disgust": [],
	"Anger": []
}

concepts_dict = {
	"Concept": [],
	"Relevance": []
}

categories_dict = {
	"Category": [],
	"Score": []
}

# Layout text

st.title('IBM Watson Natural Language Understanding - Text Analysis')

st.header('Natural Language Understanding includes a set of text analytics features that you can use to extract meaning from unstructured data. Try your own text or extract text from a URL to analyse using the base model.')

input_type = st.radio('Select your input type', ['Text','URL'])

api_key = st.text_input('Enter API key')
endpoint_url = st.text_input('Enter endpoint URL')

# Watson credentials

authenticator = IAMAuthenticator(api_key)
natural_language_understanding = NaturalLanguageUnderstandingV1(
	version='2020-08-01',
	authenticator=authenticator)

natural_language_understanding.set_service_url(endpoint_url)

# Script

if input_type == 'Text':
	txt = st.text_area('Enter text to be analysed...')
else:
	link = st.text_input('Enter URL to be analysed')

analyse_button = st.button('Analyse Text')

def nlp_analysis(input_type):

	if input_type == 'Text':
		response = natural_language_understanding.analyze(
			text=txt,
			language="en", 
			features=Features(
				entities=EntitiesOptions(sentiment=True, emotion=True, limit=50),
				keywords=KeywordsOptions(sentiment=True, emotion=True, limit=50),
				concepts=ConceptsOptions(limit=10),
				categories=CategoriesOptions(limit=10),
				semantic_roles=SemanticRolesOptions()
				)).get_result()

		dictionaries(response)

	else:
		response = natural_language_understanding.analyze(
			url=link,
			language="en", 
			features=Features(
				entities=EntitiesOptions(sentiment=True, emotion=True, limit=50),
				keywords=KeywordsOptions(sentiment=True, emotion=True, limit=50),
				concepts=ConceptsOptions(limit=10),
				categories=CategoriesOptions(limit=10),
				semantic_roles=SemanticRolesOptions()
				)).get_result()

		dictionaries(response)

def dictionaries(response):

		for entity in range(0,len(response["entities"])):
			entities_dict["Name"].append(response["entities"][entity]["text"])
			entities_dict["Type"].append(response["entities"][entity]["type"])
			entities_dict["Confidence"].append(response["entities"][entity]["confidence"])
			entities_dict["Sentiment Score"].append(response["entities"][entity]["sentiment"]["score"])
			entities_dict["Label"].append(response["entities"][entity]["sentiment"]["label"])
			entities_dict["Relevance"].append(response["entities"][entity]["relevance"])
			entities_dict["Sadness"].append(response["entities"][entity]["emotion"]["sadness"])
			entities_dict["Joy"].append(response["entities"][entity]["emotion"]["joy"])
			entities_dict["Fear"].append(response["entities"][entity]["emotion"]["fear"])
			entities_dict["Disgust"].append(response["entities"][entity]["emotion"]["disgust"])
			entities_dict["Anger"].append(response["entities"][entity]["emotion"]["anger"])

		for keyword in range(0,len(response["keywords"])):
			keywords_dict["Keyword"].append(response["keywords"][keyword]["text"])
			keywords_dict["Count"].append(response["keywords"][keyword]["count"])
			keywords_dict["Sentiment Score"].append(response["keywords"][keyword]["sentiment"]["score"])
			keywords_dict["Label"].append(response["keywords"][keyword]["sentiment"]["label"])
			keywords_dict["Relevance"].append(response["keywords"][keyword]["relevance"])
			keywords_dict["Sadness"].append(response["keywords"][keyword]["emotion"]["sadness"])
			keywords_dict["Joy"].append(response["keywords"][keyword]["emotion"]["joy"])
			keywords_dict["Fear"].append(response["keywords"][keyword]["emotion"]["fear"])
			keywords_dict["Disgust"].append(response["keywords"][keyword]["emotion"]["disgust"])
			keywords_dict["Anger"].append(response["keywords"][keyword]["emotion"]["anger"])

		for concept in range(0,len(response["concepts"])):
			concepts_dict["Concept"].append(response["concepts"][concept]["text"])
			concepts_dict["Relevance"].append(response["concepts"][concept]["relevance"])

		for category in range(0,len(response["categories"])):
			categories_dict["Category"].append(response["categories"][category]["label"])
			categories_dict["Score"].append(response["categories"][category]["score"])
		
def data():

	st.header('Entities')

	entities_df = pd.DataFrame(entities_dict)
	st.dataframe(entities_df)

	st.header('Keywords')

	keywords_df = pd.DataFrame(keywords_dict)
	st.dataframe(keywords_df)

	st.header('Concepts')

	concepts_df = pd.DataFrame(concepts_dict)
	st.dataframe(concepts_df)

	st.header('Categories')

	categories_df = pd.DataFrame(categories_dict)
	st.dataframe(categories_df)

if analyse_button and input_type == 'Text':
	nlp_analysis('Text')
	data()
elif analyse_button and input_type == 'URL':
	nlp_analysis('URL')
	data()
