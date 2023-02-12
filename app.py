import streamlit as st
import pandas as pd
import numpy as np
import pickle

import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

PAGE_CONFIG = {"page_title":"Project Dissertation - KU","page_icon":":smiley:","layout":"centered"}

my_details = pd.DataFrame({"Name": ['Deepak Jangir Dayanand'],"KU ID": ['K2161089']})
my_details.set_index('Name', inplace=True)


## Importing all the models implemented
model_rf = pickle.load(open('rf_model_v2.pkl', 'rb'))
model_cv = pickle.load(open('cv_model.pkl', 'rb'))

def lemmatize_clean_text(text):

    # Lemmatize words
  def get_pos_tag(tag):
      if tag.startswith('J'):
          return wordnet.ADJ
      elif tag.startswith('V'):
          return wordnet.VERB
      elif tag.startswith('N'):
          return wordnet.NOUN
      elif tag.startswith('R'):
          return wordnet.ADV
      else:
          # Default lemmatization
          return wordnet.NOUN

  regex = [
      r'<[^>]+>', #HTML tags
      r'@(\w+)', # @-mentions
      r"#(\w+)", # hashtags
      r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
      r'[^0-9a-z #+_\\r\\n\\t]', #BAD SYMBOLS
  ]

  lemmatizer = WordNetLemmatizer()


  REPLACE_URLS = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
  REPLACE_HASH = re.compile(r'#(\w+)')
  REPLACE_AT = re.compile(r'@(\w+)')
  REPLACE_HTML_TAGS = re.compile(r'<[^>]+>')
  REPLACE_BY = re.compile(r"[^a-z0-9\-]")

  STOP_WORDS = set(stopwords.words('english'))

  text = text.lower()
  text = REPLACE_HTML_TAGS.sub(' ',text)
  text = REPLACE_URLS.sub('', text)
  text = REPLACE_HASH.sub('', text)
  text = REPLACE_AT.sub('', text)
  text = REPLACE_BY.sub(' ', text)

  text = " ".join(lemmatizer.lemmatize(word.strip(), get_pos_tag(pos_tag([word.strip()])[0][1])) \
                  for word in text.split() if word not in STOP_WORDS and len(word)>3)

  return text

def convert_and_combine(q1,q2):

	questions = [str(q1)] + [str(q2)]

	q1_arr, q2_arr = np.vsplit(model_cv.transform(questions).toarray(),2)
	temp_df1 = pd.DataFrame(q1_arr)
	temp_df2 = pd.DataFrame(q2_arr)

	temp_df = pd.concat([temp_df1, temp_df2], axis=1)

	return temp_df

def extract_features(q1,q2):

	d = {}

	def fetch_common_words(q1,q2):
		w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
		w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
		return len(w1 & w2)

	def total_words(q1,q2):
		w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
		w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))    
		return (len(w1) + len(w2))


	d['q1len'] = len(q1)
	d['q2len'] = len(q2)
	d['q1_no_words'] = len(q1.split(" "))
	d['q2_no_words'] = len(q2.split(" "))
	d['common_words'] = fetch_common_words(q1,q2)
	d['total_words'] = total_words(q1,q2)
	d['shared_words'] = round(d['common_words']/d['total_words'],2)

	return pd.DataFrame([d])
 

st.set_page_config(**PAGE_CONFIG)
def main():
	st.title("CQA duplicate question detection using Random Forest Model")
	with st.sidebar:
		st.subheader("More about the Model")
		menu = ["Home","About"]

		choice = st.sidebar.selectbox(label = "Choose an option",options=menu)
		st.sidebar.table(my_details)

	st.subheader("Test if your set of questions are duplicate or not!!")
	

	if choice == 'Home':
		q1 = st.text_input('Input your question 1 here:') 	
		q2 = st.text_input('Input your question 2 here:')
	
		q1 = lemmatize_clean_text(q1)
		q2 = lemmatize_clean_text(q2)

		if st.button(label='Check Duplication',help='Submit the questions'):
				
				converted = convert_and_combine(q1,q2)
	
				combined_data_features = pd.concat([extract_features(str(q1),str(q2)), converted], axis=1)
		
				value = model_rf.predict([combined_data_features.iloc[0].values])

				if value[0] == 0:
					st.success("The two questions are not Duplicate")
				else:
					st.error('The two questions are duplicate')

	elif choice == 'About':
		st.markdown(
								"""
								The reason why the Random Forest Classifier was chosen is that it performed the best on the validation dataset in comparison to the other models.
								- It first pre-processes individual questions posted here to eliminate all the bad symbols and lemmatizes the words.
								- Next the preprocessed questions are converted to 3000 numerical features each and upon these features, feature engineering is applied and more new features are dervied.
								- Post everything, the data is passed to the Random Forest model and the outputs are predicted and displayed on the screen. 
								"""
								)



if __name__ == '__main__':
	main()