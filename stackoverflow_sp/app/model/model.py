import tensorflow as tf
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras.models import load_model
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve(strict=True).parent
model_path = BASE_DIR / 'model.h5'
tokenizer_path = BASE_DIR / 'tokenizer.pickle'

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model(model_path)

# stopwords_dir = os.path.join(nltk.data.find('corpora'), 'stopwords')
# if not os.path.exists(stopwords_dir):
#     os.mkdir(stopwords_dir)


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize Snowball stemmer
stemmer = SnowballStemmer('english')

# preprocessing
def input_preprocessing(inp):


    inp_title = inp.title
    inp_body = inp.body
    inp_tags = inp.tags

    # Regular expression patterns (same as before)
    code_snippet_pattern = re.compile(r'<code>.*?</code>')  # Detects code snippets
    url_pattern = re.compile(r'https?://\S+|www\.\S+')  # Detects URLs
    number_pattern = re.compile(r'\b\d+\b')  # Detects standalone numbers
    extended_single_char_pattern = re.compile(r'\b\w\b|\w(?=\d)|\d(?=\w)')  # Detects single characters and numbers attached to words

    def preprocess_text(text):
        if text == None:
            return ""
        
        # Remove code snippets and HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        code_elements = soup.find_all(['code', 'pre', 'a', 'img'])

        # Remove code elements from the HTML
        for code_element in code_elements:
            code_element.decompose()

        # Get the cleaned HTML without code snippets
        text = soup.prettify()

        text = BeautifulSoup(text, 'lxml').get_text()

        # Convert text to lowercase
        text = text.lower()

        # Remove URLs, numbers, and single characters
        text = url_pattern.sub(' ', text)
        text = number_pattern.sub(' ', text)
        text = extended_single_char_pattern.sub(' ', text)
        text = code_snippet_pattern.sub(' ', text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords and non-alphabetic characters, apply stemming
        stop_words = set(stopwords.words('english'))
        tokens = [stemmer.stem(word) if len(word) > 3 else word for word in tokens if word.isalpha() and word not in stop_words and word != '' and len(word) > 1]

        return ' '.join(tokens)

    def preprocess_tags(tags_text):
        if tags_text == None:  
            return ""
        tags_text = tags_text.replace("|", " ")
        return preprocess_text(tags_text)

        
    preprocessed_tags = preprocess_tags(preprocess_text(inp_tags))
    preprocessed_title = preprocess_text(inp_title)
    preprocessed_body = preprocess_text(inp_body)
 
    result = preprocessed_title + ' ' + preprocessed_body + ' ' + preprocessed_tags
    
    result = re.sub(r'\s+', ' ', result).strip()

    return result


# parse input for model
def make_sequences(inp: str):
    max_len = 256
    sequences = tokenizer.texts_to_sequences([inp])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences

# predict
def model_predict(inp):
    binary_pred = model.predict(inp)
    label = (binary_pred > 0.5).astype(int)
    label_scalar = np.array([l for la in label for l in la]).item()  
    int_to_label = {1: "SP", 0: "nonSP"}
    return int_to_label[label_scalar]    