import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(data):
    data = word_tokenize(data.lower())

    mat = str.maketrans('', '', string.punctuation)
    data = [data_point.translate(mat) for data_point in data if data_point.isalpha()]

    stop_words = set(stopwords.words('english'))
    data = [data_point for data_point in data if data_point not in stop_words]

    lemmatized_data = [lemmatizer.lemmatize(data_point) for data_point in data]

    data = ' '.join(lemmatized_data)
    return data


dataset = pd.read_json(r'data\News_Category_Dataset_IS_course.json', lines=True)
# print(dataset.isnull().any())
dataset = dataset.dropna()
# print(dataset.isnull().any())

# Ignore links
dataset = dataset.drop(['link'], axis=1)

# Add preprocessed attributes
dataset['clean_headline'] = dataset.apply(lambda x: preprocess(x["headline"]), axis=1)
dataset['clean_short_description'] = dataset.apply(lambda x: preprocess(x["short_description"]), axis=1)

dataset['clean_text'] = dataset['clean_headline'] + " " + dataset['clean_short_description'] + " " + dataset["date"].dt.strftime('%Y')

dataset.to_json(r"data\preprocess_data.json", orient="records", lines=True)
