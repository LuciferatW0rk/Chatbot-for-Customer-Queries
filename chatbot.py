import re
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer


def chat(queries, responses):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(queries)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: bye")
            break
        query = [user_input]
        query_tfidf = tfidf.transform(query)
        similarity = cosine_similarity(query_tfidf, tfidf_matrix)
        response_index = np.argmax(similarity)
        print("Chatbot: " + responses[response_index])


data = open('DATA.txt', 'r')
Queries = data.readlines()

response = open('Responses.txt', 'r')
Responses = response.readlines()

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
sno = SnowballStemmer('english')


def getcleanedtext(text):
    text = text.lower()
    # tokenization and stopword removal
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [sno.stem(token) for token in new_tokens]

    clean_text = " ".join(stemmed_tokens)
    return clean_text

queries = [getcleanedtext(i) for i in Queries]

chat(queries, Responses)
