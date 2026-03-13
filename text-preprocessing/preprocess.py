import  re
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
import spacy
from nltk.corpus import stopwords

#download resources
nltk.download('stopwords')

#stopwords             
stop_words = set(stopwords.words("english"))

#load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.rename(columns={"v2": "message"})

documents = df["message"]

#text cleaning function
def clean_text(text):

    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([
        word for word in text.split()
        if len(word) > 2 and word not in stop_words
    ])
    return text

#clean documents
cleaned_documents = [clean_text(doc) for doc in documents]

#load spacy model
nlp = spacy.load("en_core_web_sm")

tokens = []
lemmas = []

for doc in cleaned_documents:
    document = nlp(doc)
    tokenize = [token.text for token in document]
    lemmatize = [token.lemma_ for token in document]

    tokens.append(tokenize)
    lemmas.append(lemmatize)

#add columns
df['tokens'] = tokens
df['lemmas'] = lemmas


df_to_show = df[['message', 'tokens', 'lemmas']]

html_file = "preprocessing_results.html"
df_to_show.to_html(html_file, index=False)
print("HTML file created:", html_file)


