
# Dataset link: https://www.kaggle.com/datasets/mustfkeskin/turkish-movie-sentiment-analysis-dataset

from zemberek import TurkishMorphology
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.metrics import confusion_matrix

#load Turkish stopwords
stop_words = set(stopwords.words("turkish"))

#load the dataset
df = pd.read_csv("turkish_movie_sentiment_dataset.csv")
document = df["comment"]

morphology = TurkishMorphology.create_with_defaults()

def clean_text(text):
    text =text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+","", text)
    text = re.sub(r"[^a-zçğıöşü\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if len(word) > 2]) 
    return text


def preprocess_text(text):
    text = clean_text(text)
    words = text.split()
    lemmatized = []
    for word in words:
        if word in stop_words:
            continue
        analysis = list(morphology.analyze(word))
        if analysis:
            lemmatized.append(analysis[0].get_stem())#take the stem of the word
        else:
            lemmatized.append(word)#if fails keep the original word
    return " ".join(lemmatized)

#preprocess first 10k comments
cleaned_documents = [preprocess_text(doc) for doc in document[:10000]]

#label points into sentiment classes
def label_points(point):
    if point <= 2:
        return "negative"
    elif point == 3:
        return None
    else:
        return "positive"


df["point"] = df["point"].str.replace(",", ".").astype(float)
df["point"] = df["point"].apply(label_points)
df = df.dropna(subset=["point"])


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(cleaned_documents)

y = df["point"][:10000]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#oversample minority class
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)

X_res, y_res = ros.fit_resample(X_train, y_train)



#Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb_model = MultinomialNB()
nb_model.fit(X_res, y_res)
nb_pred = nb_model.predict(X_test)

print("Naive Bayes results:")
print(classification_report(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))

#Logistic Regression model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_res, y_res)
lr_pred = lr_model.predict(X_test)

print("Logistic Regression results:")
print(classification_report(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred)) 