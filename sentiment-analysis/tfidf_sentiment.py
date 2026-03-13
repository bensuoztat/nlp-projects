from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import numpy as np
from sklearn.metrics import confusion_matrix

df = pd.read_csv("twitter.csv")
document = df["tweet"]

def clean_text(text):
    text =text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+","", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if len(word) > 2]) 
    return text

cleaned_documents = [clean_text(doc) for doc in document]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(cleaned_documents)

#feature_names = vectorizer.get_feature_names_out()
#print("First 20 features:")
#print(feature_names[:20])

y = df["label"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

print("Naive Bayes results:")
print(classification_report(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))

#Logistic Regression model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("Logistic Regression results:")
print(classification_report(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))