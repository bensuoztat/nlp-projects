import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#load dataset
df = pd.read_csv("twitter.csv")

#text cleaning and preprocessing
def preprocess_text(text):

    tokens = word_tokenize(text.lower())

    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = " ".join(lemmatized_tokens)

    return processed_text

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    scores = analyzer.polarity_scores(text)

    if scores["compound"] >= 0.05:
        return 1
    else:
        return 0

#apply sentiment analysis
df["sentiment"] = df["tweet"].apply(get_sentiment)

#evaluation
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(df["label"], df["sentiment"]))

print("Classification Report:")
print(classification_report(df["label"], df["sentiment"]))

print("Predicted Sentiment Distribution")
counts = df["sentiment"].value_counts()
print(f"Positive (1): {counts.get(1, 0)}")
print(f"Negative (0): {counts.get(0, 0)}")