# nlp-projects
A set of NLP projects demonstrating text preprocessing, analysis, and modeling

## Text Preprocessing

Cleaning, tokenization, and lemmatization of text data.

**Files included:**  
preprocess.py, spam.csv

## Sentiment Analysis

### English Tweet Sentiment Analysis
- Uses VADER to generate sentiment scores from tweets.   
  -Files included: vader_sentiment.py, twitter.csv
- Uses TF-IDF features and trains Naive Bayes and Logistic Regression models.  
  -Files included: tfidf_sentiment.py, twitter.csv

### Turkish Movie Reviews Sentiment Analysis
- Sentiment classification using Zemberek for lemmatization.
- Oversampling used to handle class imbalance.
- Models: Naive Bayes, Logistic Regression.  
  -Files included: zemberek_sentiment.py, turkish_movie_sentiment_dataset.csv
