"""Train the sentiment model and save it to Models/.

This reproduces the pipeline from "Data Exploration & Modelling (1).ipynb":
Porter-stemmed text -> CountVectorizer(2500) -> MinMaxScaler -> XGBoost.

Run once before starting the API:  python train_model.py
"""
import os
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

DATA_FILE = "amazon_alexa (1).tsv"
MODELS_DIR = "Models"


def main():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    data = pd.read_csv(DATA_FILE, delimiter="\t", quoting=3).dropna(subset=["verified_reviews"])
    print(f"Dataset shape : {data.shape}")

    corpus = []
    for text in data["verified_reviews"].astype(str):
        review = re.sub("[^a-zA-Z]", " ", text).lower().split()
        corpus.append(" ".join(stemmer.stem(w) for w in review if w not in stop_words))

    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(corpus).toarray()
    y = data["feedback"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

    scaler = MinMaxScaler()
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)

    model_xgb = XGBClassifier()
    model_xgb.fit(X_train_scl, y_train)
    print(f"Training accuracy : {model_xgb.score(X_train_scl, y_train):.3f}")
    print(f"Testing accuracy  : {model_xgb.score(X_test_scl, y_test):.3f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    pickle.dump(cv, open(os.path.join(MODELS_DIR, "countVectorizer.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb"))
    pickle.dump(model_xgb, open(os.path.join(MODELS_DIR, "model_xgb.pkl"), "wb"))
    print(f"Saved model files to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
