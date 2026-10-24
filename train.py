"""Train the sentiment model and save it to Models/.

This reproduces the final pipeline from "Data Exploration & Modelling (1).ipynb":
Porter-stemmed reviews -> CountVectorizer(max_features=2500) -> MinMaxScaler -> XGBClassifier.

Usage:
    python train.py
"""

import os
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

DATA_FILE = "amazon_alexa (1).tsv"
MODELS_DIR = "Models"


def main():
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    data = pd.read_csv(DATA_FILE, delimiter="\t", quoting=3)
    data = data.dropna()

    corpus = []
    for review in data["verified_reviews"]:
        words = re.sub("[^a-zA-Z]", " ", review).lower().split()
        corpus.append(" ".join(stemmer.stem(w) for w in words if w not in stop_words))

    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(corpus).toarray()
    y = data["feedback"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

    scaler = MinMaxScaler()
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)

    model = XGBClassifier()
    model.fit(X_train_scl, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test_scl)):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, obj in [("countVectorizer.pkl", cv), ("scaler.pkl", scaler), ("model_xgb.pkl", model)]:
        with open(os.path.join(MODELS_DIR, name), "wb") as f:
            pickle.dump(obj, f)
    print(f"Saved models to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
