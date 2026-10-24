# Sentiment Analysis End-to-End (Flask + XGBoost)

An end-to-end text sentiment project trained on ~3,150 Amazon Alexa product reviews. It goes from data exploration and model training in a notebook to a Flask web app and API that predict whether a review is **Positive** or **Negative**, for a single sentence or a whole CSV file.

![App screenshot](Screenshot%202025-02-13%20144305.png)

## 🔗 Live demo

Not hosted yet. Deploy it for free on [Render](https://render.com): **New → Blueprint**, pick this repository, and Render reads `render.yaml` (it installs the requirements, trains the model and starts `gunicorn api:app`). You can also explore the modelling notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amitmohanty022/Sentiment-Analysis-End-to-End-Flask2/blob/main/Data%20Exploration%20%26%20Modelling%20%281%29.ipynb)

## How it works

1. **Preprocessing:** keep letters only, lowercase, remove English stop words, Porter stemming.
2. **Features:** `CountVectorizer(max_features=2500)` followed by `MinMaxScaler`.
3. **Model:** `XGBClassifier` (about 94% accuracy on a 30% hold-out split).
4. **Serving:** `api.py` (Flask) serves the web page at `/` and a prediction endpoint at `/predict`.

## Project structure

| Path | Purpose |
| --- | --- |
| `Data Exploration & Modelling (1).ipynb` | EDA and model comparison (Random Forest, Decision Tree, XGBoost) |
| `train.py` | Re-creates the final pipeline and saves it to `Models/` |
| `api.py` | Flask app and REST API (trains the model automatically on first start if `Models/` is missing) |
| `templates/landing.html` | Web UI: single text or CSV upload with a sentiment pie chart |
| `main.py` | Optional Streamlit front end that calls the API |
| `amazon_alexa (1).tsv` | Training data |

## Run locally

```bash
git clone https://github.com/amitmohanty022/Sentiment-Analysis-End-to-End-Flask2.git
cd Sentiment-Analysis-End-to-End-Flask2
pip install -r requirements.txt
python train.py            # creates Models/*.pkl
python api.py              # http://localhost:5000
```

Optional Streamlit front end (with the API running):

```bash
streamlit run main.py
```

## API

```bash
# single text
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "I love my Echo"}'
# -> {"prediction": "Positive"}

# bulk: CSV with a "Sentence" column, returns Predictions.csv
curl -X POST http://localhost:5000/predict -F "file=@reviews.csv" -o Predictions.csv
```

## Author

**Amit Kumar Mohanty** · [GitHub](https://github.com/amitmohanty022) · [LinkedIn](https://www.linkedin.com/in/amitkrmohanty/)
