# Sentiment Analysis – End to End (Flask + Streamlit)

Predicts whether a review is **Positive** or **Negative**. The model is trained on ~3,100 Amazon Alexa reviews with an XGBoost classifier (about 94% test accuracy).

- Type one sentence and get its sentiment, or
- Upload a CSV file of sentences and download the predictions with a pie chart.

**No API key needed.** Everything runs on your computer.

## How to run (step by step)

You need **Python 3.10 or newer** ([download](https://www.python.org/downloads/)).

**1. Download the project**

```bash
git clone https://github.com/amitmohanty022/Sentiment-Analysis-End-to-End-Flask2.git
cd Sentiment-Analysis-End-to-End-Flask2
```

**2. Create a virtual environment and activate it**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

**4. Start the API**

```bash
python api.py
```

The first time, it trains the model automatically, which takes under a minute, and saves it in `Models/`. Leave this terminal open.

**5. Open the app**

- **Web page:** go to http://127.0.0.1:5000
- **Streamlit app (optional):** open a second terminal, activate the venv (step 2), and run:

  ```bash
  streamlit run main.py
  ```

**6. Use it**

- Type a sentence and click **Predict**, or
- Upload a CSV that has a column named **`Sentence`** and click **Predict**, then download `Predictions.csv`.

Example CSV:

```csv
Sentence
I love this speaker
It stopped working after a week
```

## Project files

| File | Purpose |
|---|---|
| `api.py` | Flask API (`/predict`, `/test`) and web page |
| `main.py` | Streamlit front end that calls the API |
| `train_model.py` | Trains the model and saves it to `Models/` (run again any time to retrain) |
| `templates/landing.html` | Web page served at `/` |
| `Data Exploration & Modelling (1).ipynb` | Original data exploration and model comparison |
| `amazon_alexa (1).tsv` | Training data |

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` in Streamlit | Start `python api.py` first (step 4) and keep it running. |
| `KeyError: 'Sentence'` | Your CSV must have a header row with a column called `Sentence`. |
| Port 5000 already in use (macOS AirPlay) | Turn off *AirPlay Receiver* in System Settings, or stop the other program using port 5000. |
