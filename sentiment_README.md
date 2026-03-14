# Sentiment Analysis & Stock Prediction Dashboard

An interactive Streamlit dashboard covering sentiment analysis concepts and their application to stock return prediction. Built for the **Global MBA (GMBA)** program at **SP Jain School of Global Management**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## Overview

This dashboard demonstrates how text sentiment from financial news can be quantified and used as a feature in stock prediction models. It covers sentiment analysis theory, interactive scoring demos, and a head-to-head comparison of prediction models with and without sentiment features.

**Dataset:** Synthetic stock prices + financial news headlines — generated automatically, no setup required.

---

## Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Home & Overview** | Workflow, dataset preview, key statistics |
| 2 | **Sentiment Analysis Theory** | Lexicon-based, VADER, and ML approaches explained with examples |
| 3 | **Dataset Exploration** | Candlestick chart, return distribution, news headlines browser |
| 4 | **Sentiment Scoring Demo** | Type any text and see Lexicon, VADER, and TextBlob scores side-by-side with word-level matching |
| 5 | **Sentiment Visualizations** | Sentiment over time, distribution, scatter plot vs returns, rolling averages, positive/negative ratio |
| 6 | **Stock Prediction — Without Sentiment** | Baseline model using only price-based technical features |
| 7 | **Stock Prediction — With Sentiment** | Same model with sentiment features added, with look-ahead bias prevention |
| 8 | **Head-to-Head Comparison** | Side-by-side metrics, direction accuracy, prediction overlay, automated verdict |
| 9 | **Summary & Takeaways** | Concept reference, method comparison, evaluation metrics, limitations |

---

## Getting Started

### Run Locally

```bash
git clone https://github.com/your-username/sentiment-stock-dashboard.git
cd sentiment-stock-dashboard

pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"

streamlit run sentiment_dashboard.py
```

### Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in → **New app**
3. Select repo → set main file to `sentiment_dashboard.py` → **Deploy**

---

## Repository Structure

```
sentiment-stock-dashboard/
├── sentiment_dashboard.py   # Streamlit application
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `pandas` / `numpy` | Data manipulation |
| `plotly` | Interactive charts |
| `scikit-learn` | Prediction models and metrics |
| `scipy` | Statistical analysis |
| `nltk` | VADER sentiment analysis |
| `textblob` | Alternative sentiment scoring |

---

## Author

**Dr. Anshul Gupta**
Associate Professor & Area Head — Technology Management
SP Jain School of Global Management

---

<p align="center">
  <em>Built with ❤️ for the Global MBA Program</em>
</p>
