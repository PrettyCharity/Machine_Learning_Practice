# Recurrent Neural Networks for Stock Prediction

This directory contains experimental Jupyter Notebooks and a production-ready script for predicting short-term stock prices using Recurrent Neural Networks (GRU) via PyTorch.

## Overview

The `predict_future.py` tool is a robust Command Line Interface (CLI) that implements a Gated Recurrent Unit (GRU) to predict stock prices 1 to 30 days into the future.
A newly enhanced version, `predict_future_enhanced.py`, integrates multiple sources for sentiment analysis.

### Methodology

To achieve the highest short-term predictive accuracy, the algorithm integrates the following features:
- **Baseline Data:** Historical `Open`, `High`, `Low`, `Close`, and `Volume` prices fetched dynamically via the `yfinance` API.
- **Trend Indicators:** 20-day Simple Moving Average (SMA) and Exponential Moving Average (EMA) calculated via `pandas_ta` to smooth out noise.
- **Sentiment Analysis:** A Natural Language Processing (NLP) integration that parses recent news summaries using `nltk.sentiment.vader` to assess the current market momentum. The `predict_future_enhanced.py` script further improves this by scraping and combining sentiment scores from multiple free channels (Yahoo Finance, Finviz, and Google News RSS).

### Experimental Results

To determine the best approach for sentiment analysis, experiments were conducted on multiple stock tickers (MU, VST, MRVL, NVDA, S, ZS, MOD, VRT) over a 30-day holdout test set using different sentiment injection strategies.

| Model Mode       | Average RMSE | Average MAPE |
|------------------|--------------|--------------|
| Combined Multi   | 18.41        | 0.0843       |
| Baseline (YF)    | 16.70        | 0.0883       |
| Google News      | 18.39        | 0.0919       |
| Combined Average | 16.77        | 0.1001       |
| Finviz           | 22.85        | 0.1052       |

Based on the MAPE (Mean Absolute Percentage Error) metric, the `Combined Multi` mode (providing separate feature columns for Yahoo Finance, Finviz, and Google News sentiments) achieved the best predictive performance overall. This strategy was implemented in `predict_future_enhanced.py`.

The model is trained on a 60-day sliding window of these combined features. To estimate prices multiple days ahead, the script iteratively projects tomorrow's price, rebuilds the input array with that synthetic prediction, and steps forward until the target period is met.

## Execution

Ensure you have the necessary dependencies installed:
```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib pandas-ta nltk beautifulsoup4 feedparser requests
```

### Usage
Run the prediction script by passing a ticker symbol and the number of days you wish to forecast (between 1 and 30).

```bash
python predict_future_enhanced.py --ticker <TICKER> --period <DAYS>
```

**Example:**
```bash
$ python predict_future_enhanced.py --ticker AAPL --period 5
Fetching data and calculating features for AAPL...
Fetching Yahoo Finance news...
Fetching Finviz news...
Fetching Google News...
Training GRU model with 10 features (including Multi-Source Sentiment)...
Simulating future predictions for 5 days...

==================================================
PREDICTION RESULTS
==================================================
Ticker:           AAPL
Period:           5 days
Last Known Price: $258.63
Estimated Price:  $261.17
Overall Direction:Upward
==================================================
```
