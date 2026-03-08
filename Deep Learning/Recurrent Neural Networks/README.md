# Recurrent Neural Networks for Stock Prediction

This directory contains experimental Jupyter Notebooks and a production-ready script for predicting short-term stock prices using Recurrent Neural Networks (GRU) via PyTorch.

## Overview

The `predict_future.py` tool is a robust Command Line Interface (CLI) that implements a Gated Recurrent Unit (GRU) to predict stock prices 1 to 30 days into the future.

### Methodology

To achieve the highest short-term predictive accuracy, the algorithm integrates the following features:
- **Baseline Data:** Historical `Open`, `High`, `Low`, `Close`, and `Volume` prices fetched dynamically via the `yfinance` API.
- **Trend Indicators:** 20-day Simple Moving Average (SMA) and Exponential Moving Average (EMA) calculated via `pandas_ta` to smooth out noise.
- **Sentiment Analysis:** A Natural Language Processing (NLP) integration that parses recent news summaries using `nltk.sentiment.vader` to assess the current market momentum.

The model is trained on a 60-day sliding window of these combined features. To estimate prices multiple days ahead, the script iteratively projects tomorrow's price, rebuilds the input array with that synthetic prediction, and steps forward until the target period is met.

## Execution

Ensure you have the necessary dependencies installed:
```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib pandas-ta nltk
```

### Usage
Run the prediction script by passing a ticker symbol and the number of days you wish to forecast (between 1 and 30).

```bash
python predict_future.py --ticker <TICKER> --period <DAYS>
```

**Example:**
```bash
$ python predict_future.py --ticker AAPL --period 5
Fetching data and calculating features for AAPL...
Training GRU model with 8 features (including Sentiment)...
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
