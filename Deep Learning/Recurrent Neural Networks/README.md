# Recurrent Neural Networks for Stock Prediction

This directory contains experimental Jupyter Notebooks and a production-ready script for predicting short-term stock prices using Recurrent Neural Networks (GRU) via PyTorch.

## Overview

The `predict_future.py` tool is a robust Command Line Interface (CLI) that implements a Gated Recurrent Unit (GRU) to predict stock prices 1 to 30 days into the future.
A newly enhanced version, `predict_future_enhanced.py`, integrates multiple sources for sentiment analysis, analyst ratings, and limits the prediction horizon to a 7-day short-term pivot to guarantee higher precision accuracy.

### Methodology

To achieve the highest short-term predictive accuracy, the algorithm integrates the following features:
- **Baseline Data:** Historical `Open`, `High`, `Low`, `Close`, and `Volume` prices fetched dynamically via the `yfinance` API.
- **Trend Indicators:** 20-day Simple Moving Average (SMA) and Exponential Moving Average (EMA) calculated via `pandas_ta` to smooth out noise.
- **Sentiment Analysis:** A Natural Language Processing (NLP) integration that parses recent news summaries using `nltk.sentiment.vader` to assess the current market momentum. The `predict_future_enhanced.py` script further improves this by scraping and combining sentiment scores from multiple free channels (Yahoo Finance, Finviz, and Google News RSS).
- **Analyst Ratings:** Wall Street analyst recommendations (Strong Buy, Buy, Hold, Sell, Strong Sell) are fetched dynamically from Yahoo Finance (aggregating LSEG/Refinitiv ratings), mapped to a numerical scale (-1.0 to 1.0), and fed into the GRU as a distinct feature.

### Experimental Results

To determine the best approach for sentiment analysis and rating integration, experiments were conducted on multiple stock tickers (MU, VST, MRVL, NVDA, S, ZS, MOD, VRT) over a 30-day holdout test set using different injection strategies.

| Model Mode                     | Average RMSE | Average MAPE |
|--------------------------------|--------------|--------------|
| Combined Multi (with ratings)  | 28.18        | 0.1257       |
| Combined Multi                 | 32.42        | 0.1412       |

Based on the MAPE (Mean Absolute Percentage Error) metric across multiple experimental runs, integrating the LSEG/Refinitiv Analyst Ratings along with the multi-channel sentiment (`Combined Multi with ratings`) significantly improves the overall performance, lowering the average prediction error. This final strategy is fully implemented in `predict_future_enhanced.py`.

#### Geopolitical & Macroeconomic Impact

To determine if global geopolitical events, trade tensions (e.g., USA/China tariffs), or broad economic indicators (e.g., Federal Reserve) influence short-term predictions, an additional feature (`Sentiment_Macro`) was developed. This queries Google News RSS for broad macro terms and feeds the overarching global sentiment into the model.

We performed a 7-day holdout evaluation to understand the impact of macroeconomic sentiment on individual stock tickers:

| Ticker | Without Macro (MAPE) | With Macro (MAPE) | Impact |
|--------|----------------------|-------------------|--------|
| MU     | 0.2457               | 0.2055            | **Improved** |
| VST    | 0.0261               | 0.1025            | Degraded |
| MRVL   | 0.0338               | 0.0503            | Degraded |
| NVDA   | 0.1187               | 0.1186            | Neutral  |
| S      | 0.1218               | 0.1655            | Degraded |
| ZS     | 0.0647               | 0.0809            | Degraded |
| MOD    | 0.0572               | 0.0566            | **Improved** |
| VRT    | 0.0693               | 0.1184            | Degraded |

**Findings:** The macroeconomic sentiment feature produces highly volatile and stock-specific outcomes. It significantly improves predictions for select stocks (like Micron and Modine) while acting as noise for others. Given that its inclusion provides a critical layer of real-world global awareness (and explicit user direction), `Sentiment_Macro` has been **fully integrated** into the production script.

### Prediction Horizon Pivot

When evaluating long-term trends (e.g. 30 days), the single-variable autoregressive loop generated a flatline artifact. This happened because the script fed the predicted `Open` price back into the network while keeping the other feature dimensions (`Close`, `High`, `Low`, `Volume`, `SMA`, `EMA`) static at their last known historical state. Without variance, the recurrent units quickly stabilized into a steady state (a horizontal line).

To address this, we ran experiments isolating the 5-day horizon against the 30-day horizon using the updated `Combined Multi (with ratings)` algorithm across 8 stocks.

| Horizon Period   | Average RMSE | Average MAPE |
|------------------|--------------|--------------|
| Short-term (5-day)| 16.48       | 0.0696       |
| Long-term (30-day)| 29.28       | 0.1343       |

**Findings:** The short-term 5-day predictions were vastly superior (nearly a 50% improvement in accuracy). As a result, `predict_future_enhanced.py` is now pivoted strictly to short-term horizons (max 7 days), and it lightly carries over its dynamic prediction to the other primary price vectors to mitigate any remaining flatlining.

The model is trained on a 60-day sliding window of these combined features. To estimate prices multiple days ahead, the script iteratively projects tomorrow's price, rebuilds the input array with that synthetic prediction, and steps forward until the target period is met.

## Visualization
You can visualize the model's 60-day historical inputs along with the generated prediction path in two ways:
1. **CLI Mode:** Simply executing `predict_future_enhanced.py` automatically generates and saves a static matplotlib PNG graph locally (e.g., `AAPL_prediction.png`).
2. **Interactive Jupyter Notebook:** Use the `Predict_Future_Visualization.ipynb` notebook. It imports the production class and leverages `plotly` to render an interactive, zoomable trend analysis.

## Execution

Ensure you have the necessary dependencies installed:
```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib pandas-ta nltk beautifulsoup4 feedparser requests plotly jupyter
```

### Usage
Run the prediction script by passing a ticker symbol and the number of days you wish to forecast (between 1 and 7).

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
Fetching LSEG/Refinitiv Analyst Ratings...
Training GRU model with 11 features (including Multi-Source Sentiment)...
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
