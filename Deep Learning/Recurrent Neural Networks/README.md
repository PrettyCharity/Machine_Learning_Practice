# Recurrent Neural Networks for Stock Prediction

This directory contains experimental and production-ready code for predicting short-term stock prices using Recurrent Neural Networks (LSTMs and GRUs) via PyTorch.

## Feature Engineering Investigation

To improve the baseline short-term predictive accuracy of the GRU model (which only utilized `Open`, `High`, `Low`, `Close`, and `Volume`), an investigation into additional technical indicators was conducted.

Using the `pandas_ta` library, various indicators were appended to the dataset, including:
- **Trend Indicators:** Simple Moving Average (SMA), Exponential Moving Average (EMA)
- **Momentum Indicators:** Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD)
- **Volatility Indicators:** Bollinger Bands

### Results

The experiment evaluated the Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE) across several feature combinations on the `AAPL` ticker:

| Feature Set | MAPE | RMSE |
| :--- | :--- | :--- |
| **Base OHLCV** | 19.84% | 32.91 |
| **Base + Trend (SMA_20, EMA_20)** | **17.23%** | **28.89** |
| **Base + Momentum (RSI, MACD)** | 22.80% | 36.87 |

*Note: Incorporating too many features (like Bollinger Bands alongside MACD/RSI) caused the model to overfit on the noisy dimensions or fail to converge effectively within the limited epochs, leading to highly volatile error rates.*

### Conclusion

The **Base + Trend (SMA, EMA)** feature set provided the most stable and accurate short-term prediction. Incorporating the 20-day Simple Moving Average and 20-day Exponential Moving Average smoothed the noise in the raw price data, enabling the GRU algorithm to better capture the underlying directional momentum.

## Running the Advanced Production Script

The best-performing feature configuration is encapsulated in a PEP8-compliant, production-ready script.

### Prerequisites

Ensure you have the required packages installed:
```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib pandas-ta
```

### Execution

To run the advanced stock prediction script against the configured list of tickers (`MU, MOD, MRVL, NVDA, S, ZS, VST`):

```bash
python advanced_stock_prediction.py
```

To display matplotlib charts for each prediction, change `plot=False` to `plot=True` inside the `__main__` block.
