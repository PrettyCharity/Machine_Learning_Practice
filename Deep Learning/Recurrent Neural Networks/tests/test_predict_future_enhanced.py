import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict_future_enhanced import FutureStockPredictorEnhanced, GRU


@patch("predict_future_enhanced.yf.download")
@patch("predict_future_enhanced.fetch_yfinance_news")
@patch("predict_future_enhanced.fetch_finviz_news")
@patch("predict_future_enhanced.fetch_googlenews_rss")
@patch("predict_future_enhanced.fetch_macro_geopolitical_news")
@patch("predict_future_enhanced.fetch_analyst_ratings")
def test_fetch_and_prepare_data(
    mock_ratings, mock_macro, mock_gn, mock_finviz, mock_yf, mock_download
):
    # Setup mock returns
    mock_yf.return_value = ["Excellent news!"]
    mock_finviz.return_value = ["Amazing gains in the market!"]
    mock_gn.return_value = ["The stock is performing exceptionally well!"]
    mock_macro.return_value = ["Global trade talks progress."]
    mock_ratings.return_value = 0.8

    # Mock historical data
    dates = pd.date_range("2023-01-01", periods=50)
    data = {
        "Open": [100.0 + i for i in range(50)],
        "High": [102.0 + i for i in range(50)],
        "Low": [98.0 + i for i in range(50)],
        "Close": [101.0 + i for i in range(50)],
        "Volume": [1000 for _ in range(50)],
    }
    df_mock = pd.DataFrame(data, index=dates)
    df_mock.columns = pd.MultiIndex.from_product([df_mock.columns, ["AAPL"]])
    mock_download.return_value = df_mock

    # Test initialization and data preparation
    predictor = FutureStockPredictorEnhanced("AAPL", timesteps=10)
    df = predictor.fetch_and_prepare_data()

    # Verify features and shape
    assert len(df) > 0
    assert "Sentiment_YF" in df.columns
    assert "Sentiment_Finviz" in df.columns
    assert "Sentiment_GN" in df.columns
    assert "Sentiment_Macro" in df.columns
    assert "SMA_20" in df.columns
    assert "EMA_20" in df.columns
    assert "Analyst_Rating" in df.columns

    # Check that sentiment was injected
    assert df["Sentiment_YF"].iloc[-1] > 0.0
    assert df["Sentiment_Finviz"].iloc[-1] > 0.0
    assert df["Sentiment_GN"].iloc[-1] > 0.0
    assert df["Sentiment_Macro"].iloc[-1] > 0.0
    assert df["Analyst_Rating"].iloc[-1] == 0.8


@patch("predict_future_enhanced.FutureStockPredictorEnhanced.fetch_and_prepare_data")
def test_train_and_predict(mock_fetch):
    predictor = FutureStockPredictorEnhanced("AAPL", timesteps=5)

    # Create fake processed data
    dates = pd.date_range("2023-01-01", periods=30)
    data = {
        "Open": [100.0 + i for i in range(30)],
        "High": [102.0 + i for i in range(30)],
        "Low": [98.0 + i for i in range(30)],
        "Close": [101.0 + i for i in range(30)],
        "Volume": [1000 for _ in range(30)],
        "SMA_20": [100.0 + i for i in range(30)],
        "EMA_20": [100.0 + i for i in range(30)],
        "Sentiment_YF": [0.5 for _ in range(30)],
        "Sentiment_Finviz": [0.5 for _ in range(30)],
        "Sentiment_GN": [0.5 for _ in range(30)],
        "Sentiment_Macro": [0.5 for _ in range(30)],
        "Analyst_Rating": [0.8 for _ in range(30)],
        "Date": dates,
    }
    predictor.df = pd.DataFrame(data, index=dates)

    # Test training
    model = predictor.train_model()
    assert isinstance(model, GRU)

    # Test prediction
    last_price, est_price, direction, predicted_prices, future_dates = (
        predictor.predict_future(model, period=3)
    )

    assert last_price == 129.0
    assert direction in ["Upward", "Downward", "Neutral"]
    assert len(predicted_prices) == 3
    assert len(future_dates) == 3


@patch("predict_future_enhanced.FutureStockPredictorEnhanced.fetch_and_prepare_data")
def test_long_term_flatline_safeguard(mock_fetch):
    # This test simulates what the CLI argument parser does to prevent flatlining.
    # While the arg parser itself handles the period limit, we want to ensure
    # the predictor isn't artificially constrained if called via API.
    # We just run a 7-day prediction to verify it works successfully without crashing.
    predictor = FutureStockPredictorEnhanced("AAPL", timesteps=5)

    dates = pd.date_range("2023-01-01", periods=30)
    data = {
        "Open": [100.0 + i for i in range(30)],
        "High": [102.0 + i for i in range(30)],
        "Low": [98.0 + i for i in range(30)],
        "Close": [101.0 + i for i in range(30)],
        "Volume": [1000 for _ in range(30)],
        "SMA_20": [100.0 + i for i in range(30)],
        "EMA_20": [100.0 + i for i in range(30)],
        "Sentiment_YF": [0.5 for _ in range(30)],
        "Sentiment_Finviz": [0.5 for _ in range(30)],
        "Sentiment_GN": [0.5 for _ in range(30)],
        "Sentiment_Macro": [0.5 for _ in range(30)],
        "Analyst_Rating": [0.8 for _ in range(30)],
        "Date": dates,
    }
    predictor.df = pd.DataFrame(data, index=dates)

    model = predictor.train_model()
    _, _, _, predicted_prices, future_dates = predictor.predict_future(model, period=7)

    assert len(predicted_prices) == 7
    assert len(future_dates) == 7
