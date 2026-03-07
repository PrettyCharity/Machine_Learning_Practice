import datetime as dt
import sys
import os
from unittest.mock import patch

# Add parent directory to path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Deep Learning', 'Recurrent Neural Networks')))

import pytest
import numpy as np
import pandas as pd
import torch
from lstm_financial_data_prediction_pytorch import Stock, LSTM, GRU

@pytest.fixture
def sample_features():
    return ["Open", "High", "Low", "Close", "Volume"]

@pytest.fixture
def mock_yfinance_data():
    dates = pd.date_range(start="2021-01-01", end="2021-05-01", freq="B")
    data = {
        "Open": np.random.uniform(100, 150, size=len(dates)),
        "High": np.random.uniform(110, 160, size=len(dates)),
        "Low": np.random.uniform(90, 140, size=len(dates)),
        "Close": np.random.uniform(100, 150, size=len(dates)),
        "Adj Close": np.random.uniform(100, 150, size=len(dates)),
        "Volume": np.random.randint(1000000, 5000000, size=len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df

@pytest.fixture
def stock_instance():
    # Use a short timeframe to speed up the test
    start_date = dt.datetime(year=2021, month=1, day=1)
    end_date = dt.datetime(year=2021, month=5, day=1)
    return Stock("AAPL", timesteps=10, date_from=start_date, date_to=end_date)

@patch("lstm_financial_data_prediction_pytorch.yf.download")
def test_stock_get_data(mock_download, stock_instance, mock_yfinance_data):
    mock_download.return_value = mock_yfinance_data.copy()

    df = stock_instance.get_data()
    assert not df.empty
    assert "Date" in df.columns
    assert "Open" in df.columns
    mock_download.assert_called_once()

@patch("lstm_financial_data_prediction_pytorch.yf.download")
def test_stock_process(mock_download, stock_instance, sample_features, mock_yfinance_data):
    mock_download.return_value = mock_yfinance_data.copy()

    x_train, x_test, y_train, y_test = stock_instance.process(features=sample_features)

    assert isinstance(x_train, torch.Tensor)
    assert isinstance(y_train, torch.Tensor)
    assert isinstance(x_test, torch.Tensor)
    assert isinstance(y_test, torch.Tensor)

    # Check dimensions
    # x should be (batch_size, timesteps, num_features)
    assert x_train.dim() == 3
    assert x_train.size(1) == stock_instance.timesteps
    assert x_train.size(2) == len(sample_features)

    # y should be (batch_size, 1)
    assert y_train.dim() == 2
    assert y_train.size(1) == 1

    # Test set constraints
    assert x_test.dim() == 3
    assert y_test.dim() == 2
    assert x_test.size(0) == y_test.size(0)

def test_lstm_model(sample_features):
    batch_size = 16
    timesteps = 10
    input_dim = len(sample_features)

    model = LSTM(input_dim=input_dim, hidden_dim=32, num_layers=2)
    # create dummy input tensor
    dummy_input = torch.randn(batch_size, timesteps, input_dim)

    output = model(dummy_input)

    assert output.dim() == 2
    assert output.size(0) == batch_size
    assert output.size(1) == 1

def test_gru_model(sample_features):
    batch_size = 16
    timesteps = 10
    input_dim = len(sample_features)

    model = GRU(input_dim=input_dim, hidden_dim=32, num_layers=2)
    # create dummy input tensor
    dummy_input = torch.randn(batch_size, timesteps, input_dim)

    output = model(dummy_input)

    assert output.dim() == 2
    assert output.size(0) == batch_size
    assert output.size(1) == 1
