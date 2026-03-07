import datetime as dt
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import torch
import torch.nn as nn
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

# Ensure VADER lexicon is downloaded for sentiment analysis
nltk.download("vader_lexicon", quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StockWithSentiment:
    """Class to fetch, process stock data, compute technical indicators, and append sentiment."""

    def __init__(
        self,
        ticker: str,
        timesteps: int,
        date_from: dt.datetime = dt.datetime(year=2014, month=1, day=1),
        date_to: dt.datetime = dt.datetime(year=2022, month=1, day=1),
    ):
        self.ticker = ticker.upper()
        self.timesteps = timesteps
        self.date_from = date_from
        self.date_to = date_to

        self.mms_features = MinMaxScaler(feature_range=(0, 1))
        self.mms_target = MinMaxScaler(feature_range=(0, 1))

        self.df = pd.DataFrame()
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.x_train = torch.empty(0)
        self.x_test = torch.empty(0)
        self.y_train = torch.empty(0)
        self.y_test = torch.empty(0)

    def _fetch_sentiment(self) -> float:
        """Fetch news for the ticker and compute the average compound sentiment score."""
        analyzer = SentimentIntensityAnalyzer()
        try:
            # yfinance.Ticker('AAPL').news returns a list of dictionaries.
            # In older versions it was flat, in newer versions the info is often inside 'content'
            news = yf.Ticker(self.ticker).news
        except Exception as e:
            print(f"Warning: Failed to fetch news for {self.ticker}: {e}")
            return 0.0

        sentiment_scores = []
        for item in news:
            # Handle potential nested 'content' structure or flat structure
            if "content" in item and isinstance(item["content"], dict):
                content = item["content"]
            else:
                content = item

            title = content.get("title", "")
            summary = content.get("summary", "")
            text = f"{title} {summary}".strip()

            if text:
                score = analyzer.polarity_scores(text)["compound"]
                sentiment_scores.append(score)

        if not sentiment_scores:
            return 0.0

        return float(np.mean(sentiment_scores))

    def get_data(self) -> pd.DataFrame:
        """Fetch stock data, calculate SMA, EMA, and simulate historical Sentiment."""
        df = yf.download(
            self.ticker,
            start=self.date_from,
            end=self.date_to,
            auto_adjust=False,
            progress=False,
        )
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        df.columns.name = None
        df.reset_index(inplace=True)

        # Compute Technical Indicators (Best performing feature set)
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=20, append=True)
        df = df.dropna()

        # Compute average current sentiment from available yfinance news.
        avg_sentiment = self._fetch_sentiment()

        # We use the current average sentiment as a baseline anchor and add random normal
        # noise to simulate daily fluctuating historical sentiment for integration testing.
        df["Sentiment"] = avg_sentiment + np.random.normal(0, 0.1, size=len(df))
        df["Sentiment"] = df["Sentiment"].clip(-1, 1)

        return df

    def _split_data(self) -> None:
        """Split data into 70% train and 30% test sets."""
        self.df = self.get_data()
        split_idx = int(self.df.shape[0] * 0.7)
        self.train_data = self.df.iloc[:split_idx, :]
        self.test_data = self.df.iloc[split_idx:, :]

    def _batch_train_data(self, features: List[str]) -> None:
        """Batch and scale training data."""
        target_train = self.train_data.loc[:, ["Open"]].values
        self.mms_target.fit(target_train)

        training_set = self.train_data.loc[:, features].values
        training_set_scaled = self.mms_features.fit_transform(training_set)

        t, n_samples = self.timesteps, training_set.shape[0]
        x_train, y_train = [], []
        open_idx = features.index("Open")

        for i in range(t, n_samples):
            x_train.append(training_set_scaled[i - t : i, :])
            y_train.append(training_set_scaled[i, open_idx])

        self.y_train = np.array(y_train).reshape((len(y_train), 1))
        x_train_arr = np.array(x_train)
        self.x_train = np.reshape(
            x_train_arr, (x_train_arr.shape[0], x_train_arr.shape[1], len(features))
        )

    def _batch_test_data(self, features: List[str]) -> None:
        """Batch and scale testing data."""
        data_for_test = pd.concat(
            (self.train_data.tail(self.timesteps), self.test_data), axis=0
        )
        inputs = data_for_test.loc[:, features].values
        inputs = self.mms_features.transform(inputs)

        t, n_samples = self.timesteps, inputs.shape[0]
        x_test = []
        for i in range(t, n_samples):
            x_test.append(inputs[i - t : i, :])

        x_test_arr = np.array(x_test)
        self.x_test = np.reshape(
            x_test_arr, (x_test_arr.shape[0], x_test_arr.shape[1], len(features))
        )
        self.y_test = self.test_data.loc[:, "Open"].values.reshape(len(x_test_arr), 1)

    def _tensorize(self) -> None:
        """Convert numpy arrays to PyTorch tensors."""
        self.x_train = torch.from_numpy(self.x_train).to(
            device=DEVICE, dtype=torch.float
        )
        self.x_test = torch.from_numpy(self.x_test).to(device=DEVICE, dtype=torch.float)
        self.y_train = torch.from_numpy(self.y_train).to(
            device=DEVICE, dtype=torch.float
        )
        self.y_test = torch.from_numpy(self.y_test.copy()).to(
            device=DEVICE, dtype=torch.float
        )

    def process(
        self, features: List[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the full data processing pipeline."""
        if features is None:
            features = ["Open"]
        self._split_data()
        self._batch_train_data(features)
        self._batch_test_data(features)
        self._tensorize()
        return self.x_train, self.x_test, self.y_train, self.y_test

    def visualize(self, y_pred: np.ndarray, stock_name: str = "") -> None:
        """Plot actual vs predicted stock prices."""
        plt.figure(figsize=(14, 5))
        plt.plot(
            self.df["Date"], self.df["Open"], color="red", label="Real Stock Price"
        )
        plt.plot(
            self.test_data["Date"],
            y_pred,
            color="blue",
            label="Predicted Stock Price",
        )
        plt.title(f"{stock_name} Stock Prediction (Sentiment + Indicators)")
        plt.xlabel("Time")
        plt.ylabel("Stock Price ($)")
        plt.legend()
        plt.show()


class GRU(nn.Module):
    """GRU model for time-series prediction."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        out, _ = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


def train_and_evaluate(
    model_class: type,
    ticker: str,
    features: List[str],
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 150,
    lr: float = 0.005,
    dropout: float = 0.2,
    plot: bool = False,
) -> Tuple[float, float]:
    """Train a model on a given ticker and return accuracy metrics."""
    data = StockWithSentiment(ticker, timesteps=60)
    x_train, x_test, y_train, y_test_true = data.process(features)

    model = model_class(
        input_dim=len(features),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        y_pred = data.mms_target.inverse_transform(y_test_pred.cpu().numpy())

    y_true_np = y_test_true.cpu().numpy()
    mape = np.mean(np.abs((y_true_np - y_pred) / y_true_np)) * 100
    rmse = np.sqrt(np.mean((y_true_np - y_pred) ** 2))

    if plot:
        data.visualize(y_pred.ravel(), stock_name=ticker)

    return mape, rmse


if __name__ == "__main__":
    # Base + SMA_20 + EMA_20 + Sentiment features
    features_list = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_20",
        "EMA_20",
        "Sentiment",
    ]
    tickers_list = ["MU", "MOD", "MRVL", "NVDA", "S", "ZS", "VST"]

    for t in tickers_list:
        print(f"\\n{'=' * 40}\\nProcessing {t}\\n{'=' * 40}")
        try:
            mape_score, rmse_score = train_and_evaluate(
                GRU,
                t,
                features_list,
                hidden_dim=64,
                num_layers=2,
                epochs=150,
                lr=0.005,
                dropout=0.2,
                plot=False,  # Set to True to display plots
            )
            print(f"MAPE: {mape_score:.2f}%, RMSE: {rmse_score:.2f}")
        except Exception as e:
            print(f"Failed to process {t}: {e}")
