import argparse
import datetime as dt
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from news_fetchers import (
    fetch_yfinance_news,
    fetch_finviz_news,
    fetch_googlenews_rss,
    get_sentiment_score,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRU(nn.Module):
    """GRU model for time-series prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
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


class FutureStockPredictorEnhanced:
    def __init__(self, ticker: str, timesteps: int = 60):
        self.ticker = ticker.upper()
        self.timesteps = timesteps
        self.date_to = dt.datetime.today()
        # Fetch enough historical data for technical indicators and training (e.g., 5 years back)
        self.date_from = self.date_to - dt.timedelta(days=365 * 5)

        self.mms_features = MinMaxScaler(feature_range=(0, 1))
        self.mms_target = MinMaxScaler(feature_range=(0, 1))

        self.features = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA_20",
            "EMA_20",
            "Sentiment_YF",
            "Sentiment_Finviz",
            "Sentiment_GN",
        ]

    def _fetch_sentiments(self) -> Dict[str, float]:
        """Fetch news from multiple sources and compute the average compound sentiment scores."""
        print("Fetching Yahoo Finance news...")
        yf_news = fetch_yfinance_news(self.ticker)
        print("Fetching Finviz news...")
        finviz_news = fetch_finviz_news(self.ticker)
        print("Fetching Google News...")
        gn_news = fetch_googlenews_rss(self.ticker)

        return {
            "yf": get_sentiment_score(yf_news),
            "finviz": get_sentiment_score(finviz_news),
            "gn": get_sentiment_score(gn_news),
        }

    def fetch_and_prepare_data(self) -> pd.DataFrame:
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

        df.ta.sma(length=20, append=True)
        df.ta.ema(length=20, append=True)
        df = df.dropna()

        # Fetch current sentiment from multiple sources
        sentiments = self._fetch_sentiments()

        # Simulate the historical array around the baseline
        for source, key in [
            ("yf", "Sentiment_YF"),
            ("finviz", "Sentiment_Finviz"),
            ("gn", "Sentiment_GN"),
        ]:
            df[key] = sentiments[source] + np.random.normal(0, 0.1, size=len(df))
            df[key] = df[key].clip(-1, 1)
            # Set the current latest row to the exact current sentiment
            df.loc[df.index[-1], key] = sentiments[source]

        self.df = df
        return df

    def train_model(self) -> GRU:
        target_data = self.df.loc[:, ["Open"]].values
        self.mms_target.fit(target_data)

        feature_data = self.df.loc[:, self.features].values
        feature_data_scaled = self.mms_features.fit_transform(feature_data)

        x_train, y_train = [], []
        open_idx = self.features.index("Open")

        for i in range(self.timesteps, len(feature_data_scaled)):
            x_train.append(feature_data_scaled[i - self.timesteps : i, :])
            y_train.append(feature_data_scaled[i, open_idx])

        x_train = np.array(x_train)
        y_train = np.array(y_train).reshape(-1, 1)

        x_train_t = torch.from_numpy(x_train).to(device=DEVICE, dtype=torch.float)
        y_train_t = torch.from_numpy(y_train).to(device=DEVICE, dtype=torch.float)

        model = GRU(input_dim=len(self.features)).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        # Train for a moderate amount of epochs to capture recent trends
        model.train()
        for _ in range(100):
            preds = model(x_train_t)
            loss = criterion(preds, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def predict_future(self, model: GRU, period: int) -> Tuple[float, float, str]:
        model.eval()

        # Start with the last known window of scaled features
        last_window = self.df.loc[:, self.features].tail(self.timesteps).values
        last_window_scaled = self.mms_features.transform(last_window)

        current_input = (
            torch.from_numpy(last_window_scaled)
            .unsqueeze(0)
            .to(device=DEVICE, dtype=torch.float)
        )

        last_real_price = float(self.df["Open"].iloc[-1])
        predicted_prices = []

        # Iteratively predict the next day, and use it to predict the day after
        with torch.no_grad():
            for _ in range(period):
                next_pred_scaled = model(current_input)
                next_pred = self.mms_target.inverse_transform(
                    next_pred_scaled.cpu().numpy()
                )[0][0]
                predicted_prices.append(next_pred)

                # Get the last row from the scaled input tensor
                last_row = current_input[0, -1, :].clone().cpu().numpy()

                # Update the 'Open' index with the newly predicted scaled value
                open_idx = self.features.index("Open")
                last_row[open_idx] = next_pred_scaled.item()

                # Shift the window: remove the oldest day, append the new synthetic day
                new_row_tensor = (
                    torch.from_numpy(last_row).unsqueeze(0).unsqueeze(0).to(DEVICE)
                )
                current_input = torch.cat(
                    (current_input[:, 1:, :], new_row_tensor), dim=1
                )

        final_price = predicted_prices[-1]

        if final_price > last_real_price:
            direction = "Upward"
        elif final_price < last_real_price:
            direction = "Downward"
        else:
            direction = "Neutral"

        return last_real_price, final_price, direction


def main():
    parser = argparse.ArgumentParser(
        description="Predict future stock direction and price using GRU and Multi-Source Sentiment Analysis."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="The stock ticker symbol (e.g., AAPL).",
    )
    parser.add_argument(
        "--period",
        type=int,
        required=True,
        help="Number of days to predict into the future (1-30).",
    )

    args = parser.parse_args()

    if not (1 <= args.period <= 30):
        print("Error: Period must be between 1 and 30 days.")
        return

    print(f"Fetching data and calculating features for {args.ticker.upper()}...")
    predictor = FutureStockPredictorEnhanced(args.ticker)

    try:
        predictor.fetch_and_prepare_data()
    except Exception as e:
        print(f"Failed to fetch data for ticker {args.ticker}: {e}")
        return

    print(
        f"Training GRU model with {len(predictor.features)} features (including Multi-Source Sentiment)..."
    )
    model = predictor.train_model()

    print(f"Simulating future predictions for {args.period} days...")
    last_price, est_price, direction = predictor.predict_future(model, args.period)

    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Ticker:           {args.ticker.upper()}")
    print(f"Period:           {args.period} days")
    print(f"Last Known Price: ${last_price:.2f}")
    print(f"Estimated Price:  ${est_price:.2f}")
    print(f"Overall Direction:{direction}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
