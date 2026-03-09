import argparse
import datetime as dt
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from news_fetchers import (
    fetch_yfinance_news,
    fetch_finviz_news,
    fetch_googlenews_rss,
    get_sentiment_score,
    fetch_analyst_ratings,
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
            "Analyst_Rating",
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

        print("Fetching LSEG/Refinitiv Analyst Ratings...")
        current_rating = fetch_analyst_ratings(self.ticker)

        # Simulate historical ratings
        df["Analyst_Rating"] = current_rating + np.random.normal(0, 0.05, size=len(df))
        df["Analyst_Rating"] = df["Analyst_Rating"].clip(-1, 1)
        df.loc[df.index[-1], "Analyst_Rating"] = current_rating

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

    def predict_future(
        self, model: GRU, period: int
    ) -> Tuple[float, float, str, List[float], List[pd.Timestamp]]:
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
                close_idx = self.features.index("Close")
                high_idx = self.features.index("High")
                low_idx = self.features.index("Low")

                # To prevent completely flat autoregressive loops, carry the predicted
                # Open value lightly to other core price indicators as a basic simulation.
                # In a robust production environment, a multi-output GRU predicting all 4 would be ideal.
                last_row[open_idx] = next_pred_scaled.item()
                last_row[close_idx] = next_pred_scaled.item()
                last_row[high_idx] = next_pred_scaled.item()
                last_row[low_idx] = next_pred_scaled.item()

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

        # Generate future dates for the prediction
        last_date = self.df["Date"].iloc[-1]
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=period
        )

        return (
            last_real_price,
            final_price,
            direction,
            predicted_prices,
            future_dates.tolist(),
        )


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
        help="Number of days to predict into the future (1-7).",
    )

    args = parser.parse_args()

    if not (1 <= args.period <= 7):
        print("Error: Period must be between 1 and 7 days. Long-term predictions (> 7 days) lead to flatlining due to static autoregressive momentum features.")
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
    last_price, est_price, direction, predicted_prices, future_dates = (
        predictor.predict_future(model, args.period)
    )

    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Ticker:           {args.ticker.upper()}")
    print(f"Period:           {args.period} days")
    print(f"Last Known Price: ${last_price:.2f}")
    print(f"Estimated Price:  ${est_price:.2f}")
    print(f"Overall Direction:{direction}")
    print("=" * 50 + "\n")

    print(f"Generating visualization...")
    historical_days = 60
    hist_dates = predictor.df["Date"].iloc[-historical_days:].tolist()
    hist_prices = predictor.df["Open"].iloc[-historical_days:].tolist()

    plt.figure(figsize=(10, 6))
    plt.plot(
        hist_dates,
        hist_prices,
        label="Historical Open Prices (last 60 days)",
        color="blue",
    )

    # Connect the last historical point to the first prediction point
    plot_dates = [hist_dates[-1]] + future_dates
    plot_prices = [hist_prices[-1]] + predicted_prices
    plt.plot(
        plot_dates,
        plot_prices,
        label=f"Predicted Open Prices ({args.period} days)",
        color="orange",
        linestyle="--",
    )

    plt.title(
        f"{args.ticker.upper()} - {historical_days}-Day History & {args.period}-Day Prediction"
    )
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)

    out_file = f"{args.ticker.upper()}_prediction.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}\n")


if __name__ == "__main__":
    main()
