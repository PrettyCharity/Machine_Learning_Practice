from typing import List

import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import feedparser

nltk.download("vader_lexicon", quiet=True)


def fetch_yfinance_news(ticker: str) -> List[str]:
    try:
        news = yf.Ticker(ticker).news
    except Exception:
        return []

    texts = []
    for item in news:
        if "content" in item and isinstance(item["content"], dict):
            content = item["content"]
        else:
            content = item

        title = content.get("title", "")
        summary = content.get("summary", "")
        text = f"{title} {summary}".strip()
        if text:
            texts.append(text)
    return texts


def fetch_finviz_news(ticker: str) -> List[str]:
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(req.content, "html.parser")
        news_table = soup.find(id="news-table")
        texts = []
        if news_table:
            for row in news_table.find_all("tr"):
                a = row.a
                if a:
                    texts.append(a.text)
        return texts
    except Exception:
        return []


def fetch_googlenews_rss(ticker: str) -> List[str]:
    url = (
        f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        feed = feedparser.parse(url)
        texts = []
        for entry in feed.entries[:20]:
            texts.append(entry.title)
        return texts
    except Exception:
        return []


def fetch_macro_geopolitical_news() -> List[str]:
    """
    Fetches general macroeconomic and geopolitical news that might affect the stock market.
    """
    import urllib.parse
    query = "geopolitics OR tariffs OR trade war OR economy OR federal reserve"
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
        texts = []
        for entry in feed.entries[:20]:
            texts.append(entry.title)
        return texts
    except Exception:
        return []


def get_sentiment_score(texts: List[str]) -> float:
    if not texts:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in texts:
        scores.append(analyzer.polarity_scores(text)["compound"])
    return float(np.mean(scores))

def fetch_analyst_ratings(ticker: str) -> float:
    """
    Fetches analyst recommendations summary from Yahoo Finance (powered by LSEG/Refinitiv).
    Maps text categories to a numeric scale (-1.0 to 1.0):
    Strong Buy = 1.0, Buy = 0.5, Hold = 0.0, Sell = -0.5, Strong Sell = -1.0
    Returns the weighted average score.
    """
    try:
        t = yf.Ticker(ticker)
        df = t.recommendations_summary
        if df is None or df.empty:
            return 0.0

        # Get the current period (usually index 0, labeled '0m')
        current = df.iloc[0]

        strong_buy = current.get('strongBuy', 0)
        buy = current.get('buy', 0)
        hold = current.get('hold', 0)
        sell = current.get('sell', 0)
        strong_sell = current.get('strongSell', 0)

        total = strong_buy + buy + hold + sell + strong_sell
        if total == 0:
            return 0.0

        score = (strong_buy * 1.0 + buy * 0.5 + hold * 0.0 + sell * -0.5 + strong_sell * -1.0) / total
        return float(score)
    except Exception:
        return 0.0
