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


def get_sentiment_score(texts: List[str]) -> float:
    if not texts:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in texts:
        scores.append(analyzer.polarity_scores(text)["compound"])
    return float(np.mean(scores))
