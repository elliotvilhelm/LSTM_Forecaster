import os

from data_collection.yfinance_collector import download_data
from config import TICKERS

if not os.path.isdir("./data/"):
    os.mkdir("./data/")

# download_data(TICKERS)
download_data(TICKERS, interval='5m')
download_data(TICKERS, interval='15m')
