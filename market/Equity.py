import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class EquityData:
    def __init__(self, filename=None, symbol=None):
        """
        filename: .csv file of OHLC values
        symbol: ticker
        """
        self.symbol = symbol
        if filename:
            self.data = self.load_csv(filename)
        else:
            self.data = None

    def load_csv(self, filename):
        """
        filename: .csv file of OHLC values
        """
        return pd.read_csv(filename)

    def close(self):
        return self.data['Close']

    def date(self):
        return self.data['Date']

    def plot(self, attr='Close'):
        df = pd.DataFrame()
        date_time = pd.to_datetime(self.date())
        df['Close'] = self.close()
        df = df.set_index(date_time)
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # to get a tick every 15 minutes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_tick_params(rotation=45)
        fig.subplots_adjust(bottom=0.3)
        plt.plot(df)
        plt.show()
