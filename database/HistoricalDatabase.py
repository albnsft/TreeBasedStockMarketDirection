from datetime import datetime
import pandas as pd
import os
from database.request import download_yf_data


class HistoricalDatabase:
    def __init__(self, tickers: list, name: str = None):
        self.exchange = "NASDAQ"
        self.path_to_data = os.path.abspath(__file__).replace('\database\HistoricalDatabase.py', "\data")
        self.init(tickers, name)

    def transform(self, df):
        df.dropna(inplace=True)
        df['Ewm'] = df.Close.ewm(alpha=0.095, adjust=False).mean()
        return df

    def init(self, tickers: list, name: str = None):
        if os.path.exists(f'{self.path_to_data}/{name}.h5'):
            data = pd.read_hdf(f'{self.path_to_data}/{name}.h5', 'df')
        else:
            data = download_yf_data(tickers, name, self.path_to_data)
        self.data = dict(map(lambda tick: (tick, self.transform(data[tick])), data.columns.levels[0]))
        for ticker in self.data.keys():
            print(f'{ticker} number of instances: {len(self.data[ticker])}')
        self.start_date = dict(map(lambda tick: (tick, self.data[tick].index[0]), data.columns.levels[0]))
        self.end_date = dict(map(lambda tick: (tick, self.data[tick].index[-1]), data.columns.levels[0]))
        self.calendar = dict(map(lambda tick: (tick, list(self.data[tick].index)), data.columns.levels[0]))

    def get_next_timestep(self, timestamp: datetime, ticker: str):
        return self.data[ticker].loc[self.data[ticker].index >= timestamp].index[0]

    def get_last_snapshot(self, timestamp: datetime, ticker: str):
        return self.data[ticker].loc[self.data[ticker].index <= timestamp].iloc[-1]
