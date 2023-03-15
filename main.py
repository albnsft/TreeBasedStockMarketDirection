from utils.utils import split_dates, WalkForward
from database.HistoricalDatabase import HistoricalDatabase
from models.Model import Classifier
from env.FinancialEnvironment import FinancialEnvironment
from datetime import timedelta

import pandas as pd
pd.options.mode.chained_assignment = None  # Ignore Setting With Copy Warning


if __name__ == '__main__':

    tickers = ['AAPL', 'META']
    name = 'tech_stocks'
    step_size_in_hour = 24
    trading_windows_in_days = [1, 5, 10, 15, 21]
    database = HistoricalDatabase(tickers, name)

    models = ['RF', 'XGB']

    for ticker in tickers:
        print(f"***************************{ticker}***************************")
        train_dates, test_dates = split_dates(split=0.90, start_date=database.start_date[ticker], end_date=database.end_date[ticker])
        wf = WalkForward(start_date=train_dates.start, #database.start_date[ticker]
                         end_date=train_dates.end, #database.end_date[ticker]
                         val_size=0.10, #test_size=0.10
                         n_walks=5)
        for trading_window in trading_windows_in_days:
            print(f"********************{trading_window}d trading******************")
            env = FinancialEnvironment(
                database=database,
                ticker=ticker,
                step_size=timedelta(hours=step_size_in_hour*trading_window),
            )
            for model in models:
                print(f"***************************{model}***************************")
                clf_baseline = Classifier(env, wf, model_type=model, verbose=True, test_dates=test_dates)
                clf_baseline.learn()
        print('end')





