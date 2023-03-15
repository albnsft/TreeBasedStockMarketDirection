import yfinance as yf

def download_yf_data(tickers: list, name: str = None, path_do_data: str = None, period: str = '20y'):
    #Requesting from yahoo finance Open, High, Low, Close, Volume data for a list of tickers over the last year
    tickers = " ".join(tickers)
    field = list("High,Low,Close,Volume".split(','))
    print(f'Downloading daily data {tickers} from Yfinance over the {period} last years')
    df = yf.download(tickers, period=period)[field]
    df = df.T.swaplevel(1, 0).T
    file_path = f'{path_do_data}/{name}.h5'
    df.to_hdf(file_path, 'df')
    return df